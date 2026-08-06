"""Decision Transformer adapter implementation.

Reference: "Decision Transformer: Reinforcement Learning via Sequence Modeling" (Chen et al., 2021)

Decision Transformer treats RL as a sequence modeling problem:
- Uses GPT-style transformer to predict actions
- Conditions on (return-to-go, state, action) tokens
- No explicit dynamics model - pure supervised learning of trajectory
- Can be used as a world model by predicting future states

Key characteristics:
- Transformer-based (no recurrent state needed)
- Token-based: encodes (R, s, a) as discrete tokens
- Autoregressive prediction of actions
- Can be used as world model by predicting state tokens
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig
from world_model_lens.core.types import WorldModelFamily


class DecisionTransformerEmbedding(nn.Module):
    """Embedding layer for Decision Transformer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int,
        max_len: int = 30,
    ):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.action_embedding = nn.Linear(action_dim, d_model)
        self.return_embedding = nn.Linear(1, d_model)
        self.timestep_embedding = nn.Embedding(max_len, d_model)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        return_emb = self.return_embedding(returns_to_go.unsqueeze(-1))
        time_emb = self.timestep_embedding(timesteps)

        embeddings = torch.stack([return_emb, state_emb, action_emb], dim=2)
        embeddings = embeddings + time_emb.unsqueeze(2)
        return embeddings.flatten(1, 2)


class DecisionTransformerBlock(nn.Module):
    """Transformer block for Decision Transformer."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """GPT-style transformer for Decision Transformer."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_head: int,
        state_dim: int,
        action_dim: int,
        max_len: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = DecisionTransformerEmbedding(state_dim, action_dim, d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                DecisionTransformerBlock(d_model, n_head, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_dim)
        self.state_head = nn.Linear(d_model, state_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(states, actions, returns_to_go, timesteps)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln(x)

        action_logits = self.action_head(x)
        state_predictions = self.state_head(x)

        return action_logits, state_predictions


class DecisionTransformerAdapter(BaseModelAdapter):
    """Adapter for Decision Transformer.

    Transformer-based RL that treats trajectory as a sequence.
    Can be used as a world model by predicting state transitions.

    Architecture:
    - GPT-style transformer
    - Token embeddings for (return, state, action)
    - Autoregressive action prediction
    - Can predict future states for world modeling
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config
        self.state_dim = config.d_obs

        self.transformer = DecisionTransformer(
            d_model=config.d_embed,
            n_layers=config.n_layers,
            n_head=config.n_heads,
            state_dim=self.state_dim,
            action_dim=config.d_action,
            max_len=config.imagination_horizon,
        )

        self._device = torch.device("cpu")
        self._timestep = 0

    @property
    def hook_point_names(self) -> List[str]:
        names = ["transformer"]
        for i in range(self.config.n_layers):
            names.append(f"transformer_block_{i}")
        return names

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.DECISION_TRANSFORMER

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize a state tensor to [B, T, d_obs] for DT forward passes."""
        if state.dim() > 2:
            state = state.flatten(start_dim=1)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        return state

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the raw observation-space state as the DT latent/state carrier."""
        del h_prev
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs, obs

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the next state representation in observation space."""
        del h
        z = self._normalize_state(z)
        batch_size = z.shape[0]

        if action is None:
            action = torch.zeros(batch_size, 1, self.config.d_action, device=z.device)
        elif action.dim() == 1:
            action = action.unsqueeze(0).unsqueeze(0)
        elif action.dim() == 2:
            action = action.unsqueeze(1)

        returns = torch.zeros(batch_size, 1, device=z.device)
        timesteps = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)

        _, state_preds = self.transformer(z, action, returns, timesteps)
        return state_preds[:, 1]

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Decision Transformer uses the current observation-space state as its prior."""
        return h if h.dim() > 1 else h.unsqueeze(0)

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Decision Transformer state carriers are continuous representations."""
        del temperature, sample
        return logits_or_repr

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> None:
        """No decoder in Decision Transformer."""
        del h, z
        return None

    def actor_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict action from the current state token."""
        del h
        z = self._normalize_state(z)
        batch_size = z.shape[0]

        action = torch.zeros(batch_size, 1, self.config.d_action, device=z.device)
        returns = torch.zeros(batch_size, 1, device=z.device)
        timesteps = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)

        action_logits, _ = self.transformer(z, action, returns, timesteps)
        return action_logits[:, 1]

    def predict_reward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        """Decision Transformer doesn't predict rewards directly."""
        del h, z
        return None

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and latent state in observation space."""
        if device is None:
            device = self._device
        state = torch.zeros(batch_size, self.state_dim, device=device)
        return state, state

    def to(self, device: torch.device) -> "DecisionTransformerAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "DecisionTransformerAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "DecisionTransformerAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="decision_transformer",
    family=WorldModelFamily.DECISION_TRANSFORMER,
    description="Decision Transformer: RL via Sequence Modeling",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(DecisionTransformerAdapter)
