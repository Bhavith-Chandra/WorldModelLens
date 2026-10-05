"""Decision Transformer architecture.

This module follows the continuous-control implementation released with
"Decision Transformer: Reinforcement Learning via Sequence Modeling"
(Chen et al., 2021): trajectories are represented as interleaved
``(return-to-go, state, action)`` tokens and processed by a causal GPT-style
transformer. Actions are predicted from state-token outputs, while next states
and returns are predicted from action-token outputs.

Reference implementation: https://github.com/kzl/decision-transformer
"""

from typing import Optional

import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import (
    AdapterConfig,
    BaseModelAdapter,
    WorldModelCapabilities,
)
from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily


class DecisionTransformerEmbedding(nn.Module):
    """Embed and interleave return, state, and action trajectory tokens."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int,
        max_timestep: int = 4096,
    ) -> None:
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.action_embedding = nn.Linear(action_dim, d_model)
        self.return_embedding = nn.Linear(1, d_model)
        self.timestep_embedding = nn.Embedding(max_timestep, d_model)
        self.embed_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        time_emb = self.timestep_embedding(timesteps)
        state_emb = self.state_embedding(states) + time_emb
        action_emb = self.action_embedding(actions) + time_emb
        return_emb = self.return_embedding(returns_to_go) + time_emb

        # [B, T, 3, D] -> [B, 3T, D], ordered R_1, s_1, a_1, ...
        embeddings = torch.stack((return_emb, state_emb, action_emb), dim=2)
        return self.embed_ln(embeddings.flatten(1, 2))


class DecisionTransformerBlock(nn.Module):
    """Pre-normalized causal transformer block."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = self.ln1(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ffn(self.ln2(x))


class DecisionTransformer(nn.Module):
    """Causal GPT-style Decision Transformer for continuous control."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_head: int,
        state_dim: int,
        action_dim: int,
        max_len: int = 30,
        max_timestep: int = 4096,
        dropout: float = 0.1,
        action_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.embedding = DecisionTransformerEmbedding(
            state_dim,
            action_dim,
            d_model,
            max_timestep=max_timestep,
        )
        self.blocks = nn.ModuleList(
            [
                DecisionTransformerBlock(d_model, n_head, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.predict_state = nn.Linear(d_model, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(d_model, action_dim),
            nn.Tanh() if action_tanh else nn.Identity(),
        )
        self.predict_return = nn.Linear(d_model, 1)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # MultiheadAttention boolean masks use True for disallowed positions.
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next states, actions, and returns for a trajectory."""
        if states.dim() != 3 or actions.dim() != 3:
            raise ValueError("states and actions must have shape [batch, time, features]")
        if states.shape[:2] != actions.shape[:2] or states.shape[:2] != timesteps.shape:
            raise ValueError("states, actions, and timesteps must share batch/time dimensions")

        batch_size, seq_length = states.shape[:2]
        if seq_length > self.max_len:
            states = states[:, -self.max_len :]
            actions = actions[:, -self.max_len :]
            returns_to_go = returns_to_go[:, -self.max_len :]
            timesteps = timesteps[:, -self.max_len :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len :]
            seq_length = self.max_len

        x = self.embedding(states, actions, returns_to_go, timesteps)
        causal_mask = self._causal_mask(3 * seq_length, x.device)

        key_padding_mask = None
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            valid = torch.stack((valid, valid, valid), dim=2).flatten(1, 2)
            key_padding_mask = ~valid

        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)
        x = self.final_ln(x)

        # [B, 3T, D] -> [B, 3, T, D]
        x = x.reshape(batch_size, seq_length, 3, -1).permute(0, 2, 1, 3)
        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the final action prediction from a trajectory context."""
        state_preds, action_preds, return_preds = self(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
        )
        del state_preds, return_preds
        return action_preds[:, -1]


class DecisionTransformerAdapter(BaseModelAdapter):
    """Adapter around the source-faithful Decision Transformer token path.

    Decision Transformer is a return-conditioned sequence policy rather than a
    latent dynamics model. ``forward_trajectory`` is the canonical interface;
    the standard adapter methods provide one-step compatibility.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self.config = config
        self.state_dim = config.d_obs
        self.context_length = max(1, config.imagination_horizon)

        self.transformer = DecisionTransformer(
            d_model=config.d_embed,
            n_layers=config.n_layers,
            n_head=config.n_heads,
            state_dim=self.state_dim,
            action_dim=config.d_action,
            max_len=self.context_length,
            max_timestep=getattr(config, "max_timestep", 4096),
            action_tanh=getattr(config, "decision_transformer_action_tanh", True),
        )
        self._capabilities = WorldModelCapabilities(
            has_actor=True,
            uses_actions=True,
            is_rl_trained=True,
        )
        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> list[str]:
        names = ["transformer.embedding", "transformer.final_ln"]
        names.extend(f"transformer.blocks.{i}" for i in range(self.config.n_layers))
        return names

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.DECISION_TRANSFORMER

    def forward_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the source-aligned causal trajectory model."""
        return self.transformer(
            states,
            actions,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
        )

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() > 3:
            state = state.flatten(start_dim=2)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
        return state

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        del h
        states = self._normalize_state(z)
        batch_size, seq_length = states.shape[:2]
        if action is None:
            actions = torch.zeros(
                batch_size,
                seq_length,
                self.config.d_action,
                device=states.device,
                dtype=states.dtype,
            )
        elif action.dim() == 1:
            actions = action.view(1, 1, -1)
        elif action.dim() == 2:
            actions = action.unsqueeze(1)
        else:
            actions = action

        returns = torch.zeros(batch_size, seq_length, 1, device=states.device, dtype=states.dtype)
        timesteps = torch.arange(seq_length, device=states.device).expand(batch_size, -1)
        state_preds, _, _ = self.forward_trajectory(states, actions, returns, timesteps)
        return state_preds[:, -1]

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        return h if h.dim() > 1 else h.unsqueeze(0)

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        del temperature, sample
        return logits_or_repr

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del h
        states = self._normalize_state(z)
        batch_size, seq_length = states.shape[:2]
        actions = torch.zeros(
            batch_size,
            seq_length,
            self.config.d_action,
            device=states.device,
            dtype=states.dtype,
        )
        returns = torch.zeros(batch_size, seq_length, 1, device=states.device, dtype=states.dtype)
        timesteps = torch.arange(seq_length, device=states.device).expand(batch_size, -1)
        return self.transformer.get_action(states, actions, returns, timesteps)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._device
        state = torch.zeros(batch_size, self.state_dim, device=device)
        return state, state

    def to(self, device: torch.device) -> "DecisionTransformerAdapter":
        super().to(device=device)
        self._device = device
        return self

    def eval(self) -> "DecisionTransformerAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "DecisionTransformerAdapter":
        super().train(mode)
        return self


register(
    name="decision_transformer",
    family=WorldModelFamily.DECISION_TRANSFORMER,
    description="Decision Transformer: RL via causal sequence modeling",
    supports_rl=True,
    supports_video=False,
    supports_planning=False,
)(DecisionTransformerAdapter)
