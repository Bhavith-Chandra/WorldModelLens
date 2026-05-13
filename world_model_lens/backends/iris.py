"""IRIS adapter implementation.

IRIS uses VQVAE to discretize observations, then a GPT-style
transformer predicts next tokens.
"""

from typing import Any, Dict, List, Optional, Tuple, cast
import math
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig


class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_flat = z.flatten(start_dim=1)
        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q = z + (z_q - z).detach()
        return z_q, min_encoding_indices, torch.zeros(z.shape[0], device=z.device)


class VQVAEEncoder(nn.Module):
    def __init__(self, d_obs: int, d_h: int, n_e: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_obs, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        self.vq = VectorQuantizer(n_e=n_e, e_dim=d_h)
        self.out_dim = d_h

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        z_q, indices, commitment = self.vq(z)
        return z_q, indices


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class IRISTransformer(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, n_head: int, vocab_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(x) + self.pos_embedding(pos)

        hidden_states = []
        for block in self.blocks:
            x = block(x, mask)
            hidden_states.append(self.ln(x))

        logits = self.head(x)
        return logits, hidden_states[-1]


class IRISRewardHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(h))


class IRISContinueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(h))


class IRISAdapter(BaseModelAdapter):
    """IRIS: Image Reinforcement with Implicit Skills.

    Transformer-based world model with VQVAE discretization.
    """

    def __init__(
        self,
        config: AdapterConfig,
        d_model: int = 256,
        n_layers: int = 4,
        n_head: int = 4,
        vocab_size: int = 512,
    ):
        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.encoder = VQVAEEncoder(config.d_obs, d_model, n_e=vocab_size)
        self.transformer = IRISTransformer(d_model, n_layers, n_head, vocab_size)
        self.reward_head = IRISRewardHead(d_model)
        self.continue_head = IRISContinueHead(d_model)

        self._device = torch.device("cpu")

        # Set capabilities so HookedWorldModel knows to call the predictive heads
        self._capabilities.has_reward_head = True
        self._capabilities.has_continue_head = False  # Disabled for now (needs mapping)
        self._capabilities.has_actor = False  # Disabled for now (needs mapping)
        self._capabilities.has_critic = False  # Disabled for now (needs mapping)
        self._capabilities.uses_actions = True

    @property
    def hook_point_names(self) -> List[str]:
        names = super().hook_point_names
        for i in range(self.config.n_gru_layers or 4):
            names.append(f"transformer_hidden_{i}")
        return names

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into discrete latent.

        Note: IRIS tokens are discrete. 'posterior' here is the one-hot or
        categorical distribution (logits) for the current observation.
        """
        # Flatten observation if needed (e.g. from [B, C, H, W] to [B, D])
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        elif obs.dim() == 2 and obs.shape[1] != self.config.d_obs:
            # If [B, H] but H is not d_obs, maybe it's just a single frame without batch?
            # HookedWorldModel usually passes [B, ...].
            pass

        # IRIS observation encoder (VQ-VAE)
        z_q, indices = self.encoder(obs)
        # For IRIS, the posterior is degenerate (deterministic from pixels),
        # but we return it as logits over vocab for HookedWorldModel compatibility.
        # indices: [B, 1] or [B] -> logits: [B, vocab_size]
        B = obs.shape[0]
        logits = torch.zeros(B, self.vocab_size, device=obs.device, dtype=obs.dtype)
        logits.scatter_(1, indices.view(B, 1), 1.0)
        return logits, z_q

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Predict next token logits from history.

        Args:
            h: History buffer (tokens accumulated so far).
        """
        # h: [B, T] sequence of tokens
        if h.dim() == 1:
            h = h.unsqueeze(0)

        # Transformer predicts next tokens for the entire sequence
        logits, _ = self.transformer(h)
        # Return only the last timestep's logits for the next prediction
        return cast(torch.Tensor, logits[:, -1, :])

    def transition(
        self, h: torch.Tensor, z: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update history by appending the new token.

        Args:
            h: Current history [B, T]
            z: Decided token [B] or [B, 1] or one-hot [B, vocab]
            action: Optional action (not used in minimalist transformer)
        """
        # Ensure h is [B, T]
        if h.dim() == 1:
            h = h.unsqueeze(0)

        B = h.shape[0]

        # Ensure z is [B, 1] index
        if z.dim() == 2 and z.shape[1] == self.vocab_size:
            # One-hot/logits [B, V] -> [B, 1]
            z = z.argmax(dim=-1, keepdim=True)
        elif z.dim() == 1:
            if z.shape[0] == B:
                # [B] -> [B, 1]
                z = z.unsqueeze(1)
            else:
                # [V] (squeezed batch one-hot) -> should not happen but be safe
                z = z.argmax(dim=-1).view(1, 1).expand(B, 1)
        elif z.dim() == 0:
            # Scalar token -> [1, 1]
            z = z.view(1, 1).expand(B, 1)

        # Append to sequence: [B, T] + [B, 1] -> [B, T+1]
        next_h = torch.cat([h, z], dim=1)

        # Truncate if exceeding max_seq_len (e.g. 1024)
        if next_h.shape[1] > 1024:
            next_h = next_h[:, -1024:]

        return next_h

    def _get_transformer_h(self, h: torch.Tensor) -> torch.Tensor:
        """Helper to get the latent 'h' from the token history sequence."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        # We only care about the last token's representation: [B, T, D] -> [B, D]
        _, last_h = self.transformer(h)
        return cast(torch.Tensor, last_h[:, -1, :])

    def predict_reward(self, h: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict reward from history."""
        feat = self._get_transformer_h(h)
        return cast(torch.Tensor, self.reward_head(feat))

    def predict_continue(self, h: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict continue probability from history."""
        feat = self._get_transformer_h(h)
        return cast(torch.Tensor, self.continue_head(feat))

    def actor_forward(self, h: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict action logits from history."""
        feat = self._get_transformer_h(h)
        return cast(torch.Tensor, self.transformer.head(feat))

    def critic_forward(self, h: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict value from history."""
        feat = self._get_transformer_h(h)
        return cast(
            torch.Tensor, self.reward_head(feat)
        )  # In many IRIS versions, reward/value heads are shared or similar

    def initial_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create empty history."""
        device = device or self._device
        # h: [B, 1] with a zero token as a placeholder.
        h = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # z: [B, vocab] initial distribution
        z = torch.zeros(batch_size, self.vocab_size, device=device)
        return h, z

    def to(self, device: torch.device, **kwargs: object) -> "IRISAdapter":  # type: ignore[override]
        super().to(device=device)
        self._device = device
        return self

    def eval(self) -> "IRISAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "IRISAdapter":
        super().train(mode)
        return self

    def list_hookable_points(self) -> List[str]:
        """Return semantic hook names plus concrete module paths."""
        points = list(self.hook_point_names)
        points.extend(name for name, _ in self.named_modules() if name)
        return list(dict.fromkeys(points))
