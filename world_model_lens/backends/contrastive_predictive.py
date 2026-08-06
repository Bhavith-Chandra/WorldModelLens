"""Contrastive/Predictive world model adapter (CWM, SPR-style)."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig, WorldModelCapabilities
from world_model_lens.core.types import WorldModelFamily


class ContrastiveEncoder(nn.Module):
    """Shared encoder for contrastive learning."""

    def __init__(self, d_obs: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_obs, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MomentumEncoder(nn.Module):
    """Momentum encoder for contrastive learning."""

    def __init__(self, encoder: nn.Module, m: float = 0.99):
        super().__init__()
        self.encoder = encoder
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def update(self, encoder: nn.Module):
        for param_q, param_k in zip(encoder.parameters(), self.parameters()):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1 - self.m)


class LatentDynamics(nn.Module):
    """Latent dynamics predictor (predictive coding)."""

    def __init__(self, d_latent: int, d_action: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_latent),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, action], dim=-1))


class ContrastiveAdapter(BaseModelAdapter):
    """Adapter for Contrastive/Predictive world models (CWM, SPR-style)."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config
        d_latent = config.d_h
        self.encoder = ContrastiveEncoder(config.d_obs, d_latent)
        self.momentum_encoder = MomentumEncoder(self.encoder, m=0.99)
        self.projector = nn.Linear(d_latent, d_latent)
        self.predictor = nn.Linear(d_latent, d_latent)
        self.dynamics_model = LatentDynamics(d_latent, config.d_action)
        self._capabilities = WorldModelCapabilities(
            has_decoder=False,
            has_reward_head=False,
            has_continue_head=False,
            has_actor=False,
            has_critic=False,
            uses_actions=True,
            is_rl_trained=False,
        )
        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return ["encoder", "projector", "predictor", "dynamics", "latent"]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.CONTRASTIVE_PREDICTIVE

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        del h_prev
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z = self.encoder(obs)
        return z, z

    def transition(self, h: torch.Tensor, z: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        del h
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if action is None:
            action = torch.zeros(z.shape[0], self.config.d_action, device=z.device)
        elif action.dim() == 1:
            action = action.unsqueeze(0)
        return self.dynamics_model(z, action)

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        return h if h.dim() > 1 else h.unsqueeze(0)

    def sample_z(self, logits_or_repr: torch.Tensor, temperature: float = 1.0, sample: bool = True) -> torch.Tensor:
        del temperature, sample
        return logits_or_repr

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def forward_contrastive(self, z1: torch.Tensor, z2: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj1 = self.projector(z1)
        pred1 = self.predictor(proj1)
        with torch.no_grad():
            target = self.momentum_encoder(z2)
        return pred1, target

    def initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z = torch.zeros(batch_size, self.config.d_h, device=device)
        return h, z

    def to(self, device: torch.device) -> "ContrastiveAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "ContrastiveAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "ContrastiveAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="contrastive_predictive",
    family=WorldModelFamily.CONTRASTIVE_PREDICTIVE,
    description="Contrastive/Predictive world models (CWM, SPR-style)",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(ContrastiveAdapter)
