"""PlaNet adapter implementation.

Reference: "PlaNet: Learning Latent Dynamics for Planning from Pixels" (Hafner et al., 2019)

PlaNet was the predecessor to Dreamer and introduced:
- Latent RSSM: recurrent state-space model with discrete latent variables
- Planning at the latent level using model predictive control (MPC)
- Learning from images without decoder (reward prediction only)

Key differences from Dreamer:
- No decoder (learns only from reward)
- Uses image-based latent planning
- Pure model-based RL without policy gradient
- Simpler architecture than DreamerV2+

This adapter implements the latent dynamics model from PlaNet.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig
from world_model_lens.core.types import WorldModelFamily


class PlaNetEncoder(nn.Module):
    """PlaNet CNN encoder for images."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 400):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )
        self.fc = nn.Linear(256 * 6 * 6, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 if x.max() > 1.0 else x
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class PlaNetDynamics(nn.Module):
    """PlaNet RSSM dynamics (prior) model."""

    def __init__(self, d_h: int, d_z: int, d_action: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_z + d_action, hidden_size=d_h)
        self.fc = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_z),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        h_next = self.gru(x, h)
        prior_logits = self.fc(h_next)
        return h_next, prior_logits


class PlaNetPosterior(nn.Module):
    """PlaNet posterior model (encoder output)."""

    def __init__(self, d_obs: int, d_h: int, d_z: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_obs + d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_z * 2),
        )

    def forward(self, obs_encoding: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_encoding, h], dim=-1)
        return self.fc(x)


class PlaNetRewardPredictor(nn.Module):
    """PlaNet reward predictor (no decoder in original)."""

    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class PlaNetAdapter(BaseModelAdapter):
    """Adapter for PlaNet (latent RSSM from pixels)."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        self.encoder = PlaNetEncoder(in_channels=3, hidden_dim=config.d_obs)
        self.dynamics_model = PlaNetDynamics(config.d_h, config.d_z, config.d_action)
        self.posterior = PlaNetPosterior(config.d_obs, config.d_h, config.d_z)
        self.reward_predictor = PlaNetRewardPredictor(config.d_h, config.d_z)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "encoder",
            "dynamics_gru",
            "posterior",
            "reward",
            "state",
            "prior",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.PLA_NET

    def _sample_posterior(self, posterior_params: torch.Tensor) -> torch.Tensor:
        """Sample a concrete latent from posterior mean/log-scale parameters."""
        mean, log_std = posterior_params.chunk(2, dim=-1)
        std = log_std.clamp(-5, 2).exp()
        return mean + torch.randn_like(std) * std

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to a concrete posterior latent."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        obs_encoding = self.encoder(obs)

        h = h_prev
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if h.shape[0] != obs.shape[0]:
            h = torch.zeros(obs.shape[0], self.config.d_h, device=obs.device)

        posterior_params = self.posterior(obs_encoding, h)
        z_post = self._sample_posterior(posterior_params)
        return z_post, obs_encoding

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition hidden state using latent and action."""
        if action is None:
            action = torch.zeros(h.shape[0], self.config.d_action, device=h.device)

        h_next, _ = self.dynamics_model(h, z, action)
        return h_next

    def predict_reward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict reward from latent state."""
        return self.reward_predictor(h, z)

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> None:
        """PlaNet has no decoder."""
        del h, z
        return None

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize starting state."""
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z = torch.zeros(batch_size, self.config.d_z, device=device)
        return h, z

    def to(self, device: torch.device) -> "PlaNetAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "PlaNetAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "PlaNetAdapter":
        super().train(mode)
        return self

    def list_hookable_points(self) -> List[str]:
        """Return semantic hook names plus concrete module paths."""
        points = list(self.hook_point_names)
        points.extend(name for name, _ in self.named_modules() if name)
        return list(dict.fromkeys(points))


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="planet",
    family=WorldModelFamily.PLA_NET,
    description="PlaNet: Learning Latent Dynamics for Planning from Pixels",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(PlaNetAdapter)
