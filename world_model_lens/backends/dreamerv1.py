"""DreamerV1 adapter implementation.

Reference: "Dream to Control: Learning Behaviors by Latent Imagination" (Hafner et al., 2020)

DreamerV1 was the first Dreamer paper. Key characteristics:
- RSSM with discrete latent variables (like later Dreamers)
- Image decoder for reconstruction
- Value function and policy both trained from imagined trajectories
- Uses ELU activation (vs SiLU in V3)
- Simpler architecture than V2/V3

Key differences from DreamerV2/V3:
- ELU activations instead of SiLU
- Simpler encoder/decoder architecture
- Two-hot reward encoding (like V2)
- No layer normalization in some components
- Lower compute requirements
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig
from world_model_lens.core.types import WorldModelFamily


class DreamerV1Encoder(nn.Module):
    """DreamerV1 CNN encoder."""

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


class DreamerV1Decoder(nn.Module):
    """DreamerV1 CNN decoder."""

    def __init__(self, hidden_dim: int = 400, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 256 * 6 * 6)
        self.deconv = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, 256, 6, 6)
        return self.deconv(h)


class DreamerV1Dynamics(nn.Module):
    """DreamerV1 RSSM dynamics (GRU + prior)."""

    def __init__(self, d_z: int, d_action: int, d_h: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_z + d_action, hidden_size=d_h)
        self.fc = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_z),
        )

    def forward(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, action], dim=-1)
        h_next = self.gru(x, h)
        prior_logits = self.fc(h_next)
        return h_next, prior_logits


class DreamerV1Posterior(nn.Module):
    """DreamerV1 posterior model."""

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


class DreamerV1RewardHead(nn.Module):
    """DreamerV1 reward predictor (two-hot encoding)."""

    def __init__(self, d_h: int, d_z: int, bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.bins = bins
        self.low = low
        self.high = high
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, bins),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV1ValueHead(nn.Module):
    """DreamerV1 value predictor."""

    def __init__(self, d_h: int, d_z: int, bins: int = 255):
        super().__init__()
        self.bins = bins
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, bins),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV1Actor(nn.Module):
    """DreamerV1 policy (discrete actions)."""

    def __init__(self, d_h: int, d_z: int, d_action: int, n_classes: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_action * n_classes),
        )
        self.n_classes = n_classes

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV1Adapter(BaseModelAdapter):
    """Adapter for DreamerV1.

    The original Dreamer algorithm with:
    - RSSM latent dynamics
    - Image reconstruction decoder
    - Two-hot reward encoding
    - Value and policy from imagined rollouts
    - ELU activations
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DreamerV1Encoder(in_channels=3, hidden_dim=config.d_obs)
        self.decoder = DreamerV1Decoder(hidden_dim=config.d_h + config.d_z, out_channels=3)
        self.dynamics_model = DreamerV1Dynamics(config.d_z, config.d_action, config.d_h)
        self.posterior = DreamerV1Posterior(config.d_obs, config.d_h, config.d_z)
        self.reward_head = DreamerV1RewardHead(config.d_h, config.d_z)
        self.value_head = DreamerV1ValueHead(config.d_h, config.d_z)
        self.actor = DreamerV1Actor(config.d_h, config.d_z, config.d_action)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "encoder",
            "decoder",
            "dynamics_gru",
            "posterior",
            "reward",
            "value",
            "actor",
            "state",
            "prior",
            "posterior",
        ]

    def list_hookable_points(self) -> List[str]:
        """Return semantic hook names plus concrete module paths."""
        points = list(self.hook_point_names)
        points.extend(name for name, _ in self.named_modules() if name)
        return list(dict.fromkeys(points))

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.DREAMER

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

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode state to observation."""
        x = torch.cat([h, z], dim=-1)
        return self.decoder(x)

    def predict_reward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict reward."""
        return self.reward_head(h, z)

    def critic_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict value."""
        return self.value_head(h, z)

    def actor_forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict action."""
        return self.actor(h, z)

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

    def to(self, device: torch.device) -> "DreamerV1Adapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "DreamerV1Adapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "DreamerV1Adapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="dreamerv1",
    family=WorldModelFamily.DREAMER,
    description="DreamerV1: Dream to Control (original Dreamer)",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(DreamerV1Adapter)
