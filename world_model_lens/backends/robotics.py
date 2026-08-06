"""Robotics/embodied world model adapter.

Reference: robotics latent world models for manipulation and locomotion.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig, WorldModelCapabilities
from world_model_lens.core.types import WorldModelFamily


class RoboticsEncoder(nn.Module):
    """Multi-modal encoder for robotics (image, proprio, tactile)."""

    def __init__(self, image_channels: int = 3, proprio_dim: int = 14, tactile_dim: int = 0, d_latent: int = 256):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proprio_encoder = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.tactile_dim = tactile_dim
        if tactile_dim > 0:
            self.tactile_encoder = nn.Sequential(nn.Linear(tactile_dim, 32), nn.ReLU(), nn.Linear(32, 32))
            tactile_out = 32
        else:
            self.tactile_encoder = None
            tactile_out = 0
        total_dim = 128 * 16 + 64 + tactile_out
        self.fusion = nn.Sequential(nn.Linear(total_dim, d_latent), nn.ReLU(), nn.Linear(d_latent, d_latent))

    def forward(self, image: torch.Tensor, proprio: Optional[torch.Tensor] = None, tactile: Optional[torch.Tensor] = None) -> torch.Tensor:
        img_features = self.image_encoder(image).flatten(1)
        if proprio is not None:
            proprio_features = self.proprio_encoder(proprio)
        else:
            proprio_features = torch.zeros(image.shape[0], 64, device=image.device)
        if tactile is not None and self.tactile_encoder is not None:
            tactile_features = self.tactile_encoder(tactile)
        elif self.tactile_encoder is not None:
            tactile_features = torch.zeros(image.shape[0], 32, device=image.device)
        else:
            tactile_features = torch.zeros(image.shape[0], 0, device=image.device)
        return self.fusion(torch.cat([img_features, proprio_features, tactile_features], dim=-1))


class RoboticsDynamics(nn.Module):
    """Latent dynamics for robotics."""

    def __init__(self, d_latent: int, d_action: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_latent),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class RoboticsAdapter(BaseModelAdapter):
    """Adapter for Robotics/Embodied world models."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config
        self.encoder = RoboticsEncoder(image_channels=3, proprio_dim=config.d_action + 7, tactile_dim=0, d_latent=config.d_h)
        self.dynamics_model = RoboticsDynamics(config.d_h, config.d_action)
        self.decoder = nn.Sequential(nn.Linear(config.d_h, 128), nn.ReLU(), nn.Linear(128, 64 * 4 * 4))
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )
        self.reward_predictor = nn.Linear(config.d_h, 1)
        self._capabilities = WorldModelCapabilities(
            has_decoder=True,
            has_reward_head=True,
            has_continue_head=False,
            has_actor=False,
            has_critic=False,
            uses_actions=True,
            is_rl_trained=True,
        )
        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return ["image_encoder", "proprio_encoder", "fusion", "dynamics", "decoder", "latent_state"]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.ROBOTICS

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        del h_prev
        if isinstance(obs, dict):
            image = obs.get("image", obs.get("rgb"))
            proprio = obs.get("proprio")
            tactile = obs.get("tactile")
            z = self.encoder(image, proprio, tactile)
        else:
            if obs.dim() == 3:
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

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        del h
        if z.dim() == 1:
            z = z.unsqueeze(0)
        x = self.decoder(z).view(-1, 64, 4, 4)
        return self.image_decoder(x)

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        del h
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.reward_predictor(z)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z = torch.zeros(batch_size, self.config.d_h, device=device)
        return h, z

    def to(self, device: torch.device) -> "RoboticsAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "RoboticsAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "RoboticsAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="robotics",
    family=WorldModelFamily.ROBOTICS,
    description="Robotics/embodied world models (latent dynamics for manipulation)",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(RoboticsAdapter)
