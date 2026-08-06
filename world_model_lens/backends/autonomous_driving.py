"""Autonomous driving world model adapter.

Reference: "WorldDreamer: World Model for Autonomous Driving"
          "DriveWorld: 4D Latent Space for Multi-Modal Memory Autonomous Driving"

Autonomous driving world models fuse camera/LiDAR/radar into latent state
and predict future scenes or ego-motion.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig, WorldModelCapabilities
from world_model_lens.core.types import WorldModelFamily


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder for camera, LiDAR, radar."""

    def __init__(
        self,
        camera_channels: int = 3,
        lidar_channels: int = 1,
        radar_channels: int = 1,
        d_latent: int = 256,
    ):
        super().__init__()
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(camera_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(lidar_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.radar_encoder = nn.Sequential(
            nn.Conv2d(radar_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 * 16 + 64 * 16 + 64 * 16, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(
        self,
        camera: torch.Tensor,
        lidar: Optional[torch.Tensor] = None,
        radar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cam_features = self.camera_encoder(camera).flatten(1)
        if lidar is None:
            lidar_features = torch.zeros(camera.shape[0], 64 * 16, device=camera.device)
        else:
            lidar_features = self.lidar_encoder(lidar).flatten(1)
        if radar is None:
            radar_features = torch.zeros(camera.shape[0], 64 * 16, device=camera.device)
        else:
            radar_features = self.radar_encoder(radar).flatten(1)
        fused = torch.cat([cam_features, lidar_features, radar_features], dim=-1)
        return self.fusion(fused)


class DrivingDynamics(nn.Module):
    """Dynamics model for autonomous driving."""

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


class EgoMotionPredictor(nn.Module):
    """Predicts ego-motion."""

    def __init__(self, d_latent: int, hidden_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.predictor(state)


class AutonomousDrivingAdapter(BaseModelAdapter):
    """Adapter for Autonomous Driving world models."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config
        self.encoder = MultiModalEncoder(camera_channels=3, lidar_channels=1, radar_channels=1, d_latent=config.d_h)
        self.dynamics_model = DrivingDynamics(config.d_h, config.d_action)
        self.ego_predictor = EgoMotionPredictor(config.d_h)
        self.reward_predictor = nn.Linear(config.d_h, 1)
        self._capabilities = WorldModelCapabilities(
            has_decoder=False,
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
        return ["camera_encoder", "lidar_encoder", "fusion", "dynamics", "ego_motion", "latent_state"]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.AUTONOMOUS_DRIVING

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        del h_prev
        if isinstance(obs, dict):
            camera = obs.get("camera", obs.get("image"))
            lidar = obs.get("lidar")
            radar = obs.get("radar")
            z = self.encoder(camera, lidar, radar)
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

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        del h
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.reward_predictor(z)

    def predict_ego_motion(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.ego_predictor(state)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z = torch.zeros(batch_size, self.config.d_h, device=device)
        return h, z

    def to(self, device: torch.device) -> "AutonomousDrivingAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "AutonomousDrivingAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "AutonomousDrivingAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="autonomous_driving",
    family=WorldModelFamily.AUTONOMOUS_DRIVING,
    description="Autonomous driving world models (camera/LiDAR/radar fusion)",
    supports_rl=True,
    supports_video=True,
    supports_planning=True,
)(AutonomousDrivingAdapter)
