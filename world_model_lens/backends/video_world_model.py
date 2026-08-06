"""Video world model adapter (WorldDreamer-style).

Reference: "WorldDreamer: World Model for Autonomous Driving" (WorldDreamer)
          "Video Diffusion Models" (Video Diffusion)
          "Make-A-Video" (Make-A-Video)

Video world models learn to predict future video frames:
- Latent video diffusion models
- Token-based video generation
- Masked modeling for efficient prediction

This adapter supports:
- Latent video prediction models
- Token-based video world models
- Video diffusion-based world models
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig, WorldModelCapabilities
from world_model_lens.core.types import WorldModelFamily


class VideoEncoder(nn.Module):
    """Video encoder: video frames -> latent tokens."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)
        x = x / 255.0 if x.max() > 1.0 else x
        features = self.conv(x).flatten(1)
        tokens = self.proj(features)
        return tokens.view(batch_size, time_steps, -1)


class VideoDecoder(nn.Module):
    """Video decoder: latent tokens -> video frames."""

    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, latent_dim = tokens.shape
        x = self.proj(tokens.reshape(batch_size * time_steps, latent_dim))
        x = x.view(batch_size * time_steps, 256, 4, 4)
        video = self.deconv(x)
        return video.view(batch_size, time_steps, -1, video.shape[2], video.shape[3])


class VideoDynamicsTransformer(nn.Module):
    """Transformer for video dynamics prediction."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(256, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, time_steps, dim = tokens.shape
        positions = torch.arange(time_steps, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        tokens = tokens + self.pos_embedding(positions)
        return self.transformer(tokens, src_key_padding_mask=mask)


class VideoWorldModelAdapter(BaseModelAdapter):
    """Adapter for Video World Models (WorldDreamer-style)."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        self.encoder = VideoEncoder(in_channels=3, latent_dim=config.d_embed)
        self.decoder = VideoDecoder(latent_dim=config.d_embed, out_channels=3)
        self.dynamics_model = VideoDynamicsTransformer(
            d_model=config.d_embed,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
        )
        self._capabilities = WorldModelCapabilities(
            has_decoder=True,
            has_reward_head=False,
            has_continue_head=False,
            has_actor=False,
            has_critic=False,
            uses_actions=False,
            is_rl_trained=False,
        )

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return ["encoder", "decoder", "dynamics", "latent_tokens"]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.VIDEO_WORLD_MODEL

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode video frames to latent tokens."""
        del h_prev
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0).unsqueeze(1)
        z = self.encoder(obs)
        return z, z

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future latent tokens."""
        del h, action
        if z.dim() == 2:
            z = z.unsqueeze(1)
        return self.dynamics_model(z)

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Continuous video latents use the current state as the prior carrier."""
        if h.dim() == 2:
            h = h.unsqueeze(1)
        return h

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        del temperature, sample
        return logits_or_repr

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode latent tokens to video frames."""
        del h
        if z.dim() == 2:
            z = z.unsqueeze(1)
        return self.decoder(z)

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> None:
        del h, z
        return None

    def imagine(
        self,
        start_h: torch.Tensor,
        start_z: torch.Tensor,
        action_sequence: Optional[torch.Tensor] = None,
        horizon: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run imagined rollout for video prediction."""
        del action_sequence
        h_t = start_h if start_h.dim() > 1 else start_h.unsqueeze(0)
        z_t = start_z if start_z.dim() > 1 else start_z.unsqueeze(0)
        if h_t.dim() == 2:
            h_t = h_t.unsqueeze(1)
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)

        h_seq = [h_t.squeeze(0)]
        z_seq = [z_t.squeeze(0)]
        for _ in range(horizon):
            h_t = self.transition(h_t, z_t)
            z_t = self.sample_z(self.dynamics(h_t))
            h_seq.append(h_t.squeeze(0))
            z_seq.append(z_t.squeeze(0))
        return torch.stack(h_seq, dim=0), torch.stack(z_seq, dim=0)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize starting state."""
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, 1, self.config.d_embed, device=device)
        z = torch.zeros(batch_size, 1, self.config.d_embed, device=device)
        return h, z

    def to(self, device: torch.device) -> "VideoWorldModelAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "VideoWorldModelAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "VideoWorldModelAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="video_world_model",
    family=WorldModelFamily.VIDEO_WORLD_MODEL,
    description="Video world models (WorldDreamer-style)",
    supports_rl=False,
    supports_video=True,
    supports_planning=False,
)(VideoWorldModelAdapter)
