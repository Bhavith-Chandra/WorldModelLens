"""Diffusion-based world model adapter.

This adapter supports diffusion-based world models which are becoming
increasingly popular for video generation and dynamics modeling.

Supported architectures:
- DiT (Diffusion Transformer)
- UViT (Unified Vision Transformer)
- Diffusion RNN
- Latent Diffusion Models
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig
from world_model_lens.core.types import (
    WorldModelMetadata,
    ModelPurpose,
    WorldModelFamily,
    LatentType,
    DynamicsType,
    ObservationModality,
    PredictionHead,
)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class DiffusionEncoder(nn.Module):
    """Encoder for diffusion world model."""

    def __init__(
        self,
        obs_channels: int = 3,
        latent_dim: int = 256,
        hidden_channels: int = 128,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(obs_channels, hidden_channels, 4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1)

        self.res1 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        self.res2 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)

        self.out = nn.Linear(hidden_channels * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # [B, C, H, W]
            B = x.shape[0]
            x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        x = F.silu(self.conv1(x.permute(0, 2, 1)))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))

        x = self.res1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.res2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x.mean(dim=1)
        return self.out(x)


class DiffusionDecoder(nn.Module):
    """Decoder for diffusion world model."""

    def __init__(
        self,
        latent_dim: int = 256,
        obs_channels: int = 3,
        hidden_channels: int = 128,
    ):
        super().__init__()

        self.in_proj = nn.Linear(latent_dim, hidden_channels * 4)

        self.res1 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)
        self.res2 = ResidualBlock(hidden_channels * 4, hidden_channels * 4)

        self.conv1 = nn.ConvTranspose1d(
            hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1
        )
        self.conv2 = nn.ConvTranspose1d(
            hidden_channels * 2, hidden_channels, 4, stride=2, padding=1
        )
        self.conv3 = nn.Conv1d(hidden_channels, obs_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.in_proj(z))

        x = self.res1(x.unsqueeze(1)).squeeze(1)
        x = self.res2(x.unsqueeze(1)).squeeze(1)

        x = F.silu(self.conv1(x.unsqueeze(2)))
        x = F.silu(self.conv2(x))
        x = self.conv3(x)

        return x.permute(0, 2, 1)


class DiffusionDynamics(nn.Module):
    """Dynamics model using diffusion for next-state prediction.

    Instead of predicting a single next state, this predicts the distribution
    of next states using a denoising diffusion process.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        n_steps: int = 100,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.latent_dim = latent_dim

        self.time_embed = TimeEmbedding(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.alphas = self._cosine_schedule(n_steps)

    def _cosine_schedule(self, n_steps: int) -> torch.Tensor:
        """Cosine schedule for diffusion."""
        steps = torch.arange(n_steps + 1)
        alpha_bar = torch.cos((steps / n_steps + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        return alpha_bar

    def forward(
        self,
        z: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict denoised state.

        Args:
            z: Noisy latent [B, D]
            t: Timestep tensor [B]
            noise: Noise tensor [B, D]

        Returns:
            Predicted clean latent
        """
        B = z.shape[0]

        if t is None:
            t = torch.randint(0, self.n_steps, (B,), device=z.device)
        if noise is None:
            noise = torch.randn_like(z)

        t_emb = self.time_embed(t)

        x = torch.cat([z, t_emb], dim=-1)
        pred = self.net(x)

        alpha = self.alphas[t].view(B, 1).to(z.device)
        pred_clean = (z - torch.sqrt(1 - alpha) * pred) / torch.sqrt(alpha)

        return pred_clean

    def sample(
        self,
        shape: Tuple[int],
        device: torch.device = torch.device("cpu"),
        n_steps: int = 50,
    ) -> torch.Tensor:
        """Sample from the dynamics model.

        Args:
            shape: Shape of sample [B, D]
            device: Device
            n_steps: Number of denoising steps

        Returns:
            Sampled latent
        """
        x = torch.randn(shape, device=device)

        for i in range(n_steps):
            t = torch.full((shape[0],), n_steps - i - 1, device=device, dtype=torch.long)
            pred = self.forward(x, t)
            alpha = self.alphas[t].view(-1, 1)

            if i < n_steps - 1:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha) * pred + torch.sqrt(1 - alpha) * noise
            else:
                x = pred

        return x


class DiffusionWorldModelAdapter(BaseModelAdapter):
    """Adapter for diffusion-based world models.

    This adapter supports world models that use diffusion for:
    - Latent dynamics prediction
    - Next-frame video generation
    - State-space modeling

    Example:
        config = AdapterConfig(
            model_purpose=ModelPurpose.VIDEO_PREDICTION,
            d_latent=256,
        )
        adapter = DiffusionWorldModelAdapter(config)
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)

        self.config = config

        d_obs = config.d_obs or 3 * 64 * 64
        d_latent = config.d_latent or 256

        self.encoder = DiffusionEncoder(
            obs_channels=3,
            latent_dim=d_latent,
            hidden_channels=config.n_encoder_channels,
        )

        self.decoder = DiffusionDecoder(
            latent_dim=d_latent,
            obs_channels=3,
            hidden_channels=config.n_encoder_channels,
        )

        self.dynamics = DiffusionDynamics(
            latent_dim=d_latent,
            hidden_dim=config.d_h,
            n_steps=100,
        )

        self._setup_metadata()

    def _setup_metadata(self) -> None:
        """Setup world model metadata."""
        metadata = WorldModelMetadata(
            name="DiffusionWorldModel",
            family=WorldModelFamily.DIFFUSION,
            purpose=ModelPurpose.VIDEO_PREDICTION,
            latent_type=LatentType.CONTINUOUS,
            dynamics_type=DynamicsType.DIFFUSION,
            observation_modality=ObservationModality.PIXEL,
            d_latent=self.config.d_latent,
            d_hidden=self.config.d_h,
            has_encoder=True,
            has_decoder=True,
            has_dynamics=True,
        )

        metadata.add_prediction_head("reconstruction", output_shape=(3, 64, 64))
        metadata.add_prediction_head("next_latent", output_shape=(self.config.d_latent,))

        self._capabilities.set_metadata(metadata)
        self._capabilities.has_decoder = True

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent.

        Args:
            obs: Observation tensor [..., C, H, W]
            h_prev: Previous hidden state [..., d_h] (used as context)

        Returns:
            Tuple of (latent, obs_encoding)
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        if obs.shape[1] != 3:
            obs = obs[:, :3]

        z = self.encoder(obs)

        if h_prev.shape[0] > 1 and z.shape[0] == 1:
            h_prev = h_prev[:1]

        if h_prev.shape[-1] != z.shape[-1]:
            h_context = torch.zeros_like(z)
            h_context[:, : h_prev.shape[-1]] = h_prev
        else:
            h_context = h_prev

        z = z + h_context * 0.1

        return z, z

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition using diffusion dynamics.

        Args:
            h: Current hidden state [..., d_h]
            z: Current latent [..., d_z]
            action: Optional action [..., d_action]

        Returns:
            Next hidden state
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        next_z = self.dynamics.sample(z.shape, z.device)

        if h.dim() > 1 and h.shape[0] == 1 and z.shape[0] > 1:
            h = h.expand(z.shape[0], -1)

        h_new = h + next_z * 0.1

        return h_new

    def dynamics_fn(
        self,
        z: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Denoise latent using dynamics model.

        Args:
            z: Noisy latent
            n_steps: Number of denoising steps

        Returns:
            Denoised latent
        """
        return self.dynamics.sample(z.shape, z.device, n_steps)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and latent state.

        Args:
            batch_size: Number of initial states
            device: Device

        Returns:
            Tuple of (h_0, z_0)
        """
        if device is None:
            device = torch.device("cpu")

        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z = torch.randn(batch_size, self.config.d_latent, device=device)

        return h, z

    def decode(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Decode latent to observation.

        Args:
            h: Hidden state
            z: Latent

        Returns:
            Reconstructed observation
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        h_expanded = torch.zeros_like(z)
        if h.dim() == 1:
            h_expanded[:, : h.shape[0]] = h
        else:
            h_expanded[:, : h.shape[-1]] = h

        recon = self.decoder(z + h_expanded * 0.1)

        return recon

    def sample_trajectory(
        self,
        start_z: torch.Tensor,
        horizon: int = 10,
        n_diffusion_steps: int = 10,
    ) -> torch.Tensor:
        """Sample imagined trajectory.

        Args:
            start_z: Starting latent [B, D]
            horizon: Number of steps
            n_diffusion_steps: Denoising steps per transition

        Returns:
            Trajectory [horizon, B, D]
        """
        trajectory = [start_z]
        z_t = start_z

        for _ in range(horizon):
            z_next = self.dynamics_fn(z_t, n_diffusion_steps)
            trajectory.append(z_next)
            z_t = z_next

        return torch.stack(trajectory, dim=0)

    def predict(
        self,
        head_name: str,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Generic prediction for any head."""
        if head_name == "reconstruction":
            return self.decode(h, z)
        elif head_name == "next_latent":
            return self.dynamics_fn(z.unsqueeze(0)).squeeze(0)
        return None

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        """Return named parameters."""
        return dict(nn.Module.named_parameters(self))
