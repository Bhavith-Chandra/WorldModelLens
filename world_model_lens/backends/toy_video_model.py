"""Toy latent video world model adapter.

This module implements a simple video world model for testing WorldModelLens
with non-RL (pure video prediction) models.

Architecture:
- CNN encoder: frames -> latent z_t
- Latent transition: z_t -> z_{t+1} (no actions)
- CNN decoder: z_t -> reconstructed frame

This model has NO reward head, NO value head, NO actions - pure video prediction.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, WorldModelCapabilities


class ToyVideoEncoder(nn.Module):
    """Simple CNN encoder for video frames."""

    def __init__(self, obs_channels: int = 3, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.obs_channels = obs_channels
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] != 64 or x.shape[-2] != 64:
            x = nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

        h = self.conv(x).flatten(1)
        z = self.to_latent(h)
        return z


class ToyVideoDecoder(nn.Module):
    """Simple CNN decoder for video frames."""

    def __init__(self, latent_dim: int = 32, obs_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_channels = obs_channels

        self.from_latent = nn.Linear(latent_dim, hidden_dim * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)

        h = self.from_latent(z)
        h = h.view(h.shape[0], -1, 4, 4)
        x_recon = self.deconv(h)

        x_recon = nn.functional.interpolate(
            x_recon, size=(64, 64), mode="bilinear", align_corners=False
        )
        return x_recon


class ToyVideoTransition(nn.Module):
    """Simple MLP transition for latent dynamics.

    No actions - pure latent dynamics prediction.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ToyVideoWorldModel(nn.Module):
    """Complete toy video world model."""

    def __init__(
        self,
        obs_channels: int = 3,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        frame_size: int = 64,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.frame_size = frame_size

        self.encoder = ToyVideoEncoder(obs_channels, latent_dim, hidden_dim)
        self.transition = ToyVideoTransition(latent_dim, hidden_dim)
        self.decoder = ToyVideoDecoder(latent_dim, obs_channels, hidden_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def transition_forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transition(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ToyVideoAdapter(BaseModelAdapter):
    """Adapter for toy video world model.

    This adapter implements a simple video prediction model with:
    - Encoder: CNN from frames to latent z
    - Transition: MLP predicting next latent (no actions)
    - Decoder: CNN from latent to reconstructed frame

    Capabilities:
    - has_decoder: True (can reconstruct frames)
    - has_reward_head: False
    - has_continue_head: False
    - has_actor: False
    - has_critic: False
    - uses_actions: False
    - is_rl_trained: False

    Example:
        >>> from world_model_lens import HookedWorldModel
        >>> from world_model_lens.backends.toy_video_model import ToyVideoAdapter
        >>>
        >>> config = type('Config', (), {'latent_dim': 32, 'hidden_dim': 128})()
        >>> adapter = ToyVideoAdapter(config)
        >>> wm = HookedWorldModel(adapter, config)
        >>>
        >>> # Generate random video sequence
        >>> frames = torch.randn(10, 3, 64, 64)
        >>> traj, cache = wm.run_with_cache(frames)
    """

    def __init__(
        self,
        config: Any = None,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        obs_channels: int = 3,
        frame_size: int = 64,
    ):
        if config is None:
            config = type(
                "Config",
                (),
                {
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "obs_channels": obs_channels,
                    "frame_size": frame_size,
                },
            )()

        super().__init__(config)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.obs_channels = obs_channels
        self.frame_size = frame_size

        self.model = ToyVideoWorldModel(
            obs_channels=obs_channels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            frame_size=frame_size,
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

    @property
    def capabilities(self) -> WorldModelCapabilities:
        return self._capabilities

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into latent representation.

        Args:
            obs: Observation tensor [..., C, H, W]
            h_prev: Previous hidden state [..., d_h] (unused for pure video model)

        Returns:
            Tuple of (z_posterior, z_prior_or_repr).
            Both are the same for this model since there's no separate prior network.
        """
        z = self.model.encode(obs)

        if z.dim() == 1:
            z = z.unsqueeze(0)

        prior = self.model.transition_forward(
            torch.zeros_like(z) if h_prev is None else h_prev[..., : self.latent_dim]
        )

        return z, z

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition latent state (action is ignored).

        Args:
            h: Current hidden state [..., d_h] (unused)
            z: Current latent [..., d_z]
            action: Ignored (no actions in video model)

        Returns:
            Next hidden state (same as next latent).
        """
        next_z = self.model.transition_forward(z)
        return next_z

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Compute prior from hidden state.

        For video model, this just returns the h encoding as the prior.
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)

        if h.shape[-1] != self.latent_dim:
            h_proj = nn.functional.linear(
                h,
                torch.eye(self.latent_dim, h.shape[-1], device=h.device)
                .flatten()
                .view(self.latent_dim, -1),
            )
        else:
            h_proj = h

        return h_proj

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode latent to observation reconstruction.

        Args:
            h: Hidden state [..., d_h] (unused)
            z: Latent tensor [..., d_z]

        Returns:
            Reconstructed observation or None.
        """
        if not self._capabilities.has_decoder:
            return None

        return self.model.decode(z)

    def initial_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and latent state.

        Args:
            batch_size: Number of initial states.
            device: Optional device to place tensors on.

        Returns:
            Tuple of (h_0, z_0) initial states.
        """
        h_0 = torch.zeros(batch_size, self.latent_dim)
        z_0 = torch.zeros(batch_size, self.latent_dim)
        if device is not None:
            h_0 = h_0.to(device)
            z_0 = z_0.to(device)
        return h_0, z_0

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample from distribution or return representation as-is.

        For continuous video model, just returns the representation.

        Args:
            logits_or_repr: Representation tensor [..., d_z]
            temperature: Ignored for continuous model.

        Returns:
            Sampled representation.
        """
        if logits_or_repr.dim() == 1:
            return logits_or_repr
        return logits_or_repr

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        """Return named parameters."""
        return dict(self.model.named_parameters())

    def forward(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, transition, decode.

        Args:
            obs: Observation tensor.
            h_prev: Previous hidden state.

        Returns:
            Tuple of (z_posterior, z_prior, reconstruction).
        """
        z = self.model.encode(obs)
        z_prior = self.model.transition_forward(z)
        recon = self.model.decode(z)
        return z, z_prior, recon


def create_toy_video_adapter(
    latent_dim: int = 32,
    hidden_dim: int = 128,
    obs_channels: int = 3,
) -> ToyVideoAdapter:
    """Factory function to create a toy video adapter.

    Args:
        latent_dim: Dimension of latent space.
        hidden_dim: Hidden layer dimension.
        obs_channels: Number of observation channels.

    Returns:
        Configured ToyVideoAdapter instance.
    """
    config = type(
        "Config",
        (),
        {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "obs_channels": obs_channels,
        },
    )()

    return ToyVideoAdapter(
        config=config,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        obs_channels=obs_channels,
    )
