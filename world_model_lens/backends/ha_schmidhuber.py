"""Ha & Schmidhuber "World Models" adapter implementation.

Reference: "World Models" (Ha & Schmidhuber, 2018)

This adapter implements the classic World Models architecture:
- VAE (Variational Autoencoder) for visual compression
- MDN-RNN (Mixture Density Network + RNN) for latent dynamics
- Simple linear controller for action selection

The model was originally used for car racing in VizDoom.
This adapter supports both the original MDN-RNN dynamics and can
be extended to work with modern variants.

Key characteristics:
- Continuous VAE latent
- MDN outputs mixture of Gaussians for next latent
- Simple controller: linear(z, h) -> action
- No built-in reward prediction (used with external reward)
- Non-RL by default: can be extended for RL use
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.base_adapter import BaseModelAdapter, AdapterConfig
from world_model_lens.core.types import WorldModelFamily


class VAEEncoder(nn.Module):
    """VAE Encoder: image -> latent mean + logvar."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
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
        self.fc_mean = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x / 255.0 if x.max() > 1.0 else x
        h = self.conv(x)
        h = h.flatten(1)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder: latent -> image reconstruction."""

    def __init__(self, latent_dim: int = 32, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 6 * 6)
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 6, 6)
        return self.deconv(h)


class MDNRNN(nn.Module):
    """Mixture Density Network RNN for latent dynamics.

    Outputs parameters of a mixture of Gaussians for predicting next latent.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int, n_mixtures: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures

        input_dim = latent_dim + action_dim
        self.rnn = nn.GRUCell(input_dim, hidden_dim)

        self.mdn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 3 * n_mixtures),
        )

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is not None:
            x = torch.cat([z, action], dim=-1)
        else:
            x = z

        h_next = self.rnn(x, h)

        batch_size = h_next.shape[0]
        mdn_output = self.mdn_fc(h_next)
        mdn_output = mdn_output.view(batch_size, self.latent_dim, self.n_mixtures, 3)

        pi = F.softmax(mdn_output[..., 0], dim=-1)
        mu = mdn_output[..., 1]
        log_sigma = mdn_output[..., 2]

        return h_next, pi, mu, log_sigma

    def sample_next_z(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, latent_dim, n_mixtures = pi.shape
        mixture_indices = torch.multinomial(pi.reshape(-1, n_mixtures), 1).view(batch_size, latent_dim)
        gather_index = mixture_indices.unsqueeze(-1)
        mu_selected = mu.gather(-1, gather_index).squeeze(-1)
        sigma_selected = log_sigma.exp().gather(-1, gather_index).squeeze(-1)
        return mu_selected + torch.randn_like(mu_selected) * sigma_selected


class SimpleController(nn.Module):
    """Simple linear controller: (z, h) -> action.

    This is the controller from the original World Models paper.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, h], dim=-1)
        return self.fc(x)


class HaSchmidhuberWorldModelAdapter(BaseModelAdapter):
    """Adapter for Ha & Schmidhuber "World Models" (VAE + MDN-RNN + Controller)."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        latent_dim = config.d_z
        hidden_dim = config.d_h
        action_dim = config.d_action
        n_mixtures = 5

        self.vae_encoder = VAEEncoder(in_channels=3, latent_dim=latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim=latent_dim, out_channels=3)
        self.mdn_rnn = MDNRNN(latent_dim, hidden_dim, action_dim, n_mixtures)
        self.controller = SimpleController(latent_dim, hidden_dim, action_dim)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "vae_encoder",
            "vae_decoder",
            "mdn_rnn_hidden",
            "controller",
            "observation",
            "latent",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.HA_SCHMIDHUBER

    def encode(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to a VAE latent."""
        del h_prev
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        mean, logvar = self.vae_encoder(obs)
        std = (0.5 * logvar).exp()
        z = mean + std * torch.randn_like(std)
        return z, mean

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Advance the MDN-RNN hidden state."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if action is not None and action.dim() == 1:
            action = action.unsqueeze(0)

        if action is None:
            action = torch.zeros(z.shape[0], self.config.d_action, device=z.device)

        h_next, _, _, _ = self.mdn_rnn(z, h, action)
        return h_next

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Sample the next latent prior from the MDN parameters predicted from h."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        batch_size = h.shape[0]
        mdn_output = self.mdn_rnn.mdn_fc(h)
        mdn_output = mdn_output.view(batch_size, self.config.d_z, self.mdn_rnn.n_mixtures, 3)
        pi = F.softmax(mdn_output[..., 0], dim=-1)
        mu = mdn_output[..., 1]
        log_sigma = mdn_output[..., 2]
        return self.mdn_rnn.sample_next_z(pi, mu, log_sigma)

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Ha/Schmidhuber uses continuous VAE latents, so priors pass through directly."""
        del temperature, sample
        return logits_or_repr

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode latent to observation with the VAE decoder."""
        del h
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.vae_decoder(z)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Controller forward: (z, h) -> action."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.controller(z, h)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and latent state."""
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z = torch.zeros(batch_size, self.config.d_z, device=device)
        return h, z

    def to(self, device: torch.device) -> "HaSchmidhuberWorldModelAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "HaSchmidhuberWorldModelAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "HaSchmidhuberWorldModelAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily

register(
    name="ha_schmidhuber",
    family=WorldModelFamily.HA_SCHMIDHUBER,
    description="Ha & Schmidhuber World Models (VAE + MDN-RNN + Controller)",
    supports_rl=False,
    supports_video=True,
    supports_planning=True,
)(HaSchmidhuberWorldModelAdapter)
