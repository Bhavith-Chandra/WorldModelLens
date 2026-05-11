"""Toy scientific / latent dynamics model adapter.

This module implements a simple scientific latent dynamics model for testing
WorldModelLens with non-RL models that predict latent state evolution.

Architecture:
- Observation encoder: vector obs -> latent z
- Latent dynamics: z_t -> z_{t+1} (no actions, nonlinear dynamics)
- No decoder, no rewards, no actions

This model simulates scientific systems like:
- Lorenz attractor dynamics
- Epidemic models
- Chemical reaction kinetics
- Physical system simulations
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import BaseModelAdapter, WorldModelCapabilities


class SimpleDynamicsEncoder(nn.Module):
    """Simple encoder from observations to latent state.

    For scientific models, observations might be sensor readings or
    measured quantities that get mapped to latent state.
    """

    def __init__(self, obs_dim: int = 10, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class LatentDynamicsModel(nn.Module):
    """Nonlinear latent dynamics model.

    Models the evolution of latent state over time without actions.
    Can represent various scientific dynamics (Lorenz, Lotka-Volterra, etc.).
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.scale = nn.Parameter(torch.ones(latent_dim) * 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dz = self.net(z)
        z_next = z + self.scale * dz
        return z_next


class ToyScientificWorldModel(nn.Module):
    """Complete toy scientific dynamics model."""

    def __init__(
        self,
        obs_dim: int = 10,
        latent_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = SimpleDynamicsEncoder(obs_dim, latent_dim, hidden_dim)
        self.dynamics_net = LatentDynamicsModel(latent_dim, hidden_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.encoder(obs)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        return self.dynamics_net(z)


class ToyScientificAdapter(BaseModelAdapter):
    """Adapter for toy scientific latent dynamics model.

    This adapter implements a simple scientific dynamics model with:
    - Encoder: MLP from observation vector to latent
    - Dynamics: MLP predicting next latent state (no actions)
    - No decoder (pure latent dynamics)
    - No reward head
    - No value head

    Capabilities:
    - has_decoder: False
    - has_reward_head: False
    - has_continue_head: False
    - has_actor: False
    - has_critic: False
    - uses_actions: False
    - is_rl_trained: False

    Example:
        >>> from world_model_lens import HookedWorldModel
        >>> from world_model_lens.backends.toy_scientific_model import ToyScientificAdapter
        >>>
        >>> config = type('Config', (), {'obs_dim': 10, 'latent_dim': 16})()
        >>> adapter = ToyScientificAdapter(config)
        >>> wm = HookedWorldModel(adapter, config)
        >>>
        >>> # Generate synthetic observations
        >>> observations = torch.randn(50, 10)  # 50 timesteps, 10D observations
        >>> traj, cache = wm.run_with_cache(observations)
    """

    def __init__(
        self,
        config: Any = None,
        obs_dim: int = 10,
        latent_dim: int = 16,
        hidden_dim: int = 64,
    ):
        if config is None:
            config = type(
                "Config",
                (),
                {
                    "obs_dim": obs_dim,
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                },
            )()

        super().__init__(config)

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.model = ToyScientificWorldModel(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        self._capabilities = WorldModelCapabilities(
            has_decoder=False,
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
            obs: Observation tensor [..., obs_dim]
            h_prev: Previous hidden state [..., latent_dim] (used as latent state)

        Returns:
            Tuple of (z_posterior, z_prior_or_repr).
            For scientific model, both are the latent encoding.
        """
        z = self.model.encode(obs)

        if z.dim() == 1:
            z = z.unsqueeze(0)

        prior = self.model.step(z)

        return z, prior

    def transition(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition latent state (action is ignored).

        Args:
            h: Current hidden state [..., latent_dim] (unused)
            z: Current latent [..., latent_dim]
            action: Ignored (no actions in scientific model)

        Returns:
            Next latent state.
        """
        next_z = self.model.step(z)
        return next_z

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Compute dynamics prediction from hidden state.

        Args:
            h: Hidden state [..., latent_dim]

        Returns:
            Next latent state prediction.
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.model.step(h)

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
        """Return representation as-is (continuous model).

        Args:
            logits_or_repr: Representation tensor [..., latent_dim]
            temperature: Ignored.

        Returns:
            The same representation.
        """
        if logits_or_repr.dim() == 1:
            return logits_or_repr
        return logits_or_repr

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        """Return named parameters."""
        return dict(self.model.named_parameters())

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode and predict next state.

        Args:
            obs: Observation tensor.

        Returns:
            Tuple of (z_current, z_next).
        """
        z = self.model.encode(obs)
        z_next = self.model.step(z)
        return z, z_next


def create_toy_scientific_adapter(
    obs_dim: int = 10,
    latent_dim: int = 16,
    hidden_dim: int = 64,
) -> ToyScientificAdapter:
    """Factory function to create a toy scientific adapter.

    Args:
        obs_dim: Dimension of observation space.
        latent_dim: Dimension of latent state space.
        hidden_dim: Hidden layer dimension.

    Returns:
        Configured ToyScientificAdapter instance.
    """
    config = type(
        "Config",
        (),
        {
            "obs_dim": obs_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
        },
    )()

    return ToyScientificAdapter(
        config=config,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )


def generate_lorenz_trajectory(
    n_steps: int = 100,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate a trajectory from the Lorenz attractor.

    This can be used to create synthetic trajectories for the scientific model.

    Args:
        n_steps: Number of timesteps.
        dt: Integration timestep.
        sigma, rho, beta: Lorenz system parameters.
        initial_state: Optional starting point.

    Returns:
        Tensor of shape [n_steps, 3] with (x, y, z) trajectory.
    """
    if initial_state is None:
        x, y, z = 1.0, 1.0, 1.0
    else:
        x, y, z = initial_state[0].item(), initial_state[1].item(), initial_state[2].item()

    trajectory = []
    for _ in range(n_steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        x += dx
        y += dy
        z += dz

        trajectory.append([x, y, z])

    return torch.tensor(trajectory, dtype=torch.float32)


def generate_pendulum_trajectory(
    n_steps: int = 100,
    dt: float = 0.05,
    damping: float = 0.1,
    initial_angle: float = 1.0,
) -> torch.Tensor:
    """Generate a trajectory from a damped pendulum.

    State is [theta, omega] (angle, angular velocity).

    Args:
        n_steps: Number of timesteps.
        dt: Integration timestep.
        damping: Damping coefficient.
        initial_angle: Starting angle in radians.

    Returns:
        Tensor of shape [n_steps, 2] with (theta, omega) trajectory.
    """
    theta = initial_angle
    omega = 0.0
    g = 9.81
    L = 1.0

    trajectory = []
    for _ in range(n_steps):
        # Use as_tensor to avoid creating a new tensor when `theta` is already
        # a torch.Tensor (prevents a copy-construction warning).
        alpha = -(g / L) * torch.sin(torch.as_tensor(theta)) - damping * omega
        omega += alpha * dt
        theta += omega * dt

        trajectory.append([theta, omega])

    return torch.tensor(trajectory, dtype=torch.float32)
