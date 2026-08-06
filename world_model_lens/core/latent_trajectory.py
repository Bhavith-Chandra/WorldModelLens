"""Latent trajectory representation for sequences of states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LatentTrajectory:
    """A sequence of LatentState objects representing a trajectory.

    Attributes:
        states: Ordered list of LatentState objects.
        env_name: Name of the environment.
        episode_id: Unique episode identifier.
        imagined: Whether this trajectory was imagined (vs real).
        fork_point: Timestep where trajectory was forked, if applicable.
        metadata: Additional trajectory-level data.
    """

    states: list[Any]
    env_name: str = "unknown"
    episode_id: int | None = None
    imagined: bool = False
    fork_point: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of timesteps in the trajectory."""
        return len(self.states)

    @property
    def h_sequence(self) -> torch.Tensor:
        """Stacked hidden states [T, d_h]."""
        return torch.stack([s.h_t for s in self.states], dim=0)

    @property
    def z_posterior_sequence(self) -> torch.Tensor:
        """Stacked z_posterior [T, n_cat, n_cls]."""
        return torch.stack([s.z_posterior for s in self.states], dim=0)

    @property
    def z_prior_sequence(self) -> torch.Tensor:
        """Stacked z_prior [T, n_cat, n_cls]."""
        return torch.stack([s.z_prior for s in self.states], dim=0)

    @property
    def kl_sequence(self) -> torch.Tensor:
        """Per-timestep KL divergence [T]."""
        return torch.stack([s.kl for s in self.states], dim=0)

    @property
    def rewards_pred(self) -> torch.Tensor | None:
        """Stacked predicted rewards [T, ...] if available."""
        vals = [s.reward_pred for s in self.states if s.reward_pred is not None]
        return torch.stack(vals, dim=0) if vals else None

    @property
    def rewards_real(self) -> torch.Tensor | None:
        """Stacked actual rewards [T, ...] if available."""
        vals = [s.reward_real for s in self.states if s.reward_real is not None]
        return torch.stack(vals, dim=0) if vals else None

    @property
    def actions(self) -> torch.Tensor | None:
        """Stacked actions [T, d_action] if available."""
        vals = [s.action for s in self.states if s.action is not None]
        return torch.stack(vals, dim=0) if vals else None

    def surprise_peaks(
        self,
        threshold: float,
    ) -> list[tuple[int, float]]:
        """Find timesteps where KL exceeds threshold.

        Args:
            threshold: KL value threshold.

        Returns:
            List of (timestep, kl_value) tuples above threshold.
        """
        return [
            (i, self.kl_sequence[i].item())
            for i in range(self.length)
            if self.kl_sequence[i].item() > threshold
        ]

    def slice(self, start: int, end: int) -> "LatentTrajectory":
        """Slice the trajectory between start and end.

        Args:
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            New LatentTrajectory with sliced states.
        """
        return LatentTrajectory(
            states=self.states[start:end],
            env_name=self.env_name,
            episode_id=self.episode_id,
            imagined=self.imagined,
            fork_point=self.fork_point,
            metadata=self.metadata.copy(),
        )

    def to_device(self, device: torch.device) -> "LatentTrajectory":
        """Move all states to the specified device.

        Args:
            device: Target torch device.

        Returns:
            New LatentTrajectory on target device.
        """
        return LatentTrajectory(
            states=[s.to_device(device) for s in self.states],
            env_name=self.env_name,
            episode_id=self.episode_id,
            imagined=self.imagined,
            fork_point=self.fork_point,
            metadata=self.metadata.copy(),
        )

    def fork_at(self, timestep: int) -> "LatentTrajectory":
        """Create a new trajectory starting from the given timestep.

        Args:
            timestep: Fork point index.

        Returns:
            New LatentTrajectory representing the prefix from fork point.
        """
        return LatentTrajectory(
            states=self.states[timestep:],
            env_name=self.env_name,
            episode_id=self.episode_id,
            imagined=self.imagined,
            fork_point=timestep,
            metadata=self.metadata.copy(),
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Any:
        return self.states[index]
