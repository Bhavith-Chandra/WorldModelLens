from __future__ import annotations

"""Representation engineering tools for world model latent spaces.

This module provides tools for:
- Latent steering: Modify latent representations to achieve desired outcomes
- Concept vectors: Extract and manipulate semantic directions
- Autoencoder-based editing: Use SAEs for interpretable editing
- Activation addition: Add computed directions to steer model behavior
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SteeringResult:
    """Result of latent steering operation."""

    modified_trajectory: Any
    original_trajectory: Any
    direction: torch.Tensor
    magnitude: float
    metric_change: float | None = None
    details: dict[str, Any] = None


class LatentSteering:
    """Steer latent representations toward desired outcomes."""

    def __init__(self, wm: Any):
        """Initialize steering engine.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm

    def compute_direction(
        self,
        positive_trajectories: list[Any],
        negative_trajectories: list[Any],
        component: str = "h",
    ) -> torch.Tensor:
        """Compute steering direction from positive/negative examples.

        Args:
            positive_trajectories: Trajectories with desired property
            negative_trajectories: Trajectories without desired property
            component: Component to extract

        Returns:
            Steering direction vector
        """
        pos_activations = []
        for traj in positive_trajectories:
            for state in traj.states:
                if hasattr(state, component):
                    pos_activations.append(getattr(state, component))

        neg_activations = []
        for traj in negative_trajectories:
            for state in traj.states:
                if hasattr(state, component):
                    neg_activations.append(getattr(state, component))

        pos_mean = torch.stack(pos_activations).mean(dim=0)
        neg_mean = torch.stack(neg_activations).mean(dim=0)

        direction = pos_mean - neg_mean
        return direction / (direction.norm() + 1e-8)

    def steer(
        self,
        trajectory: Any,
        direction: torch.Tensor,
        magnitude: float = 1.0,
        component: str = "h",
    ) -> SteeringResult:
        """Steer trajectory in direction.

        Args:
            trajectory: Trajectory to steer
            direction: Steering direction
            magnitude: Steering magnitude
            component: Component to modify

        Returns:
            SteeringResult
        """
        modified_states = []

        for state in trajectory.states:
            if hasattr(state, component):
                current = getattr(state, component)
                steered = current + magnitude * direction.to(current.device)

                new_state = state.__class__(
                    state=steered if component == "state" else state.state,
                    timestep=state.timestep,
                    action=state.action,
                    reward=state.reward,
                    done=state.done,
                    metadata=state.metadata.copy() if state.metadata else {},
                )
                modified_states.append(new_state)
            else:
                modified_states.append(state)

        from world_model_lens import WorldTrajectory

        modified_trajectory = WorldTrajectory(
            states=modified_states,
            source="steered",
        )

        return SteeringResult(
            modified_trajectory=modified_trajectory,
            original_trajectory=trajectory,
            direction=direction,
            magnitude=magnitude,
            details={"component": component},
        )


class ConceptVectorExtractor:
    """Extract interpretable concept vectors from latent space."""

    def __init__(self, wm: Any):
        """Initialize extractor.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm

    def extract_from_probe(
        self,
        probe_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Extract concept vector from probe weights.

        Args:
            probe_weights: Probe weight matrix

        Returns:
            Concept direction
        """
        if probe_weights.dim() == 1:
            return probe_weights / probe_weights.norm()

        return probe_weights.mean(dim=0)

    def extract_from_difference(
        self,
        states_with_concept: torch.Tensor,
        states_without_concept: torch.Tensor,
    ) -> torch.Tensor:
        """Extract concept by difference.

        Args:
            states_with_concept: States with concept
            states_without_concept: States without concept

        Returns:
            Concept direction
        """
        mean_with = states_with_concept.mean(dim=0)
        mean_without = states_without_concept.mean(dim=0)

        direction = mean_with - mean_without
        return direction / (direction.norm() + 1e-8)

    def extract_svd(
        self,
        states: torch.Tensor,
        n_components: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract principal directions using SVD.

        Args:
            states: States to analyze
            n_components: Number of components

        Returns:
            Tuple of (components, singular_values)
        """
        U, S, V = torch.svd(states)

        components = V[:n_components]
        singular_values = S[:n_components]

        return components, singular_values


class ActivationAdditionSteering:
    """Steer by adding computed directions to activations."""

    def __init__(self, wm: Any):
        """Initialize steering.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm
        self.directions: dict[str, torch.Tensor] = {}

    def add_direction(
        self,
        name: str,
        direction: torch.Tensor,
    ) -> None:
        """Add a named direction.

        Args:
            name: Direction name
            direction: Direction vector
        """
        self.directions[name] = direction

    def remove_direction(self, name: str) -> None:
        """Remove a direction.

        Args:
            name: Direction name
        """
        if name in self.directions:
            del self.directions[name]

    def apply(
        self,
        activation: torch.Tensor,
        directions: list[str] | None = None,
        magnitudes: list[float] | None = None,
    ) -> torch.Tensor:
        """Apply steering to activation.

        Args:
            activation: Input activation
            directions: List of direction names to apply
            magnitudes: Magnitudes for each direction

        Returns:
            Steered activation
        """
        result = activation.clone()

        dirs = directions or list(self.directions.keys())
        mags = magnitudes or [1.0] * len(dirs)

        for name, mag in zip(dirs, mags, strict=False):
            if name in self.directions:
                direction = self.directions[name].to(activation.device)
                result = result + mag * direction

        return result

    def get_steering_hook(
        self,
        layer_name: str,
        direction_name: str,
        magnitude: float = 1.0,
    ) -> Callable:
        """Get a hook for steering at a specific layer.

        Args:
            layer_name: Layer to hook
            direction_name: Direction to apply
            magnitude: Steering magnitude

        Returns:
            Hook function
        """
        direction = self.directions.get(direction_name)

        def hook(module, input, output):
            if direction is not None:
                return output + magnitude * direction.to(output.device)
            return output

        return hook


class LatentSpaceEditor:
    """Edit latent space using various methods."""

    def __init__(self, wm: Any):
        """Initialize editor.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm

    def project(
        self,
        state: torch.Tensor,
        target_space: str,
    ) -> torch.Tensor:
        """Project latent to different space.

        Args:
            state: Input latent
            target_space: Target space ("posterior", "prior", "action")

        Returns:
            Projected latent
        """
        if target_space == "posterior":
            return state
        elif target_space == "prior":
            prior, _ = self.wm.adapter.dynamics(state)
            return prior
        else:
            return state

    def interpolate(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Interpolate between two states.

        Args:
            state_a: First state
            state_b: Second state
            alpha: Interpolation factor

        Returns:
            Interpolated state
        """
        return (1 - alpha) * state_a + alpha * state_b

    def denoise(
        self,
        state: torch.Tensor,
        noise: float = 0.1,
        steps: int = 10,
    ) -> torch.Tensor:
        """Denoise latent using diffusion.

        Args:
            state: Noisy state
            noise: Initial noise level
            steps: Number of denoising steps

        Returns:
            Denoised state
        """
        x = state + torch.randn_like(state) * noise

        for _ in range(steps):
            x = x - 0.1 * (x - state)

        return x
