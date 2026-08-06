"""Temporal Attribution Maps.

Show which timestep influences which timestep.
Even if model doesn't have explicit attention.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np

from world_model_lens.core.hooks import HookPoint


@dataclass
class TemporalAttributionMatrix:
    """Matrix of temporal attributions."""

    matrix: np.ndarray  # [T_source, T_target]
    source_timesteps: np.ndarray
    target_timesteps: np.ndarray
    max_attribution: float
    total_attribution: float


def _make_ablate_hook(target_t: int):
    """Factory that returns a hook zeroing activations at *target_t* only.

    Using a factory avoids the classic closure-in-loop bug where all
    closures capture the same loop variable by reference.
    """
    def hook(tensor: torch.Tensor, ctx) -> torch.Tensor:
        if ctx.timestep == target_t:
            return torch.zeros_like(tensor)
        return tensor
    return hook


def _make_multi_ablate_hook(*target_ts: int):
    """Factory that zeros activations at any of the given timesteps."""
    ts = set(target_ts)
    def hook(tensor: torch.Tensor, ctx) -> torch.Tensor:
        if ctx.timestep in ts:
            return torch.zeros_like(tensor)
        return tensor
    return hook


class TemporalAttributionMap:
    """Visualize temporal causal influence.

    Shows which timesteps influence which downstream timesteps.
    Similar to attention maps but computed via interventions.

    Example:
        mapper = TemporalAttributionMap(world_model)

        # Compute influence matrix
        obs = torch.randn(20, 3, 64, 64)
        actions = torch.randn(20, d_action)
        matrix = mapper.compute_influence_matrix(obs, actions)

        # Get influence from specific timestep
        t5_influence = mapper.influence_from_timestep(obs, actions, source_t=5)

        # Causal flow
        flow = mapper.trace_causal_flow(obs, actions, source_t=0, target_t=15)
    """

    def __init__(self, world_model: Any):
        """Initialize mapper.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model
        self._cache = {}

    def compute_influence_matrix(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_metric: str = "state_norm",
    ) -> TemporalAttributionMatrix:
        """Compute full influence matrix.

        Entry (i, j) shows how much timestep i influences timestep j.

        Args:
            observations: Input observations
            actions: Action sequence
            target_metric: Metric to measure influence on

        Returns:
            TemporalAttributionMatrix
        """
        T = observations.shape[0]

        # Get baseline
        baseline_traj, _ = self.wm.run_with_cache(observations, actions)

        influence = np.zeros((T, T))

        # For each source timestep, ablate z_posterior at that step and
        # measure the downstream effect.
        for source_t in range(T):
            hp = HookPoint(
                name="z_posterior",
                stage="post",
                fn=_make_ablate_hook(source_t),
            )
            intervened_traj = self.wm.run_with_hooks(
                observations, actions, fwd_hooks=[hp],
            )

            # Measure effect at each target
            for target_t in range(T):
                baseline_val = self._extract_metric(baseline_traj, target_t, target_metric)
                intervened_val = self._extract_metric(intervened_traj, target_t, target_metric)

                influence[source_t, target_t] = abs(baseline_val - intervened_val)

        source_ts = np.arange(T)
        target_ts = np.arange(T)

        return TemporalAttributionMatrix(
            matrix=influence,
            source_timesteps=source_ts,
            target_timesteps=target_ts,
            max_attribution=influence.max(),
            total_attribution=influence.sum(),
        )

    def influence_from_timestep(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        source_t: int,
    ) -> np.ndarray:
        """Get influence from specific source timestep to all targets.

        Args:
            observations: Input observations
            actions: Action sequence
            source_t: Source timestep

        Returns:
            Array of influences to each target timestep
        """
        T = observations.shape[0]

        baseline_traj, _ = self.wm.run_with_cache(observations, actions)

        hp = HookPoint(
            name="z_posterior",
            stage="post",
            fn=_make_ablate_hook(source_t),
        )
        intervened_traj = self.wm.run_with_hooks(
            observations, actions, fwd_hooks=[hp],
        )

        influences = []

        for target_t in range(T):
            baseline_val = self._extract_metric(baseline_traj, target_t, "state_norm")
            intervened_val = self._extract_metric(intervened_traj, target_t, "state_norm")
            influences.append(abs(baseline_val - intervened_val))

        return np.array(influences)

    def influence_to_timestep(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_t: int,
    ) -> np.ndarray:
        """Get influence to specific target timestep from all sources.

        Args:
            observations: Input observations
            actions: Action sequence
            target_t: Target timestep

        Returns:
            Array of influences from each source timestep
        """
        T = observations.shape[0]

        baseline_traj, _ = self.wm.run_with_cache(observations, actions)

        target_baseline = self._extract_metric(baseline_traj, target_t, "state_norm")

        influences = []

        for source_t in range(T):
            hp = HookPoint(
                name="z_posterior",
                stage="post",
                fn=_make_ablate_hook(source_t),
            )
            intervened_traj = self.wm.run_with_hooks(
                observations, actions, fwd_hooks=[hp],
            )

            intervened_val = self._extract_metric(intervened_traj, target_t, "state_norm")
            influences.append(abs(target_baseline - intervened_val))

        return np.array(influences)

    def trace_causal_flow(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        source_t: int,
        target_t: int,
    ) -> Dict[int, float]:
        """Trace causal flow from source to target.

        Uses iterative ablation to find path.

        Args:
            observations: Input observations
            actions: Action sequence
            source_t: Source timestep
            target_t: Target timestep

        Returns:
            Dict mapping intermediate timesteps to their importance
        """
        baseline_traj, _ = self.wm.run_with_cache(observations, actions)

        baseline_target = self._extract_metric(baseline_traj, target_t, "state_norm")

        path_importance = {}

        # Check each intermediate timestep
        for mid_t in range(source_t + 1, target_t):
            hp = HookPoint(
                name="z_posterior",
                stage="post",
                fn=_make_multi_ablate_hook(source_t, mid_t),
            )
            intervened_traj = self.wm.run_with_hooks(
                observations, actions, fwd_hooks=[hp],
            )

            intervened_val = self._extract_metric(intervened_traj, target_t, "state_norm")
            path_importance[mid_t] = abs(baseline_target - intervened_val)

        return path_importance

    def find_critical_timesteps(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_t: int,
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Find most critical timesteps for target.

        Args:
            observations: Input observations
            actions: Action sequence
            target_t: Target timestep
            top_k: Number of top timesteps to return

        Returns:
            List of (timestep, importance) tuples
        """
        influences = self.influence_to_timestep(observations, actions, target_t)

        top_indices = np.argsort(influences)[-top_k:][::-1]

        return [(int(i), float(influences[i])) for i in top_indices]

    def temporal_gradient(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> np.ndarray:
        """Compute temporal gradient (change in influence over time).

        Args:
            observations: Input observations
            actions: Action sequence

        Returns:
            Array of gradients
        """
        matrix = self.compute_influence_matrix(observations, actions)

        gradient = np.gradient(matrix.matrix, axis=1)

        return gradient

    def causal_strength(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        source_t: int,
        target_t: int,
    ) -> float:
        """Compute causal strength between source and target.

        Args:
            observations: Input observations
            actions: Action sequence
            source_t: Source timestep
            target_t: Target timestep

        Returns:
            Causal strength value
        """
        influences = self.influence_from_timestep(observations, actions, source_t)

        if target_t < len(influences):
            return influences[target_t]

        return 0.0

    def _extract_metric(
        self,
        trajectory: Any,
        timestep: int,
        metric: str,
    ) -> float:
        """Extract metric from trajectory using the real LatentState API."""
        if timestep < 0:
            timestep = len(trajectory.states) + timestep

        if timestep >= len(trajectory.states):
            timestep = len(trajectory.states) - 1

        state = trajectory.states[timestep]

        if metric == "state_norm":
            # Use .flat (concatenation of h_t and z_posterior)
            return state.flat.norm().item()
        elif metric == "h_norm":
            return state.h_t.norm().item()
        elif metric == "z_norm":
            return state.z_posterior.flatten().norm().item()
        elif metric == "reward":
            r = state.reward_pred
            if r is not None:
                return float(r) if not isinstance(r, torch.Tensor) else r.item()
            return 0.0

        return 0.0

    def to_attention_style(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        normalize: bool = True,
    ) -> np.ndarray:
        """Convert to attention-style matrix (normalized, softmax-like).

        Args:
            observations: Input observations
            actions: Action sequence
            normalize: Whether to normalize

        Returns:
            Attention-style matrix
        """
        matrix = self.compute_influence_matrix(observations, actions)

        attn = matrix.matrix.copy()

        if normalize:
            # Row-normalize (each source sums to 1)
            row_sums = attn.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            attn = attn / row_sums

        return attn
