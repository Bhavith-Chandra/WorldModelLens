"""Intervention Visualization.

Show effects of patches/counterfactuals.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import numpy as np


@dataclass
class InterventionResult:
    """Result of intervention visualization."""

    before_trajectory: Any
    after_trajectory: Any
    divergence_curve: Dict[int, float]
    total_divergence: float
    affected_timesteps: List[int]


class InterventionVisualizer:
    """Visualize effects of interventions.

    Example:
        viz = InterventionVisualizer(world_model)

        # Get original and intervened trajectories
        obs = torch.randn(20, 3, 64, 64)
        traj, _ = world_model.run_with_cache(obs)

        # Create intervention
        def ablate_hook(tensor, ctx):
            return tensor * 0

        cf_traj = world_model.run_with_advanced_hooks(
            obs,
            hook_specs={"t=5.z": ablate_hook},
        )

        # Visualize divergence
        result = viz.visualize_intervention(traj, cf_traj)

        # Divergence curve
        curve = viz.divergence_curve(traj, cf_traj)

        # State comparison
        comparison = viz.compare_states(traj, cf_traj, timestep=10)
    """

    def __init__(self, world_model: Any):
        """Initialize visualizer.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def visualize_intervention(
        self,
        before_trajectory: Any,
        after_trajectory: Any,
    ) -> InterventionResult:
        """Visualize intervention effect.

        Args:
            before_trajectory: Original trajectory
            after_trajectory: Intervened trajectory

        Returns:
            InterventionResult with divergence metrics
        """
        divergence_curve = self.divergence_curve(before_trajectory, after_trajectory)
        total = sum(divergence_curve.values())

        affected = [t for t, d in divergence_curve.items() if d > 0.01]

        return InterventionResult(
            before_trajectory=before_trajectory,
            after_trajectory=after_trajectory,
            divergence_curve=divergence_curve,
            total_divergence=total,
            affected_timesteps=affected,
        )

    def divergence_curve(
        self,
        before_trajectory: Any,
        after_trajectory: Any,
    ) -> Dict[int, float]:
        """Compute divergence over time.

        Args:
            before_trajectory: Original trajectory
            after_trajectory: Intervened trajectory

        Returns:
            Dict mapping timestep to divergence
        """
        divergence = {}

        for t in range(min(len(before_trajectory.states), len(after_trajectory.states))):
            s_before = before_trajectory.states[t].flat
            s_after = after_trajectory.states[t].flat

            div = torch.nn.functional.mse_loss(s_before, s_after).item()
            divergence[t] = div

        return divergence

    def cumulative_divergence(
        self,
        before_trajectory: Any,
        after_trajectory: Any,
    ) -> Dict[int, float]:
        """Compute cumulative divergence.

        Args:
            before_trajectory: Original trajectory
            after_trajectory: Intervened trajectory

        Returns:
            Dict mapping timestep to cumulative divergence
        """
        curve = self.divergence_curve(before_trajectory, after_trajectory)

        cumulative = {}
        running = 0.0

        for t, div in sorted(curve.items()):
            running += div
            cumulative[t] = running

        return cumulative

    def compare_states(
        self,
        before_trajectory: Any,
        after_trajectory: Any,
        timestep: int,
    ) -> dict:
        """Compare states at specific timestep.

        Args:
            before_trajectory: Original trajectory
            after_trajectory: Intervened trajectory
            timestep: Which timestep to compare

        Returns:
            Dict with before, after, difference
        """
        if timestep >= len(before_trajectory.states):
            timestep = len(before_trajectory.states) - 1
        if timestep >= len(after_trajectory.states):
            timestep = len(after_trajectory.states) - 1

        s_before = before_trajectory.states[timestep].flat
        s_after = after_trajectory.states[timestep].flat

        diff = (s_before - s_after).abs()

        return {
            "before": s_before.detach().cpu(),
            "after": s_after.detach().cpu(),
            "difference": diff.detach().cpu(),
            "l2_distance": torch.nn.functional.mse_loss(s_before, s_after).item(),
        }

    def intervention_heatmap(
        self,
        before_trajectory: Any,
        after_trajectory: Any,
    ) -> np.ndarray:
        """Create heatmap of intervention effects across timesteps.

        Args:
            before_trajectory: Original trajectory
            after_trajectory: Intervened trajectory

        Returns:
            2D array [T, d_z] of effects
        """
        T = min(len(before_trajectory.states), len(after_trajectory.states))
        # Flatten the first state to get the true feature count
        d_z = before_trajectory.states[0].flat.shape[0]

        heatmap = np.zeros((T, d_z))

        for t in range(T):
            s_before = before_trajectory.states[t].flat
            s_after = after_trajectory.states[t].flat

            diff = (s_before - s_after).abs()
            n = min(len(diff), d_z)
            heatmap[t, :n] = diff[:n].detach().cpu().numpy()

        return heatmap

    def dimension_importance(
        self,
        trajectory: Any,
        cache: Any,
    ) -> np.ndarray:
        """Compute importance of each dimension.

        Args:
            trajectory: WorldTrajectory
            cache: ActivationCache

        Returns:
            Array of dimension importance scores
        """
        # Flatten state to get true dimensionality
        d_z = trajectory.states[0].flat.shape[0]
        importance = np.zeros(d_z)

        for state in trajectory.states:
            s = state.flat
            importance[:len(s)] += s.abs().detach().cpu().numpy()[:d_z]

        return importance / max(len(trajectory.states), 1)

    def intervention_summary(
        self,
        before_trajectory: Any,
        after_trajectory: Any,
    ) -> dict:
        """Get summary of intervention effects.

        Args:
            before_trajectory: Original trajectory
            after_trajectory: Intervened trajectory

        Returns:
            Dict with summary statistics
        """
        div_curve = self.divergence_curve(before_trajectory, after_trajectory)
        cum_div = self.cumulative_divergence(before_trajectory, after_trajectory)

        final_before = before_trajectory.states[-1].flat
        final_after = after_trajectory.states[-1].flat

        return {
            "total_divergence": sum(div_curve.values()),
            "max_single_step_div": max(div_curve.values()) if div_curve else 0,
            "final_state_l2": torch.nn.functional.mse_loss(final_before, final_after).item(),
            "final_cumulative_div": cum_div[max(cum_div.keys())] if cum_div else 0,
            "affected_timesteps": sum(1 for d in div_curve.values() if d > 0.01),
        }

    def plot_before_after_comparison(
        self,
        before_trajectory: Any,
        after_trajectory: Any,
        timesteps: List[int],
    ) -> dict:
        """Get before/after comparison for plotting.

        Args:
            before_trajectory: Original trajectory
            after_trajectory: Intervened trajectory
            timesteps: Which timesteps to compare

        Returns:
            Dict with comparison data
        """
        result = {
            "timesteps": [],
            "before_norms": [],
            "after_norms": [],
            "divergences": [],
        }

        for t in timesteps:
            if t < len(before_trajectory.states) and t < len(after_trajectory.states):
                s_before = before_trajectory.states[t].flat.norm().item()
                s_after = after_trajectory.states[t].flat.norm().item()
                div = abs(s_before - s_after)

                result["timesteps"].append(t)
                result["before_norms"].append(s_before)
                result["after_norms"].append(s_after)
                result["divergences"].append(div)

        return result
