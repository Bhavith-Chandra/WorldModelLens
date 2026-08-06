"""Counterfactual Engine for World Models.

Enables branching rollouts and divergence analysis for causal understanding.

Key Capabilities:
1. Branch Trees: Fork trajectories at any point
2. Divergence Metrics: Quantify how counterfactuals diverge
3. Rollout Comparison: Compare alternative futures

Mathematical Framework:
- Original trajectory: τ_o = [z_0, z_1, ..., z_T]
- Counterfactual: τ_c = [z_0, ..., z_k', z_{k+1}, ..., z_T']
- Divergence: D(τ_o, τ_c)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from world_model_lens.core.hooks import HookPoint


@dataclass
class BranchTree:
    """Tree of counterfactual branches.

    Structure:
        root: Original trajectory
        branches: List of child branches, each with:
            - fork_point: Where branch diverged
            - intervention: What was changed
            - trajectory: Resulting trajectory
            - divergence: How much it diverged from original
    """

    root_trajectory: Any
    branches: List["Branch"] = field(default_factory=list)
    max_depth: int = 5

    def add_branch(
        self,
        fork_point: int,
        intervention: "Intervention",
        trajectory: Any,
        divergence: float,
    ) -> "Branch":
        """Add a counterfactual branch."""
        branch = Branch(
            fork_point=fork_point,
            intervention=intervention,
            trajectory=trajectory,
            divergence=divergence,
        )
        self.branches.append(branch)
        return branch


@dataclass
class Branch:
    """A single counterfactual branch."""

    fork_point: int
    intervention: "Intervention"
    trajectory: Any
    divergence: float
    parent: Optional["Branch"] = None
    children: List["Branch"] = field(default_factory=list)


@dataclass
class Intervention:
    """Specification for counterfactual intervention."""

    target_timestep: int
    target_type: str  # "dimension", "state", "action"
    target_indices: Optional[List[int]] = None
    intervention_fn: Optional[Callable] = None
    description: str = ""


@dataclass
class DivergenceMetrics:
    """Quantitative divergence between trajectories.

    Multiple metrics for comprehensive comparison:
    - L2 distance: Euclidean distance between states
    - Cosine distance: Angular difference
    - KL divergence: Distribution divergence (if available)
    - Behavioral: Outcome-level differences
    """

    l2_distance: float
    cosine_similarity: float
    kl_divergence: Optional[float] = None
    reward_difference: float = 0.0
    final_state_distance: float = 0.0
    trajectory_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            "l2_distance": self.l2_distance,
            "cosine_similarity": self.cosine_similarity,
            "reward_difference": self.reward_difference,
            "final_state_distance": self.final_state_distance,
            "trajectory_distance": self.trajectory_distance,
        }
        if self.kl_divergence is not None:
            result["kl_divergence"] = self.kl_divergence
        return result


def rollout_comparison(
    original: Any,
    counterfactual: Any,
    metrics: List[str] = None,
) -> DivergenceMetrics:
    """Compare two trajectories.

    Args:
        original: Original WorldTrajectory
        counterfactual: Counterfactual WorldTrajectory
        metrics: Which metrics to compute

    Returns:
        DivergenceMetrics with comparison
    """
    if metrics is None:
        metrics = ["l2", "cosine", "behavioral"]

    # Compute state-level distances
    l2_distances = []
    cos_distances = []

    for t in range(min(len(original.states), len(counterfactual.states))):
        s_orig = original.states[t].state.flatten()
        s_cf = counterfactual.states[t].state.flatten()

        l2_distances.append(torch.nn.functional.mse_loss(s_orig, s_cf).item())
        cos_sim = torch.nn.functional.cosine_similarity(
            s_orig.unsqueeze(0), s_cf.unsqueeze(0)
        ).item()
        cos_distances.append(1 - cos_sim)

    avg_l2 = np.mean(l2_distances) if l2_distances else 0.0
    avg_cos = np.mean(cos_distances) if cos_distances else 1.0

    # Final state distance
    final_orig = original.states[-1].state.flatten()
    final_cf = counterfactual.states[-1].state.flatten()
    final_dist = torch.nn.functional.mse_loss(final_orig, final_cf).item()

    # Reward difference
    def _reward_t(s: Any) -> torch.Tensor:
        r = s.reward if getattr(s, "reward", None) is not None else getattr(s, "reward_pred", None)
        if r is None and hasattr(s, "predictions") and s.predictions:
            r = s.predictions.get("reward")
        return r if r is not None else torch.tensor(0.0)

    orig_rewards = [_reward_t(s) for s in original.states]
    cf_rewards = [_reward_t(s) for s in counterfactual.states]

    orig_sum = sum(r.item() if isinstance(r, torch.Tensor) else r for r in orig_rewards)
    cf_sum = sum(r.item() if isinstance(r, torch.Tensor) else r for r in cf_rewards)
    reward_diff = abs(orig_sum - cf_sum)

    # Trajectory-level (integral of distances)
    traj_dist = sum(l2_distances)

    return DivergenceMetrics(
        l2_distance=avg_l2,
        cosine_similarity=avg_cos,
        reward_difference=reward_diff,
        final_state_distance=final_dist,
        trajectory_distance=traj_dist,
    )


class CounterfactualEngine:
    """Engine for generating and analyzing counterfactuals.

    Example:
        engine = CounterfactualEngine(world_model)

        # Single counterfactual
        cf = engine.intervene(
            observations=obs,
            intervention=Intervention(
                target_timestep=5,
                target_type="dimension",
                target_indices=[0, 1, 2],
            ),
        )

        # Build branch tree
        tree = engine.build_branch_tree(
            observations=obs,
            interventions=[
                Intervention(target_timestep=5, target_indices=[0]),
                Intervention(target_timestep=5, target_indices=[1]),
                Intervention(target_timestep=5, target_indices=[2]),
            ],
        )

        # Analyze divergence over time
        divergence_curve = engine.trace_divergence(
            original=traj,
            counterfactual=cf,
        )
    """

    def __init__(self, world_model: Any):
        """Initialize counterfactual engine.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def intervene(
        self,
        observations: torch.Tensor,
        intervention: Intervention,
        actions: Optional[torch.Tensor] = None,
    ) -> Any:
        """Generate counterfactual trajectory.

        Args:
            observations: Original observations
            intervention: What to intervene on
            actions: Original actions

        Returns:
            Counterfactual WorldTrajectory
        """
        intervention_fn = self._make_intervention_fn(intervention)
        component = self._hook_component_for_intervention(intervention)
        hook = HookPoint(
            name=component,
            fn=intervention_fn,
            timestep=intervention.target_timestep,
        )

        cf_traj = self.wm.run_with_hooks(
            observations=observations,
            actions=actions,
            fwd_hooks=[hook],
        )

        return cf_traj

    def _hook_component_for_intervention(self, intervention: Intervention) -> str:
        """Map intervention target to HookedWorldModel component names (see run_with_cache)."""
        if intervention.target_type == "dimension":
            return "z_posterior"
        if intervention.target_type == "action":
            return "action"
        if intervention.target_type == "state":
            return "state"
        return "z_posterior"

    def _make_intervention_fn(self, intervention: Intervention) -> Callable:
        """Create intervention function from intervention spec."""
        if intervention.intervention_fn is not None:
            return intervention.intervention_fn

        def ablate_fn(tensor, ctx):
            if ctx.timestep != intervention.target_timestep:
                return tensor
            result = tensor.clone()

            if intervention.target_type == "dimension" and intervention.target_indices:
                for dim in intervention.target_indices:
                    if dim < result.shape[-1]:
                        result[..., dim] = 0
            else:
                result = result * 0

            return result

        return ablate_fn

    def build_branch_tree(
        self,
        observations: torch.Tensor,
        interventions: List[Intervention],
        base_actions: Optional[torch.Tensor] = None,
    ) -> BranchTree:
        """Build tree of counterfactual branches.

        Args:
            observations: Base observations
            interventions: List of interventions to apply
            base_actions: Base actions

        Returns:
            BranchTree with all branches
        """
        # Get original trajectory
        original_traj, _ = self.wm.run_with_cache(observations, base_actions)

        tree = BranchTree(root_trajectory=original_traj)

        for intervention in interventions:
            cf_traj = self.intervene(observations, intervention, base_actions)

            divergence = rollout_comparison(original_traj, cf_traj)

            tree.add_branch(
                fork_point=intervention.target_timestep,
                intervention=intervention,
                trajectory=cf_traj,
                divergence=divergence.trajectory_distance,
            )

        return tree

    def trace_divergence(
        self,
        original: Any,
        counterfactual: Any,
    ) -> Dict[int, float]:
        """Trace how divergence grows over time.

        Args:
            original: Original trajectory
            counterfactual: Counterfactual trajectory

        Returns:
            Dict mapping timestep to cumulative divergence
        """
        divergence_curve = {}
        cumulative = 0.0

        for t in range(min(len(original.states), len(counterfactual.states))):
            s_orig = original.states[t].state.flatten()
            s_cf = counterfactual.states[t].state.flatten()

            step_div = torch.nn.functional.mse_loss(s_orig, s_cf).item()
            cumulative += step_div
            divergence_curve[t] = cumulative

        return divergence_curve

    def counterfactual_rollout(
        self,
        start_state: Any,
        intervention: Intervention,
        horizon: int = 50,
    ) -> Any:
        """Generate counterfactual rollout from start state.

        Args:
            start_state: WorldState to start from
            intervention: Intervention to apply
            horizon: How many steps to rollout

        Returns:
            Counterfactual trajectory
        """
        # Create dummy observations for the rollout
        dummy_obs = torch.randn(horizon, 3, 64, 64)

        intervention_fn = self._make_intervention_fn(intervention)
        component = self._hook_component_for_intervention(intervention)
        hook = HookPoint(
            name=component,
            fn=intervention_fn,
            timestep=intervention.target_timestep,
        )

        cf_traj = self.wm.run_with_hooks(
            dummy_obs,
            actions=None,
            fwd_hooks=[hook],
        )

        return cf_traj

    def compare_interventions(
        self,
        observations: torch.Tensor,
        interventions: List[Intervention],
        base_actions: Optional[torch.Tensor] = None,
        target_metric: str = "reward_pred",
    ) -> Dict[int, Dict[str, float]]:
        """Compare multiple interventions.

        Args:
            observations: Base observations
            interventions: List of interventions to compare
            base_actions: Base actions
            target_metric: Metric to compare on

        Returns:
            Dict mapping intervention index to metrics
        """
        # Get baseline
        baseline_traj, _ = self.wm.run_with_cache(observations, base_actions)
        baseline_outcome = self._extract_outcome(baseline_traj, target_metric)

        results = {}

        for i, intervention in enumerate(interventions):
            cf_traj = self.intervene(observations, intervention, base_actions)
            cf_outcome = self._extract_outcome(cf_traj, target_metric)

            divergence = rollout_comparison(baseline_traj, cf_traj)

            results[i] = {
                "intervention_description": intervention.description,
                "target_timestep": intervention.target_timestep,
                "baseline_outcome": baseline_outcome,
                "counterfactual_outcome": cf_outcome,
                "outcome_delta": cf_outcome - baseline_outcome,
                **divergence.to_dict(),
            }

        return results

    def _extract_outcome(self, trajectory: Any, metric: str) -> float:
        """Extract outcome metric from trajectory."""
        if metric == "reward_pred":
            rewards = []
            for s in trajectory.states:
                r = (
                    s.reward
                    if getattr(s, "reward", None) is not None
                    else getattr(s, "reward_pred", None)
                )
                if r is None and hasattr(s, "predictions") and s.predictions:
                    r = s.predictions.get("reward")
                rewards.append(r if r is not None else torch.tensor(0.0))
            return sum(r.item() if isinstance(r, torch.Tensor) else r for r in rewards)
        elif metric == "final_state_norm":
            return trajectory.states[-1].state.norm().item()
        return 0.0

    def find_critical_timestep(
        self,
        observations: torch.Tensor,
        target_state: torch.Tensor,
    ) -> int:
        """Find the timestep where intervention has biggest effect.

        Args:
            observations: Input observations
            target_state: Target state to match

        Returns:
            Critical timestep index
        """
        baseline_traj, _ = self.wm.run_with_cache(observations)

        best_timestep = 0
        best_divergence = float("inf")

        for t in range(len(baseline_traj.states)):
            intervention = Intervention(
                target_timestep=t,
                target_type="state",
            )

            cf_traj = self.intervene(observations, intervention)

            current_state = cf_traj.states[-1].state
            divergence = torch.nn.functional.mse_loss(current_state, target_state).item()

            if divergence < best_divergence:
                best_divergence = divergence
                best_timestep = t

        return best_timestep
