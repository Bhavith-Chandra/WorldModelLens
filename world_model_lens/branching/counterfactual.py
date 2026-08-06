"""Counterfactual analysis for world model trajectories.

This module provides tools for answering "what if" questions about world model
behavior. It supports:
- State counterfactuals: "What if the state were different?"
- Action counterfactuals: "What if I took a different action?"
- Intervention counterfactuals: "What if I intervened at this point?"
- Counterfactual comparison and attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Optional
import torch
import numpy as np


@dataclass
class CounterfactualResult:
    """Result of a counterfactual analysis."""

    original_trajectory: Any
    counterfactual_trajectory: Any
    intervention_point: int
    intervention_type: str
    difference_metric: float
    state_differences: torch.Tensor
    latent_differences: torch.Tensor | None = None
    attribution: dict[str, float] = field(default_factory=dict)

    def compare_observations(self) -> dict[str, float]:
        """Compare observations between original and counterfactual."""
        if not hasattr(self.original_trajectory, "observations") or not hasattr(
            self.counterfactual_trajectory, "observations"
        ):
            return {}

        orig_obs = self.original_trajectory.observations
        cf_obs = self.counterfactual_trajectory.observations

        if orig_obs is None or cf_obs is None:
            return {}

        mse = torch.nn.functional.mse_loss(orig_obs, cf_obs).item()
        mae = torch.nn.functional.l1_loss(orig_obs, cf_obs).item()

        return {
            "mse": mse,
            "mae": mae,
        }

    def compute_divergence(self) -> float:
        """Compute KL divergence between trajectories."""
        if self.state_differences is None:
            return 0.0

        states_orig = self.original_trajectory.states
        states_cf = self.counterfactual_trajectory.states

        if states_orig is None or states_cf is None:
            return 0.0

        divergence = 0.0
        for i in range(min(len(states_orig), len(states_cf))):
            s_orig = states_orig[i].state
            s_cf = states_cf[i].state

            p = torch.softmax(s_orig, dim=-1)
            q = torch.softmax(s_cf, dim=-1)
            divergence += torch.nn.functional.kl_div(
                torch.log(q + 1e-8), p, reduction="batchmean"
            ).item()

        return divergence / min(len(states_orig), len(states_cf))


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual generation."""

    intervention_type: str = "state"
    num_samples: int = 1
    noise_scale: float = 0.1
    intervention_components: list[str] = field(default_factory=lambda: ["state"])
    preserve_other_components: bool = True


class CounterfactualGenerator:
    """Generate counterfactual trajectories for world model analysis."""

    def __init__(
        self,
        wm: Any,
        config: Optional[CounterfactualConfig] = None,
    ):
        """Initialize counterfactual generator.

        Args:
            wm: HookedWorldModel instance
            config: Counterfactual configuration
        """
        self.wm = wm
        self.config = config or CounterfactualConfig()

    def generate_state_counterfactual(
        self,
        trajectory: Any,
        intervention_point: int,
        new_state: torch.Tensor,
    ) -> CounterfactualResult:
        """Generate counterfactual by modifying state at intervention point.

        Args:
            trajectory: Original trajectory
            intervention_point: Timestep to intervene
            new_state: New state to inject

        Returns:
            CounterfactualResult with comparison
        """
        original_states = [s.state for s in trajectory.states]
        original_trajectory = trajectory

        cf_states = original_states.copy()
        cf_states[intervention_point] = new_state

        cf_trajectory = self._build_counterfactual_trajectory(trajectory, cf_states)

        state_diffs = torch.stack([torch.norm(a - b) for a, b in zip(original_states, cf_states)])

        return CounterfactualResult(
            original_trajectory=original_trajectory,
            counterfactual_trajectory=cf_trajectory,
            intervention_point=intervention_point,
            intervention_type="state",
            difference_metric=state_diffs.mean().item(),
            state_differences=state_diffs,
        )

    def generate_action_counterfactual(
        self,
        trajectory: Any,
        intervention_point: int,
        new_action: torch.Tensor,
    ) -> CounterfactualResult:
        """Generate counterfactual by modifying action at intervention point.

        Args:
            trajectory: Original trajectory
            intervention_point: Timestep to intervene
            new_action: New action to use

        Returns:
            CounterfactualResult with comparison
        """
        original_actions = trajectory.actions
        original_trajectory = trajectory

        cf_actions = original_actions.clone() if original_actions is not None else None
        if cf_actions is not None:
            cf_actions[intervention_point] = new_action

        cf_trajectory = self._run_with_modified_actions(trajectory, intervention_point, cf_actions)

        state_diffs = torch.stack(
            [torch.norm(a.state - b.state) for a, b in zip(trajectory.states, cf_trajectory.states)]
        )

        return CounterfactualResult(
            original_trajectory=original_trajectory,
            counterfactual_trajectory=cf_trajectory,
            intervention_point=intervention_point,
            intervention_type="action",
            difference_metric=state_diffs.mean().item(),
            state_differences=state_diffs,
        )

    def generate_noise_counterfactual(
        self,
        trajectory: Any,
        intervention_point: int,
        noise_scale: float | None = None,
    ) -> CounterfactualResult:
        """Generate counterfactual by adding noise to latent state.

        Args:
            trajectory: Original trajectory
            intervention_point: Timestep to intervene
            noise_scale: Scale of noise to add

        Returns:
            CounterfactualResult with comparison
        """
        scale = noise_scale or self.config.noise_scale
        state = trajectory.states[intervention_point].state
        noise = torch.randn_like(state) * scale
        noisy_state = state + noise

        return self.generate_state_counterfactual(trajectory, intervention_point, noisy_state)

    def generate_intervention_counterfactual(
        self,
        trajectory: Any,
        intervention_point: int,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> CounterfactualResult:
        """Generate counterfactual with custom intervention function.

        Args:
            trajectory: Original trajectory
            intervention_point: Timestep to intervene
            intervention_fn: Function that transforms the state

        Returns:
            CounterfactualResult with comparison
        """
        original_state = trajectory.states[intervention_point].state
        new_state = intervention_fn(original_state)

        return self.generate_state_counterfactual(trajectory, intervention_point, new_state)

    def _build_counterfactual_trajectory(
        self,
        original_trajectory: Any,
        states: list[torch.Tensor],
    ) -> Any:
        """Build counterfactual trajectory from modified states."""
        from world_model_lens import WorldTrajectory, WorldState

        new_states = []
        for i, state_tensor in enumerate(states):
            orig = original_trajectory.states[i]
            new_state = WorldState(
                state=state_tensor,
                timestep=i,
                action=orig.action,
                reward=orig.reward,
                done=orig.done,
                metadata=orig.metadata.copy() if orig.metadata else {},
            )
            new_states.append(new_state)

        return WorldTrajectory(
            states=new_states,
            source="counterfactual",
        )

    def _run_with_modified_actions(
        self,
        trajectory: Any,
        intervention_point: int,
        actions: torch.Tensor | None,
    ) -> Any:
        """Run model with modified actions from intervention point."""
        from world_model_lens import WorldTrajectory, WorldState

        start_state = trajectory.states[0].state
        new_states = [trajectory.states[0]]

        for t in range(1, len(trajectory.states)):
            if actions is not None and t >= intervention_point:
                action = actions[t]
            else:
                action = trajectory.states[t].action

            action_input = action.unsqueeze(0) if action is not None else None
            next_state, _ = self.wm.adapter.dynamics(new_states[-1].state, action_input)

            new_state = WorldState(
                state=next_state.squeeze(0),
                timestep=t,
                action=action,
                reward=trajectory.states[t].reward,
                done=trajectory.states[t].done,
            )
            new_states.append(new_state)

        return WorldTrajectory(states=new_states, source="counterfactual")

    def compare_counterfactuals(
        self,
        results: list[CounterfactualResult],
    ) -> dict[str, Any]:
        """Compare multiple counterfactual results.

        Args:
            results: List of CounterfactualResult to compare

        Returns:
            Dictionary with comparison metrics
        """
        if not results:
            return {}

        metrics = {
            "num_counterfactuals": len(results),
            "mean_difference": np.mean([r.difference_metric for r in results]),
            "std_difference": np.std([r.difference_metric for r in results]),
            "max_difference": max(r.difference_metric for r in results),
            "min_difference": min(r.difference_metric for r in results),
        }

        intervention_types = {}
        for r in results:
            it = r.intervention_type
            intervention_types[it] = intervention_types.get(it, 0) + 1
        metrics["intervention_types"] = intervention_types

        return metrics


class AttributionAnalyzer:
    """Analyze attribution in counterfactual trajectories."""

    def __init__(self, wm: Any):
        """Initialize attribution analyzer.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm

    def attribute_difference(
        self,
        counterfactual_result: CounterfactualResult,
    ) -> dict[str, float]:
        """Attribute trajectory difference to components.

        Args:
            counterfactual_result: Result to analyze

        Returns:
            Dictionary mapping component names to attribution scores
        """
        original = counterfactual_result.original_trajectory
        cf = counterfactual_result.counterfactual_trajectory
        intervention_point = counterfactual_result.intervention_point

        attribution = {}

        for t in range(intervention_point, len(original.states)):
            orig_state = original.states[t].state
            cf_state = cf.states[t].state

            diff = torch.norm(orig_state - cf_state).item()
            attribution[f"timestep_{t}"] = diff

        total = sum(attribution.values())
        if total > 0:
            attribution = {k: v / total for k, v in attribution.items()}

        return attribution

    def find_critical_points(
        self,
        counterfactual_result: CounterfactualResult,
        threshold: float = 0.1,
    ) -> list[int]:
        """Find critical points where divergence starts.

        Args:
            counterfactual_result: Result to analyze
            threshold: Threshold for divergence detection

        Returns:
            List of critical timestep indices
        """
        state_diffs = counterfactual_result.state_differences
        critical = []

        for i in range(1, len(state_diffs)):
            if state_diffs[i] - state_diffs[i - 1] > threshold:
                critical.append(i)

        return critical
