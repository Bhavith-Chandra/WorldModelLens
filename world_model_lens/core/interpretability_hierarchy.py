"""Unit of Interpretability Framework.

Defines the hierarchical structure for multi-scale interpretability.

ATOMIC UNIT HIERARCHY
=====================

Level 1: Latent Dimension (Neuron-like)
    - Single scalar value in latent vector
    - Granularity: finest
    - Use cases:
        * Which neuron detects "velocity"?
        * Which dimension encodes "object presence"?
        * Circuit analysis

Level 2: Latent Vector (State-level)
    - Full latent representation z_t
    - Granularity: medium
    - Use cases:
        * What is the agent "thinking" at this moment?
        * Representation similarity
        * State-space analysis

Level 3: Trajectory Segment (Behavior-level)
    - Sequence of latent states
    - Granularity: coarsest
    - Use cases:
        * What behavior is the agent exhibiting?
        * Causal chains over time
        * Long-horizon planning analysis

INTERPRETABILITY WORKFLOW
=========================

1. Start at Level 3 (trajectory) to find interesting time periods
2. Drill down to Level 2 (state) to understand what's happening
3. Zoom to Level 1 (dimension) to find specific mechanisms

Example:
    # Level 3: Find that agent plans between t=5 and t=10
    traj_attr = TrajectoryAttribution(wm)
    result = traj_attr.attribute(source_timestep=5, target_timestep=20)

    # Level 2: What's in the agent's mind at t=7?
    viz = PredictionVisualizer(wm)
    latent_dist = viz.latent_distribution(traj, dim=None)

    # Level 1: Which dimension matters?
    effect = CausalEffectEstimator(wm)
    dim_effect = effect.estimate_effect(source=(0, 7), target="reward")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import torch
import numpy as np


class InterpretabilityLevel(Enum):
    """Three levels of interpretability."""

    DIMENSION = "dimension"  # Level 1: Neuron-like
    STATE = "state"  # Level 2: Latent vector
    TRAJECTORY = "trajectory"  # Level 3: Sequence


@dataclass
class InterpretabilityUnit:
    """Base class for interpretability units."""

    level: InterpretabilityLevel
    identifier: Any
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "identifier": str(self.identifier),
            "description": self.description,
        }


@dataclass
class DimensionUnit(InterpretabilityUnit):
    """Level 1: Single latent dimension.

    Attributes:
        dim: Dimension index
        timestep: Which timestep (if applicable)
        importance: Estimated importance score
        concepts: Associated concepts (if discovered)
    """

    def __init__(
        self,
        dim: int,
        timestep: Optional[int] = None,
        importance: float = 0.0,
        concepts: Optional[List[str]] = None,
    ):
        super().__init__(
            level=InterpretabilityLevel.DIMENSION,
            identifier=(dim, timestep),
            description=f"Dimension {dim}" + (f" at t={timestep}" if timestep else ""),
        )
        self.dim = dim
        self.timestep = timestep
        self.importance = importance
        self.concepts = concepts or []


@dataclass
class StateUnit(InterpretabilityUnit):
    """Level 2: Full latent state.

    Attributes:
        timestep: Timestep of this state
        latent_vector: The actual latent values
        norm: Magnitude of latent
        active_dims: Dimensions with highest activation
    """

    def __init__(
        self,
        timestep: int,
        latent_vector: torch.Tensor,
        active_dims: Optional[List[int]] = None,
    ):
        super().__init__(
            level=InterpretabilityLevel.STATE,
            identifier=timestep,
            description=f"State at t={timestep}",
        )
        self.timestep = timestep
        self.latent_vector = latent_vector
        self.norm = latent_vector.norm().item()
        self.active_dims = active_dims or []


@dataclass
class TrajectoryUnit(InterpretabilityUnit):
    """Level 3: Trajectory segment.

    Attributes:
        start: Start timestep
        end: End timestep
        states: List of latent states
        behavior: Identified behavior type
        causal_effect: Effect on outcome
    """

    def __init__(
        self,
        start: int,
        end: int,
        states: Optional[List[torch.Tensor]] = None,
        behavior: str = "",
        causal_effect: float = 0.0,
    ):
        super().__init__(
            level=InterpretabilityLevel.TRAJECTORY,
            identifier=(start, end),
            description=f"Trajectory t={start}:{end}",
        )
        self.start = start
        self.end = end
        self.states = states or []
        self.behavior = behavior
        self.causal_effect = causal_effect


class InterpretabilityHierarchy:
    """Manages the interpretability hierarchy.

    Provides methods to navigate between levels.

    Example:
        hierarchy = InterpretabilityHierarchy(world_model)

        # Find interesting trajectory segment (Level 3)
        traj_unit = hierarchy.find_behavior(
            observations,
            target_metric="reward",
        )

        # Zoom to state at specific timestep (Level 2)
        state_unit = hierarchy.examine_state(
            traj_unit.start + 3,
            trajectory,
        )

        # Find critical dimension (Level 1)
        dim_unit = hierarchy.find_critical_dimension(
            state_unit.timestep,
            trajectory,
        )
    """

    def __init__(self, world_model: Any):
        """Initialize hierarchy manager.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def find_behavior(
        self,
        observations: torch.Tensor,
        target_metric: str = "reward_pred",
    ) -> TrajectoryUnit:
        """Find interesting trajectory segment (Level 3).

        Args:
            observations: Input observations
            target_metric: What to optimize for

        Returns:
            TrajectoryUnit with behavior analysis
        """
        from world_model_lens.causal.trajectory_attribution import TrajectoryAttribution

        traj, _, _ = self.wm.run_with_cache(observations)

        T = len(traj.states)

        best_start = 0
        best_effect = 0.0

        attr = TrajectoryAttribution(self.wm)

        for start in range(T - 1):
            for end in range(start + 1, T):
                result = attr.attribute(
                    source_timestep=start,
                    target_timestep=end,
                    target_metric=target_metric,
                )

                if abs(result.total_effect) > abs(best_effect):
                    best_effect = result.total_effect
                    best_start = start

        return TrajectoryUnit(
            start=best_start,
            end=T - 1,
            causal_effect=best_effect,
        )

    def examine_state(
        self,
        timestep: int,
        trajectory: Any,
    ) -> StateUnit:
        """Examine state at specific timestep (Level 2).

        Args:
            timestep: Which timestep
            trajectory: WorldTrajectory

        Returns:
            StateUnit with latent analysis
        """
        if timestep >= len(trajectory.states):
            timestep = len(trajectory.states) - 1

        state = trajectory.states[timestep]

        latent = state.obs_encoding if state.obs_encoding is not None else state.state

        if latent.dim() > 1:
            latent = latent.squeeze(0)

        # Find most active dimensions
        abs_vals = latent.abs()
        topk = min(10, len(abs_vals))
        _, top_indices = torch.topk(abs_vals, topk)

        return StateUnit(
            timestep=timestep,
            latent_vector=latent.detach().cpu(),
            active_dims=top_indices.tolist(),
        )

    def find_critical_dimension(
        self,
        timestep: int,
        trajectory: Any,
        target_metric: str = "reward_pred",
    ) -> DimensionUnit:
        """Find most critical dimension at timestep (Level 1).

        Args:
            timestep: Which timestep
            trajectory: WorldTrajectory
            target_metric: What to attribute to

        Returns:
            DimensionUnit with dimension analysis
        """
        from world_model_lens.causal.effect_estimator import (
            CausalEffectEstimator,
            InterventionSpec,
        )

        state = trajectory.states[timestep]
        latent = state.obs_encoding if state.obs_encoding is not None else state.state

        if latent.dim() > 1:
            latent = latent.squeeze(0)

        d_z = latent.shape[-1]

        effect_estimator = CausalEffectEstimator(self.wm, num_samples=10)

        best_dim = 0
        best_importance = 0.0

        for dim in range(min(d_z, 20)):
            intervention = InterventionSpec(
                target_type="dimension",
                target_indices=[dim],
                intervention_type="ablation",
            )

            effect = effect_estimator.estimate_effect(
                source=(dim, timestep),
                target=target_metric,
                intervention=intervention,
            )

            if abs(effect.effect_size) > abs(best_importance):
                best_importance = effect.effect_size
                best_dim = dim

        return DimensionUnit(
            dim=best_dim,
            timestep=timestep,
            importance=best_importance,
        )

    def drill_down(
        self,
        observations: torch.Tensor,
        focus_timestep: int,
    ) -> Dict[str, Any]:
        """Drill from trajectory to dimension (Level 3 -> 2 -> 1).

        Args:
            observations: Input observations
            focus_timestep: Where to focus

        Returns:
            Dict with all three levels of analysis
        """
        traj, _, _ = self.wm.run_with_cache(observations)

        # Level 2: Examine state
        state_unit = self.examine_state(focus_timestep, traj)

        # Level 1: Find critical dimension
        dim_unit = self.find_critical_dimension(focus_timestep, traj)

        return {
            "level_1_dimension": dim_unit.to_dict(),
            "level_2_state": state_unit.to_dict(),
            "level_3_trajectory": {
                "start": 0,
                "end": len(traj.states) - 1,
                "length": len(traj.states),
            },
        }
