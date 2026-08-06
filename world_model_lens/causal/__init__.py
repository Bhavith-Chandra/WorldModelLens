"""Causal Interpretability Framework.

This module provides a formal causal framework for world model interpretability:

Level 1 (Atomic): Latent Dimension
    - Single neuron-level causal effects
    - Dimension ablation studies

Level 2 (State): Latent Vector
    - Full state-level interventions
    - Representation-level causality

Level 3 (Trajectory): Sequence-Leveli
    - Temporal causal chains
    - Long-horizon effect propagation

Key Components:
- CausalEffectEstimator: Formal A/B testing for world models
- TrajectoryAttribution: Time-indexed causal tracing
- CounterfactualEngine: Branching rollouts and divergence analysis
"""

from world_model_lens.causal.counterfactual import (
    BranchTree,
    CounterfactualEngine,
    DivergenceMetrics,
    Intervention,
    rollout_comparison,
)
from world_model_lens.causal.effect_estimator import CausalEffect, CausalEffectEstimator
from world_model_lens.causal.trajectory_attribution import AttributionResult, TrajectoryAttribution

__all__ = [
    "CausalEffectEstimator",
    "CausalEffect",
    "TrajectoryAttribution",
    "AttributionResult",
    "CounterfactualEngine",
    "BranchTree",
    "Intervention",
    "DivergenceMetrics",
    "rollout_comparison",
]
