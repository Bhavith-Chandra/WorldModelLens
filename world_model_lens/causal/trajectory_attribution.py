"""Trajectory-Level Attribution for World Models.

Answers the question:
    "Which latent at timestep t affects outcome at timestep T?"

Not just "which neuron matters" but "which neuron at timestep 7
affects the outcome at timestep 20".

This is the temporal counterpart to standard feature attribution.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import numpy as np


@dataclass
class AttributionResult:
    """Result of trajectory-level attribution.

    Attributes:
        source_timestep: Where the attribution originates
        target_timestep: Where the effect is measured
        attribution_scores: Scores for each latent dimension [d_z]
        top_dims: Most important dimensions
        top_scores: Scores for top dimensions
        total_effect: Sum of attribution scores
        effect_propagation: How effect propagates through time
    """

    source_timestep: int
    target_timestep: int
    attribution_scores: torch.Tensor
    top_dims: List[int] = field(default_factory=list)
    top_scores: List[float] = field(default_factory=list)
    total_effect: float = 0.0
    effect_propagation: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalCausalGraph:
    """Graph of causal relationships across timesteps.

    Nodes: (timestep, latent_dim) pairs
    Edges: Causal influence from source to target timestep
    """

    nodes: List[Tuple[int, int]] = field(default_factory=list)
    edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = field(default_factory=dict)
    edge_weights: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = field(default_factory=dict)

    def add_edge(
        self,
        source: Tuple[int, int],
        target: Tuple[int, int],
        weight: float,
    ) -> None:
        """Add causal edge from source to target."""
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)

        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(target)

        self.edge_weights[(source, target)] = weight


class TrajectoryAttribution:
    """Trajectory-level causal attribution.

    Provides methods to attribute outcomes to specific latent-timestep pairs.

    Example:
        attr = TrajectoryAttribution(world_model)

        # Which latent at t=7 affects final reward?
        result = attr.attribute(
            source_timestep=7,
            target_timestep=-1,  # final
            target_metric="reward",
        )

        print(f"Top dims: {result.top_dims}")

        # Full causal graph
        graph = attr.build_causal_graph(
            trajectory_length=20,
            target_metric="final_reward",
        )
    """

    def __init__(self, world_model: Any):
        """Initialize trajectory attributor.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def attribute(
        self,
        source_timestep: int,
        target_timestep: int = -1,
        target_metric: str = "reward_pred",
        intervention_type: str = "ablation",
    ) -> AttributionResult:
        """Attribute target to source timestep latent.

        Args:
            source_timestep: Where the latent originates
            target_timestep: Where to measure effect (-1 for final)
            target_metric: What to predict ("reward_pred", "reconstruction_error")
            intervention_type: How to intervene

        Returns:
            AttributionResult with per-dimension scores
        """
        # Get baseline trajectory
        observations = torch.randn(20, 3, 64, 64)
        baseline_traj, _, baseline_cache = self.wm.run_with_cache(observations)

        # Get latent dimension
        z_sample = baseline_cache.get(("z_posterior", source_timestep))
        if z_sample is None:
            z_sample = baseline_traj.states[source_timestep].state

        d_z = z_sample.shape[-1]
        attribution_scores = torch.zeros(d_z)

        # Measure effect of ablating each dimension
        for dim in range(min(d_z, 32)):  # Limit for efficiency
            effect = self._measure_dimension_effect(
                source_timestep,
                dim,
                target_timestep,
                target_metric,
                baseline_traj,
                intervention_type,
            )
            attribution_scores[dim] = effect

        # Get top dimensions
        topk = min(10, d_z)
        scores, indices = torch.topk(attribution_scores.abs(), topk)

        return AttributionResult(
            source_timestep=source_timestep,
            target_timestep=target_timestep,
            attribution_scores=attribution_scores,
            top_dims=indices.tolist(),
            top_scores=scores.tolist(),
            total_effect=attribution_scores.sum().item(),
        )

    def _measure_dimension_effect(
        self,
        source_t: int,
        dim: int,
        target_t: int,
        metric: str,
        baseline_traj: Any,
        intervention_type: str,
    ) -> float:
        """Measure causal effect of single dimension ablation."""

        # Run with ablation
        def ablate_hook(tensor, ctx):
            if ctx.timestep != source_t:
                return tensor
            result = tensor.clone()
            result[..., dim] = 0
            return result

        observations = torch.randn(20, 3, 64, 64)
        intervened_traj, _ = self.wm.run_with_advanced_hooks(
            observations,
            hook_specs={f"t={source_t}.z": ablate_hook},
        )

        # Extract target metric
        baseline_outcome = self._extract_metric(baseline_traj, target_t, metric)
        intervened_outcome = self._extract_metric(intervened_traj, target_t, metric)

        return abs(baseline_outcome - intervened_outcome)

    def _extract_metric(
        self,
        trajectory: Any,
        timestep: int,
        metric: str,
    ) -> float:
        """Extract metric from trajectory."""
        if timestep < 0:
            timestep = len(trajectory.states) + timestep

        if timestep >= len(trajectory.states):
            timestep = len(trajectory.states) - 1

        state = trajectory.states[timestep]

        if metric == "reward_pred":
            return state.predictions.get("reward", torch.tensor(0.0)).item()
        elif metric == "state_norm":
            return state.state.norm().item()
        elif metric == "obs_encoding_norm":
            if state.obs_encoding is not None:
                return state.obs_encoding.norm().item()
        return 0.0

    def build_causal_graph(
        self,
        trajectory_length: int,
        target_metric: str = "final_reward",
        max_source_timestep: int = -5,
    ) -> TemporalCausalGraph:
        """Build full causal graph across timesteps.

        Args:
            trajectory_length: Length of trajectory
            target_metric: Metric to attribute to
            max_source_timestep: Latest source timestep to consider

        Returns:
            TemporalCausalGraph with edges weighted by causal effect
        """
        graph = TemporalCausalGraph()

        # Get observations
        observations = torch.randn(trajectory_length, 3, 64, 64)
        baseline_traj, _, baseline_cache = self.wm.run_with_cache(observations)

        # For each source timestep
        for source_t in range(trajectory_length):
            if max_source_timestep < 0 and source_t < trajectory_length + max_source_timestep:
                continue

            # Get attribution for this source
            result = self.attribute(
                source_timestep=source_t,
                target_timestep=-1,
                target_metric=target_metric,
            )

            # Add edges for top dimensions
            for dim, score in zip(result.top_dims[:5], result.top_scores[:5]):
                graph.add_edge(
                    (source_t, dim),
                    (-1, dim),  # Target is final timestep
                    score,
                )

        return graph

    def compute_effect_propagation(
        self,
        source_timestep: int,
        target_timesteps: List[int],
    ) -> Dict[int, float]:
        """Compute how effect propagates through time.

        Args:
            source_timestep: Where effect originates
            target_timesteps: Which timesteps to measure effect at

        Returns:
            Dict mapping timestep to effect magnitude
        """
        observations = torch.randn(20, 3, 64, 64)
        baseline_traj, _, _ = self.wm.run_with_cache(observations)

        propagation = {}

        for target_t in target_timesteps:
            # Ablate all dimensions at source
            def full_ablate(tensor, ctx):
                if ctx.timestep == source_timestep:
                    return tensor * 0
                return tensor

            intervened_traj, _ = self.wm.run_with_advanced_hooks(
                observations,
                hook_specs={f"t={source_timestep}.z": full_ablate},
            )

            baseline_outcome = self._extract_metric(baseline_traj, target_t, "state_norm")
            intervened_outcome = self._extract_metric(intervened_traj, target_t, "state_norm")

            propagation[target_t] = abs(baseline_outcome - intervened_outcome)

        return propagation

    def latent_interaction_matrix(
        self,
        timesteps: List[int],
        metric: str = "reward_pred",
    ) -> torch.Tensor:
        """Compute interaction matrix between latent dimensions across timesteps.

        Entry (i, j, t) shows how much latent i at timestep t
        interacts with latent j in determining the outcome.

        Args:
            timesteps: Which timesteps to analyze
            metric: Target metric

        Returns:
            Tensor of shape [len(timesteps), d_z, d_z]
        """
        observations = torch.randn(20, 3, 64, 64)
        baseline_traj, _, _ = self.wm.run_with_cache(observations)

        z_sample = baseline_traj.states[0].state
        d_z = z_sample.shape[-1]
        n_t = len(timesteps)

        interaction_matrix = torch.zeros(n_t, d_z, d_z)

        for ti, t in enumerate(timesteps):
            for dim_i in range(min(d_z, 10)):
                for dim_j in range(dim_i + 1, min(d_z, 10)):
                    # Measure second-order effect
                    effect = self._measure_pair_effect(t, dim_i, dim_j, metric)
                    interaction_matrix[ti, dim_i, dim_j] = effect
                    interaction_matrix[ti, dim_j, dim_i] = effect

        return interaction_matrix

    def _measure_pair_effect(
        self,
        timestep: int,
        dim1: int,
        dim2: int,
        metric: str,
    ) -> float:
        """Measure second-order effect of dimension pair."""
        observations = torch.randn(20, 3, 64, 64)

        def pair_ablate(tensor, ctx):
            if ctx.timestep != timestep:
                return tensor
            result = tensor.clone()
            result[..., dim1] = 0
            result[..., dim2] = 0
            return result

        traj, _ = self.wm.run_with_advanced_hooks(
            observations,
            hook_specs={f"t={timestep}.z": pair_ablate},
        )

        baseline_traj, _, _ = self.wm.run_with_cache(observations)

        baseline_outcome = self._extract_metric(baseline_traj, -1, metric)
        intervened_outcome = self._extract_metric(traj, -1, metric)

        return abs(baseline_outcome - intervened_outcome)
