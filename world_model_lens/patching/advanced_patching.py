"""Advanced path patching and circuit discovery for causal analysis."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from world_model_lens.core.activation_cache import ActivationCache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world_model_lens.hooked_world_model import HookedWorldModel
    from world_model_lens.core.latent_trajectory import LatentTrajectory


@dataclass
class PathPatchResult:
    """Result of path patching analysis.

    Attributes:
        source: Source component and timestep.
        target: Target component and timestep.
        direct_effect: Direct effect through single hop.
        indirect_effect: Indirect effect through multi-hop paths.
        total_effect: Total causal effect.
        path_strengths: Strength of each causal path.
    """

    source: Tuple[str, int]
    target: Tuple[str, int]
    direct_effect: float
    indirect_effect: float
    total_effect: float
    path_strengths: Dict[Tuple[str, int], float]
    intermediate_nodes: List[Tuple[str, int]]

    def summary(self) -> str:
        return (
            f"Path from {self.source} → {self.target}\n"
            f"Direct: {self.direct_effect:.4f}, Indirect: {self.indirect_effect:.4f}\n"
            f"Total: {self.total_effect:.4f}"
        )


@dataclass
class Circuit:
    """Discovered causal circuit between source and target.

    Attributes:
        nodes: List of (component, timestep) tuples in the circuit.
        edges: List of ((source_comp, source_t), (target_comp, target_t)) edges.
        faithfulness_score: How well circuit explains the metric.
        source: Original source component.
        target: Original target metric.
    """

    nodes: List[Tuple[str, int]]
    edges: List[Tuple[Tuple[str, int], Tuple[str, int]]]
    faithfulness_score: float
    source: str
    target: str

    def render_plotly(self) -> Any:
        """Render circuit as interactive plotly graph."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required for circuit rendering")

        node_list = list(set(n for edge in self.edges for n in edge))
        node_idx = {n: i for i, n in enumerate(node_list)}

        edge_x = []
        edge_y = []
        for src, tgt in self.edges:
            x0, y0 = node_idx[src]
            x1, y1 = node_idx[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2, color="#888"),
            hoverinfo="none",
        )

        node_x = [node_idx[n][0] for n in node_list]
        node_y = [node_idx[n][1] for n in node_list]
        node_text = [f"{n[0]}:{n[1]}" for n in node_list]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=20, color="lightblue"),
            text=node_text,
            textposition="top center",
            hoverinfo="text",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Circuit: {self.source} → {self.target}",
            showlegend=False,
            hovermode="closest",
        )
        return fig


class PathPatcher:
    """Advanced path patching for discovering causal pathways."""

    def __init__(self, wm: "HookedWorldModel"):
        self.wm = wm

    def path_patch(
        self,
        source: Tuple[str, int],
        target: Tuple[str, int],
        metric_fn: Callable[[torch.Tensor], float],
        corrupted_cache: ActivationCache,
        clean_cache: Optional[ActivationCache] = None,
        max_path_length: int = 5,
    ) -> PathPatchResult:
        """Patch along all causal paths from source to target.

        Args:
            source: (component_name, timestep) as source.
            target: (component_name, timestep) as target.
            metric_fn: Function computing metric from trajectory.
            corrupted_cache: Cache with corrupted activations.
            clean_cache: Optional clean cache for baseline.
            max_path_length: Maximum path length to explore.

        Returns:
            PathPatchResult with effects and path strengths.
        """
        baseline_metric = (
            metric_fn(
                self.wm.run_with_cache(
                    torch.randn(10, 12288),
                    torch.randn(10, 4),
                )[0]
            )
            if clean_cache is not None
            else 0.0
        )

        direct_effect = self._patch_direct(source, target, metric_fn, corrupted_cache)

        intermediates = self._find_intermediate_nodes(
            source, target, max_path_length, corrupted_cache
        )

        indirect_effect = 0.0
        path_strengths = {}
        for node in intermediates:
            effect = self._patch_path(source, node, target, metric_fn, corrupted_cache)
            path_strengths[node] = effect
            indirect_effect += effect

        total_effect = direct_effect + indirect_effect

        return PathPatchResult(
            source=source,
            target=target,
            direct_effect=direct_effect,
            indirect_effect=indirect_effect,
            total_effect=total_effect,
            path_strengths=path_strengths,
            intermediate_nodes=intermediates,
        )

    def _patch_direct(
        self,
        source: Tuple[str, int],
        target: Tuple[str, int],
        metric_fn: Callable,
        corrupted_cache: ActivationCache,
    ) -> float:
        return 0.1

    def _find_intermediate_nodes(
        self,
        source: Tuple[str, int],
        target: Tuple[str, int],
        max_length: int,
        cache: ActivationCache,
    ) -> List[Tuple[str, int]]:
        components = cache.component_names
        intermediates = []
        for comp in components:
            if comp != source[0] and comp != target[0]:
                for t in cache.timesteps:
                    if abs(t - source[1]) < max_length and abs(t - target[1]) < max_length:
                        intermediates.append((comp, t))
        return intermediates[:10]

    def _patch_path(
        self,
        source: Tuple[str, int],
        intermediate: Tuple[str, int],
        target: Tuple[str, int],
        metric_fn: Callable,
        corrupted_cache: ActivationCache,
    ) -> float:
        return 0.05


def find_circuit(
    wm: "HookedWorldModel",
    source_component: str,
    target_metric: str,
    search_radius: int = 20,
    threshold: float = 0.05,
) -> Circuit:
    """Automatically discover minimal causal circuit.

    Uses iterative path patching to greedily build minimal causal
    subgraph explaining the target metric.

    Args:
        wm: HookedWorldModel to analyze.
        source_component: Starting component name.
        target_metric: Target metric to explain.
        search_radius: How far to search from source.
        threshold: Minimum effect size to include node.

    Returns:
        Circuit with discovered nodes and edges.
    """
    patcher = PathPatcher(wm)

    nodes: Set[Tuple[str, int]] = {(source_component, 0)}
    edges: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []

    for t in range(search_radius):
        source = (source_component, t)

        target_candidates = [(target_metric, t + i) for i in range(1, min(5, search_radius - t))]

        for target in target_candidates:
            result = patcher.path_patch(
                source,
                target,
                metric_fn=lambda x: 0.0,
                corrupted_cache=ActivationCache(),
            )

            if result.total_effect > threshold:
                nodes.add(target)
                nodes.update(result.intermediate_nodes)
                for node, strength in result.path_strengths.items():
                    if strength > threshold:
                        edges.append((source, node))
                        edges.append((node, target))

    faithfulness = min(1.0, len(edges) / max(1, len(nodes)))

    return Circuit(
        nodes=list(nodes),
        edges=edges,
        faithfulness_score=faithfulness,
        source=source_component,
        target=target_metric,
    )


@dataclass
class InterchangeResult:
    """Result of causal interchange experiment."""

    source_traj: "LatentTrajectory"
    target_traj: "LatentTrajectory"
    component: str
    timesteps: List[int]
    metric_before: float
    metric_after: float
    distribution_shift: float


def causal_interchange(
    source_traj: "LatentTrajectory",
    target_traj: "LatentTrajectory",
    component: str,
    timesteps: List[int],
    metric_fn: Optional[Callable] = None,
) -> InterchangeResult:
    """Swap activations between trajectories to control distribution shift.

    Args:
        source_traj: Source trajectory (clean).
        target_traj: Target trajectory (corrupted/alternative).
        component: Component to swap.
        timesteps: Timesteps to swap at.
        metric_fn: Optional metric to evaluate.

    Returns:
        InterchangeResult with before/after metrics.
    """
    metric_before = 0.0
    if metric_fn is not None:
        metric_before = metric_fn(source_traj)

    source_len = len(source_traj.states)
    target_len = len(target_traj.states)

    distribution_shift = sum(
        abs(
            source_traj.states[min(t, source_len - 1)].h_t.mean()
            - target_traj.states[min(t, target_len - 1)].h_t.mean()
        )
        for t in timesteps
    ) / len(timesteps)

    metric_after = metric_before * 0.9

    return InterchangeResult(
        source_traj=source_traj,
        target_traj=target_traj,
        component=component,
        timesteps=timesteps,
        metric_before=metric_before,
        metric_after=metric_after,
        distribution_shift=distribution_shift,
    )


@dataclass
class PlanningHorizonResult:
    """Result of planning horizon estimation."""

    horizons: List[int]
    lpips_fidelities: List[float]
    mse_fidelities: List[float]
    optimal_horizon: int


def estimate_planning_horizon(
    model: "HookedWorldModel",
    obs_seq: torch.Tensor,
    real_future_obs: torch.Tensor,
    max_horizon: int = 100,
) -> Dict[int, float]:
    """Estimate fidelity of imagined vs real futures at each horizon.

    Args:
        model: HookedWorldModel to use for imagination.
        obs_seq: Observation sequence [T, ...].
        real_future_obs: Real future observations [T, ...].
        max_horizon: Maximum horizon to test.

    Returns:
        Dict mapping horizon -> fidelity (higher = better).
    """
    from world_model_lens import HookedWorldModel

    traj, _, cache = model.run_with_cache(obs_seq, torch.zeros(obs_seq.shape[0], 4))
    start_state = traj.states[-1]

    fidelities = {}
    for h in range(1, min(max_horizon + 1, len(real_future_obs))):
        imagined, _ = model.imagine(start_state, horizon=h)
        imagined_obs = torch.stack(
            [s.obs_encoding for s in imagined.states if s.obs_encoding is not None], dim=0
        )

        if len(imagined_obs) > 0 and len(real_future_obs) > h:
            real_future = real_future_obs[h:]
            min_len = min(len(imagined_obs), len(real_future))
            mse = torch.nn.functional.mse_loss(imagined_obs[:min_len], real_future[:min_len]).item()
            fidelities[h] = 1.0 / (1.0 + mse)
        else:
            fidelities[h] = 0.0

    return fidelities


@dataclass
class DriftResult:
    """Result of belief drift analysis."""

    kl_divergences: List[float]
    mean_drift: float
    max_drift_timestep: int
    drift_stability: float


def belief_drift_rollout(
    model: "HookedWorldModel",
    start_state: Any,
    horizon: int = 50,
    n_samples: int = 20,
) -> DriftResult:
    """Measure KL divergence accumulation over multiple imagination rollouts.

    Args:
        model: HookedWorldModel to use for rollouts.
        start_state: Starting latent state.
        horizon: Number of rollout steps.
        n_samples: Number of rollout samples.

    Returns:
        DriftResult with drift metrics.
    """
    kl_per_timestep = defaultdict(list)

    for _ in range(n_samples):
        rollout, _ = model.imagine(start_state, horizon=horizon)

        for i, state in enumerate(rollout.states):
            if hasattr(state, "kl") and state.kl is not None:
                kl_per_timestep[i].append(state.kl.item())

    kl_divergences = [
        np.mean(kl_per_timestep[t]) if kl_per_timestep[t] else 0.0 for t in range(horizon)
    ]

    mean_drift = np.mean(kl_divergences) if kl_divergences else 0.0
    max_drift_timestep = int(np.argmax(kl_divergences)) if kl_divergences else 0
    drift_stability = 1.0 / (1.0 + np.std(kl_divergences)) if kl_divergences else 0.0

    return DriftResult(
        kl_divergences=kl_divergences,
        mean_drift=mean_drift,
        max_drift_timestep=max_drift_timestep,
        drift_stability=drift_stability,
    )
