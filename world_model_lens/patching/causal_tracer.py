"""Causal tracer for finding causal paths between components in world models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.hooked_world_model import HookedWorldModel
from world_model_lens.patching.patcher import TemporalPatcher, PatchResult


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CausalEdge:
    """Represents a causal edge in the world model."""

    source: str
    target: str
    strength: float
    timestep: int
    direction: str = "forward"

    def __repr__(self) -> str:
        return f"CausalEdge({self.source} -> {self.target}, strength={self.strength:.3f}, t={self.timestep})"


@dataclass
class CausalPath:
    """A causal path from source to target."""

    nodes: List[str]
    edges: List[CausalEdge]
    total_strength: float

    def __repr__(self) -> str:
        return f"CausalPath({' -> '.join(self.nodes)}, strength={self.total_strength:.3f})"


@dataclass
class AttributionResult:
    """Result of causal attribution analysis."""

    component_scores: Dict[str, float]
    total_effect: float
    direct_effects: Dict[str, float]
    indirect_effects: Dict[str, float]
    paths: List[CausalPath]

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k components by attribution score."""
        sorted_scores = sorted(self.component_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_scores": self.component_scores,
            "total_effect": self.total_effect,
            "direct_effects": self.direct_effects,
            "indirect_effects": self.indirect_effects,
            "n_paths": len(self.paths),
        }


class CausalTracer:
    """Causal tracing for world models.

    Identifies causal paths and computes attribution scores for components.
    """

    def __init__(self, wm: HookedWorldModel, device: Optional[torch.device] = None):
        """Initialize the causal tracer.

        Args:
            wm: The hooked world model.
            device: Optional device override.
        """
        self.wm = wm
        self.device = device or _get_device()
        self.patcher = TemporalPatcher(wm, self.device)

    def _ensure_device(self, tensor: Tensor) -> Tensor:
        """Move tensor to device if needed."""
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def get_component_order(self, cache: ActivationCache) -> List[str]:
        """Get ordered list of components from cache.

        Args:
            cache: Activation cache.

        Returns:
            Ordered list of component names.
        """
        priority_order = ["z_prior", "z_posterior", "h", "state", "reward", "value", "observation"]
        components = list(cache.component_names)
        sorted_components = []
        for comp in priority_order:
            if comp in components:
                sorted_components.append(comp)
                components.remove(comp)
        sorted_components.extend(components)
        return sorted_components

    def compute_single_attribution(
        self,
        cache: ActivationCache,
        source_component: str,
        target_component: str,
        metric_fn: Callable[[ActivationCache], float],
    ) -> float:
        """Compute direct attribution from source to target.

        Args:
            cache: Activation cache.
            source_component: Source component name.
            target_component: Target component name.
            metric_fn: Metric function.

        Returns:
            Attribution score.
        """
        clean_metric = metric_fn(cache)

        corrupted_cache = ActivationCache()
        for comp in cache.component_names:
            for t in cache.timesteps:
                val = cache.get(comp, t, None)
                if val is not None:
                    original = val.to(self.device)
                    noise = torch.randn_like(original) * 0.5
                    corrupted_cache[comp, t] = original + noise

        corrupted_metric = metric_fn(corrupted_cache)

        for t in cache.timesteps:
            val = cache.get(source_component, t, None)
            if val is not None:
                corrupted_cache[source_component, t] = val.to(self.device)

        source_restored_metric = metric_fn(corrupted_cache)

        effect = (source_restored_metric - corrupted_metric) / (
            clean_metric - corrupted_metric + 1e-8
        )
        return max(0.0, min(1.0, effect))

    def trace_path(
        self,
        source: str,
        target: str,
        cache: ActivationCache,
        metric_fn: Callable[[ActivationCache], float],
        max_depth: int = 5,
    ) -> List[CausalPath]:
        """Find causal paths from source to target.

        Args:
            source: Source component name.
            target: target component name.
            cache: Activation cache.
            metric_fn: Metric function.
            max_depth: Maximum path depth.

        Returns:
            List of causal paths found.
        """
        paths: List[CausalPath] = []
        components = self.get_component_order(cache)

        if source not in components or target not in components:
            return paths

        source_idx = components.index(source)
        target_idx = components.index(target)

        if source_idx >= target_idx:
            return paths

        intermediate = components[source_idx + 1 : target_idx]

        direct_strength = self.compute_single_attribution(cache, source, target, metric_fn)

        if direct_strength > 0.01:
            edge = CausalEdge(source, target, direct_strength, 0)
            paths.append(
                CausalPath(
                    nodes=[source, target],
                    edges=[edge],
                    total_strength=direct_strength,
                )
            )

        for inter in intermediate:
            inter_strength = self.compute_single_attribution(cache, source, inter, metric_fn)
            inter_to_target = self.compute_single_attribution(cache, inter, target, metric_fn)

            if inter_strength > 0.01 and inter_to_target > 0.01:
                path_strength = inter_strength * inter_to_target
                edge1 = CausalEdge(source, inter, inter_strength, 0)
                edge2 = CausalEdge(inter, target, inter_to_target, 0)
                paths.append(
                    CausalPath(
                        nodes=[source, inter, target],
                        edges=[edge1, edge2],
                        total_strength=path_strength,
                    )
                )

        paths.sort(key=lambda p: p.total_strength, reverse=True)
        return paths

    def compute_attribution(
        self,
        cache: ActivationCache,
        target_metric_fn: Callable[[ActivationCache], float],
    ) -> AttributionResult:
        """Compute attribution scores for all components.

        Args:
            cache: Activation cache.
            target_metric_fn: Function computing target metric from cache.

        Returns:
            AttributionResult with scores for all components.
        """
        components = list(cache.component_names)
        component_scores: Dict[str, float] = {}
        direct_effects: Dict[str, float] = {}
        indirect_effects: Dict[str, float] = {}

        clean_metric = target_metric_fn(cache)

        corrupted_cache = ActivationCache()
        for comp in components:
            for t in cache.timesteps:
                val = cache.get(comp, t, None)
                if val is not None:
                    original = val.to(self.device)
                    noise = torch.randn_like(original) * 0.5
                    corrupted_cache[comp, t] = original + noise

        corrupted_metric = target_metric_fn(corrupted_cache)

        if abs(clean_metric - corrupted_metric) < 1e-8:
            for comp in components:
                component_scores[comp] = 0.0
                direct_effects[comp] = 0.0
                indirect_effects[comp] = 0.0
            return AttributionResult(
                component_scores=component_scores,
                total_effect=0.0,
                direct_effects=direct_effects,
                indirect_effects=indirect_effects,
                paths=[],
            )

        iterator = tqdm(components, desc="Computing attribution") if tqdm else components

        for comp in iterator:
            test_cache = ActivationCache()
            for c in components:
                for t in cache.timesteps:
                    val = cache.get(c, t, None)
                    if val is not None:
                        if c == comp:
                            test_cache[c, t] = val.to(self.device)
                        else:
                            corrupted_val = corrupted_cache.get(c, t, None)
                            if corrupted_val is not None:
                                test_cache[c, t] = corrupted_val

            restored_metric = target_metric_fn(test_cache)
            effect = (restored_metric - corrupted_metric) / (clean_metric - corrupted_metric)
            effect = max(0.0, min(1.0, effect))

            component_scores[comp] = effect
            direct_effects[comp] = effect

        indirect_effects = {comp: component_scores[comp] * 0.1 for comp in components}

        total_effect = sum(component_scores.values())

        paths: List[CausalPath] = []
        for i, source in enumerate(components):
            for j, target in enumerate(components):
                if i < j:
                    found_paths = self.trace_path(
                        source, target, cache, target_metric_fn, max_depth=3
                    )
                    paths.extend(found_paths)

        return AttributionResult(
            component_scores=component_scores,
            total_effect=total_effect,
            direct_effects=direct_effects,
            indirect_effects=indirect_effects,
            paths=paths,
        )

    def rank_nodes_by_importance(
        self,
        cache: ActivationCache,
        metric_fn: Callable[[ActivationCache], float],
    ) -> List[Tuple[str, float]]:
        """Rank nodes by causal importance.

        Args:
            cache: Activation cache.
            metric_fn: Metric function.

        Returns:
            List of (component, importance_score) tuples.
        """
        result = self.compute_attribution(cache, metric_fn)
        return result.top_k(k=len(result.component_scores))

    def find_bottleneck(
        self,
        cache: ActivationCache,
        metric_fn: Callable[[ActivationCache], float],
    ) -> List[Tuple[str, float, float]]:
        """Find bottleneck components that most affect output.

        Args:
            cache: Activation cache.
            metric_fn: Metric function.

        Returns:
            List of (component, importance, blockage) tuples.
        """
        components = list(cache.component_names)
        importances = []
        blockages = []

        clean_metric = metric_fn(cache)

        corrupted_cache = ActivationCache()
        for comp in components:
            for t in cache.timesteps:
                val = cache.get(comp, t, None)
                if val is not None:
                    original = val.to(self.device)
                    noise = torch.randn_like(original) * 0.5
                    corrupted_cache[comp, t] = original + noise

        corrupted_metric = metric_fn(corrupted_cache)

        for comp in components:
            importance = self.compute_single_attribution(cache, comp, "z_posterior", metric_fn)

            fully_corrupted = ActivationCache()
            for c in components:
                for t in cache.timesteps:
                    corrupted_val = corrupted_cache.get(c, t, None)
                    if corrupted_val is not None:
                        fully_corrupted[c, t] = corrupted_val

            for t in cache.timesteps:
                val = cache.get(comp, t, None)
                if val is not None:
                    fully_corrupted[comp, t] = val.to(self.device)

            restored_metric = metric_fn(fully_corrupted)
            blockage = (restored_metric - corrupted_metric) / (
                clean_metric - corrupted_metric + 1e-8
            )
            blockage = max(0.0, min(1.0, blockage))

            importances.append((comp, importance, blockage))

        importances.sort(key=lambda x: x[1], reverse=True)
        return importances


class PathPatcher:
    """Advanced path patching for causal analysis."""

    def __init__(self, wm: HookedWorldModel, device: Optional[torch.device] = None):
        """Initialize path patcher.

        Args:
            wm: The hooked world model.
            device: Optional device override.
        """
        self.wm = wm
        self.device = device or _get_device()
        self.tracer = CausalTracer(wm, device)

    def path_patch(
        self,
        cache: ActivationCache,
        source: str,
        target: str,
        intermediate: Optional[List[str]] = None,
        metric_fn: Optional[Callable[[ActivationCache], float]] = None,
    ) -> Dict[str, float]:
        """Patch along causal path from source to target.

        Args:
            cache: Activation cache.
            source: Source component.
            target: Target component.
            intermediate: Optional intermediate nodes.
            metric_fn: Optional metric function.

        Returns:
            Dictionary with direct_effect, indirect_effect, total_effect.
        """
        if metric_fn is None:
            metric_fn = lambda c: float(c["z_posterior", 0].abs().sum().item())

        direct_effect = self.tracer.compute_single_attribution(cache, source, target, metric_fn)

        indirect_effect = 0.0
        if intermediate:
            for inter in intermediate:
                inter_eff = self.tracer.compute_single_attribution(cache, source, inter, metric_fn)
                inter_to_target = self.tracer.compute_single_attribution(
                    cache, inter, target, metric_fn
                )
                indirect_effect += inter_eff * inter_to_target

        return {
            "direct_effect": direct_effect,
            "indirect_effect": indirect_effect,
            "total_effect": direct_effect + indirect_effect,
        }


def default_causal_metric(cache: ActivationCache) -> float:
    """Default metric: sum of z_posterior L2 norm.

    Args:
        cache: Activation cache.

    Returns:
        Metric value.
    """
    if "z_posterior" in cache.component_names and 0 in cache.timesteps:
        z = cache["z_posterior", 0]
        return float(z.abs().sum().item())
    return 0.0


def reward_based_metric(cache: ActivationCache) -> float:
    """Metric based on reward prediction.

    Args:
        cache: Activation cache.

    Returns:
        Reward prediction value.
    """
    if "reward" in cache.component_names and 0 in cache.timesteps:
        return float(cache["reward", 0].sum().item())
    return 0.0


def surprise_based_metric(cache: ActivationCache) -> float:
    """Metric based on surprise (KL divergence).

    Args:
        cache: Activation cache.

    Returns:
        Sum of KL divergences.
    """
    if "kl" in cache.component_names and 0 in cache.timesteps:
        return float(cache["kl", 0].sum().item())
    return 0.0
