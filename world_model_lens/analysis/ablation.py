from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

"""Ablation engine for systematic analysis of world model components.

This module provides tools for:
- Component ablation (remove/zero/scramble specific components)
- Layer-by-layer ablation analysis
- Neuron-level ablation
- Causal ablation studies
"""


@dataclass
class AblationResult:
    """Result of ablation analysis."""

    original_metric: float
    ablated_metric: float
    impact: float
    component: str
    method: str
    details: dict[str, Any]


@dataclass
class AblationStudyResult:
    """Result of full ablation study."""

    results: list[AblationResult]
    most_important: list[tuple[str, float]]
    least_important: list[tuple[str, float]]
    summary: dict[str, Any]


class AblationEngine:
    """Systematic ablation engine for world model analysis."""

    def __init__(
        self,
        wm: Any,
        metric_fn: Callable[[Any], float],
    ):
        """Initialize ablation engine.

        Args:
            wm: HookedWorldModel instance
            metric_fn: Function to compute metric on trajectory
        """
        self.wm = wm
        self.metric_fn = metric_fn

    def ablate_component(
        self,
        cache: Any,
        component: str,
        method: str = "zero",
        value: float = 0.0,
    ) -> AblationResult:
        """Ablate a specific component in activation cache.

        Args:
            cache: ActivationCache
            component: Component name
            method: Ablation method ("zero", "mean", "scramble", "constant")
            value: Constant value for "constant" method

        Returns:
            AblationResult
        """
        original = cache[component].clone() if component in cache.component_names else None

        if original is None:
            return AblationResult(
                original_metric=0.0,
                ablated_metric=0.0,
                impact=0.0,
                component=component,
                method=method,
                details={"error": "component not found"},
            )

        original_metric = self.metric_fn(cache)

        ablated = self._apply_ablation(original, method, value)

        cache[component] = ablated
        ablated_metric = self.metric_fn(cache)

        cache[component] = original

        impact = original_metric - ablated_metric

        return AblationResult(
            original_metric=original_metric,
            ablated_metric=ablated_metric,
            impact=impact,
            component=component,
            method=method,
            details={
                "original_mean": original.mean().item(),
                "ablated_mean": ablated.mean().item(),
            },
        )

    def ablate_timestep(
        self,
        cache: Any,
        timestep: int,
        method: str = "zero",
    ) -> AblationResult:
        """Ablate all components at a specific timestep.

        Args:
            cache: ActivationCache
            timestep: Timestep to ablate
            method: Ablation method

        Returns:
            AblationResult
        """
        original_values = {}

        for comp in cache.component_names:
            key = (comp, timestep)
            if key in cache._cache or comp in cache._cache:
                original_values[comp] = cache[comp, timestep].clone()

        original_metric = self.metric_fn(cache)

        for comp in original_values:
            ablated = self._apply_ablation(original_values[comp], method)
            cache[comp, timestep] = ablated

        ablated_metric = self.metric_fn(cache)

        for comp, original in original_values.items():
            cache[comp, timestep] = original

        impact = original_metric - ablated_metric

        return AblationResult(
            original_metric=original_metric,
            ablated_metric=ablated_metric,
            impact=impact,
            component=f"timestep_{timestep}",
            method=method,
            details={"num_components": len(original_values)},
        )

    def ablate_dimensions(
        self,
        cache: Any,
        component: str,
        dimensions: list[int],
        method: str = "zero",
    ) -> AblationResult:
        """Ablate specific dimensions of a component.

        Args:
            cache: ActivationCache
            component: Component name
            dimensions: List of dimension indices
            method: Ablation method

        Returns:
            AblationResult
        """
        original = cache[component].clone()

        original_metric = self.metric_fn(cache)

        ablated = original.clone()
        for dim in dimensions:
            if dim < ablated.shape[-1]:
                ablated = self._apply_ablation(ablated, method, dim=dim)

        cache[component] = ablated
        ablated_metric = self.metric_fn(cache)

        cache[component] = original

        impact = original_metric - ablated_metric

        return AblationResult(
            original_metric=original_metric,
            ablated_metric=ablated_metric,
            impact=impact,
            component=f"{component}_dims_{dimensions}",
            method=method,
            details={"num_dims": len(dimensions)},
        )

    def layer_ablation_study(
        self,
        cache: Any,
        layers: list[str],
        method: str = "zero",
    ) -> AblationStudyResult:
        """Perform layer-by-layer ablation study.

        Args:
            cache: ActivationCache
            layers: List of layer names
            method: Ablation method

        Returns:
            AblationStudyResult
        """
        results = []

        for layer in layers:
            result = self.ablate_component(cache, layer, method)
            results.append(result)

        impacts = [(r.component, r.impact) for r in results]
        impacts.sort(key=lambda x: x[1], reverse=True)

        return AblationStudyResult(
            results=results,
            most_important=impacts[:3],
            least_important=impacts[-3:],
            summary={
                "total_layers": len(layers),
                "mean_impact": np.mean([r.impact for r in results]),
                "method": method,
            },
        )

    def sweep_ablation(
        self,
        cache: Any,
        component: str,
        num_ablations: int = 10,
        method: str = "random",
    ) -> list[AblationResult]:
        """Sweep through random ablations.

        Args:
            cache: ActivationCache
            component: Component to ablate
            num_ablations: Number of random ablations
            method: Method for selecting dimensions

        Returns:
            List of AblationResult
        """
        original = cache[component].clone()
        dim = original.shape[-1]

        results = []

        for i in range(num_ablations):
            if method == "random":
                num_dims = np.random.randint(1, dim // 4)
                dims = np.random.choice(dim, num_dims, replace=False).tolist()
            elif method == "progressive":
                num_dims = (i + 1) * dim // num_ablations
                dims = list(range(num_dims))
            else:
                dims = []

            result = self.ablate_dimensions(cache, component, dims)
            results.append(result)

        cache[component] = original

        return results

    def _apply_ablation(
        self,
        tensor: torch.Tensor,
        method: str,
        value: float = 0.0,
        dim: int | None = None,
    ) -> torch.Tensor:
        """Apply ablation to tensor.

        Args:
            tensor: Input tensor
            method: Ablation method
            value: Constant value
            dim: Specific dimension (for dimension ablation)

        Returns:
            Ablated tensor
        """
        if dim is not None:
            result = tensor.clone()
            if method == "zero":
                result[..., dim] = 0.0
            elif method == "mean":
                result[..., dim] = tensor[..., dim].mean()
            elif method == "scramble":
                result[..., dim] = tensor[..., dim][torch.randperm(len(tensor))]
            elif method == "constant":
                result[..., dim] = value
            return result

        if method == "zero":
            return torch.zeros_like(tensor)
        elif method == "mean":
            return torch.ones_like(tensor) * tensor.mean()
        elif method == "scramble":
            return tensor[torch.randperm(len(tensor))]
        elif method == "constant":
            return torch.ones_like(tensor) * value
        elif method == "gaussian":
            return torch.randn_like(tensor) * tensor.std() + tensor.mean()
        else:
            return tensor


class CausalAblationEngine:
    """Ablation engine for causal analysis."""

    def __init__(self, wm: Any):
        """Initialize causal ablation engine.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm

    def ablate_path(
        self,
        cache: Any,
        source: str,
        target: str,
        metric_fn: Callable[[Any], float],
    ) -> AblationResult:
        """Ablate causal path between components.

        Args:
            cache: ActivationCache
            source: Source component
            target: Target component
            metric_fn: Metric function

        Returns:
            AblationResult
        """
        original_source = cache[source].clone()
        original_target = cache[target].clone()

        original_metric = metric_fn(cache)

        cache[source] = torch.zeros_like(cache[source])
        ablated_metric = metric_fn(cache)

        cache[source] = original_source
        cache[target] = original_target

        impact = original_metric - ablated_metric

        return AblationResult(
            original_metric=original_metric,
            ablated_metric=ablated_metric,
            impact=impact,
            component=f"{source}->{target}",
            method="causal_ablate",
            details={"source": source, "target": target},
        )

    def find_bottleneck(
        self,
        cache: Any,
        components: list[str],
        metric_fn: Callable[[Any], float],
    ) -> str:
        """Find bottleneck component.

        Args:
            cache: ActivationCache
            components: List of components
            metric_fn: Metric function

        Returns:
            Name of bottleneck component
        """
        impacts = {}

        for comp in components:
            result = AblationEngine(self.wm, metric_fn).ablate_component(cache, comp, method="zero")
            impacts[comp] = result.impact

        bottleneck = max(impacts.items(), key=lambda x: x[1])
        return bottleneck[0]
