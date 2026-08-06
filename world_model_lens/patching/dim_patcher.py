from __future__ import annotations
"""Dimension patcher for per-dimension analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import torch


@dataclass
class DimPatchingResult:
    """Result of dimension patching."""

    dimension: int
    original_metric: float
    patched_metric: float
    impact: float
    method: str


class DimensionPatcher:
    """Patch individual dimensions for analysis."""

    def __init__(self, wm: Any):
        """Initialize dimension patcher.

        Args:
            wm: HookedWorldModel instance
        """
        self.wm = wm

    def patch_dimension(
        self,
        cache: Any,
        component: str,
        dimension: int,
        method: str = "zero",
        metric_fn: Callable[[Any], float] | None = None,
    ) -> DimPatchingResult:
        """Patch a specific dimension.

        Args:
            cache: ActivationCache
            component: Component name
            dimension: Dimension index
            method: Patching method
            metric_fn: Optional metric function

        Returns:
            DimPatchingResult
        """
        original = cache[component].clone()

        if method == "zero":
            modified = original.clone()
            modified[..., dimension] = 0.0
        elif method == "mean":
            modified = original.clone()
            modified[..., dimension] = original[..., dimension].mean()
        else:
            modified = original

        if metric_fn:
            original_metric = metric_fn(cache)
            cache[component] = modified
            patched_metric = metric_fn(cache)
            cache[component] = original
        else:
            original_metric = 0.0
            patched_metric = 0.0

        return DimPatchingResult(
            dimension=dimension,
            original_metric=original_metric,
            patched_metric=patched_metric,
            impact=original_metric - patched_metric,
            method=method,
        )

    def sweep_dimensions(
        self,
        cache: Any,
        component: str,
        metric_fn: Callable[[Any], float],
    ) -> list[DimPatchingResult]:
        """Sweep all dimensions.

        Args:
            cache: ActivationCache
            component: Component name
            metric_fn: Metric function

        Returns:
            List of DimPatchingResult
        """
        original = cache[component]
        dims = original.shape[-1]

        results = []
        for dim in range(dims):
            result = self.patch_dimension(cache, component, dim, "zero", metric_fn)
            results.append(result)

        return results


__all__ = ["DimensionPatcher", "DimPatchingResult"]
