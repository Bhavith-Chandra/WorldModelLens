"""Sweep result for patching sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch
import numpy as np


@dataclass
class SweepResult:
    """Result of a patching sweep."""

    component: str
    results: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)

    def add_result(
        self,
        target: Any,
        original_value: float,
        patched_value: float,
    ) -> None:
        """Add a result to the sweep.

        Args:
            target: Target (timestep, dimension, etc.)
            original_value: Original metric value
            patched_value: Patched metric value
        """
        impact = original_value - patched_value
        self.results.append(
            {
                "target": target,
                "original": original_value,
                "patched": patched_value,
                "impact": impact,
                "impact_pct": (impact / (original_value + 1e-8)) * 100,
            }
        )

    def get_top_important(self, n: int = 10) -> list[dict[str, Any]]:
        """Get top N most important targets.

        Args:
            n: Number to return

        Returns:
            List of top results
        """
        sorted_results = sorted(self.results, key=lambda x: abs(x["impact"]), reverse=True)
        return sorted_results[:n]

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics.

        Returns:
            Summary dict
        """
        if not self.results:
            return {}

        impacts = [r["impact"] for r in self.results]

        self.summary = {
            "component": self.component,
            "num_targets": len(self.results),
            "mean_impact": np.mean(impacts),
            "std_impact": np.std(impacts),
            "max_impact": np.max(impacts),
            "min_impact": np.min(impacts),
            "total_impact": np.sum(impacts),
        }

        return self.summary


@dataclass
class LayerSweepResult:
    """Result of layer-by-layer sweep."""

    layer_results: dict[str, SweepResult] = field(default_factory=dict)

    def add_layer_result(self, layer: str, result: SweepResult) -> None:
        """Add result for a layer."""
        self.layer_results[layer] = result

    def get_layer_importance(self) -> list[tuple[str, float]]:
        """Get importance ranking by layer.

        Returns:
            List of (layer, importance) sorted by importance
        """
        importances = []
        for layer, result in self.layer_results.items():
            summary = result.get_summary()
            importances.append((layer, summary.get("mean_impact", 0.0)))

        return sorted(importances, key=lambda x: x[1], reverse=True)


__all__ = ["SweepResult", "LayerSweepResult"]
