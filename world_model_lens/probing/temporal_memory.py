"""Temporal memory analyzer for sequential latent dynamics.

This module provides temporal analysis tools that work with ANY world model:
- Memory retention analysis
- Temporal dependencies
- Sequential pattern detection
- Working memory capacity estimation
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import torch
from dataclasses import dataclass

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache


@dataclass
class TemporalMemoryResult:
    """Result of temporal memory analysis."""

    retention_scores: torch.Tensor
    memory_capacity: float
    temporal_dependencies: Dict[str, float]
    working_memory_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_capacity": self.memory_capacity,
            "working_memory_estimate": self.working_memory_estimate,
            "temporal_dependencies": self.temporal_dependencies,
            "n_timesteps_analyzed": len(self.retention_scores),
        }


class TemporalMemoryProber:
    """Prober for temporal memory and sequential dynamics.

    Works with ANY world model - analyzes how well the model retains
    information over time without requiring RL-specific features.

    Analysis types:
    - memory_retention: How well past information is retained
    - temporal_dependencies: Time lag correlations
    - sequential_patterns: Recurring patterns in latent sequences
    - working_memory_capacity: Estimated memory buffer size
    """

    def __init__(self, wm: Any = None):
        """Initialize temporal memory prober.

        Args:
            wm: Optional HookedWorldModel for capability checking.
        """
        self.wm = wm
        self._caps = wm.capabilities if wm and hasattr(wm, "capabilities") else None

    @property
    def capabilities(self):
        """Access world model capabilities."""
        return self._caps

    def memory_retention(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
        max_lag: int = 20,
    ) -> TemporalMemoryResult:
        """Analyze memory retention over time lags.

        Works with ANY world model - measures how well latent representations
        retain information from past timesteps.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze.
            max_lag: Maximum time lag to analyze.

        Returns:
            TemporalMemoryResult with retention scores.
        """
        latents = []
        for t in sorted(cache.timesteps):
            try:
                z = cache[component, t]
                if z is not None:
                    latents.append(z.flatten())
            except (KeyError, TypeError):
                pass

        if len(latents) < 3:
            return TemporalMemoryResult(
                retention_scores=torch.tensor([]),
                memory_capacity=0.0,
                temporal_dependencies={},
                working_memory_estimate=0.0,
            )

        latents_tensor = torch.stack(latents)
        if latents_tensor.dim() > 2:
            latents_tensor = latents_tensor.flatten(1)

        retention_scores = []
        for lag in range(1, min(max_lag + 1, len(latents))):
            correlations = []
            for i in range(len(latents_tensor) - lag):
                v1 = latents_tensor[i]
                v2 = latents_tensor[i + lag]

                norm1 = torch.norm(v1)
                norm2 = torch.norm(v2)

                if norm1 > 1e-8 and norm2 > 1e-8:
                    corr = torch.dot(v1, v2) / (norm1 * norm2)
                    correlations.append(corr.item())

            if correlations:
                retention_scores.append(sum(correlations) / len(correlations))
            else:
                retention_scores.append(0.0)

        retention_tensor = torch.tensor(retention_scores)

        decay_rate = 0.0
        if len(retention_scores) >= 3:
            log_retention = torch.log(retention_tensor.clamp(min=1e-8)).numpy()
            time_indices = torch.arange(1, len(retention_scores) + 1, dtype=torch.float32).numpy()
            try:
                import numpy as np

                coeffs = np.polyfit(time_indices, log_retention, 1)
                decay_rate = -float(coeffs[0])
            except Exception:
                decay_rate = 0.0

        memory_capacity = 1.0 / decay_rate if decay_rate > 0 else float(len(latents))

        working_memory_estimate = memory_capacity * 0.5

        return TemporalMemoryResult(
            retention_scores=retention_tensor,
            memory_capacity=memory_capacity,
            temporal_dependencies={
                "short_term": float(retention_tensor[0].item())
                if len(retention_tensor) > 0
                else 0.0,
                "medium_term": float(retention_tensor[min(5, len(retention_tensor) - 1)].item())
                if len(retention_tensor) > 5
                else 0.0,
                "long_term": float(retention_tensor[-1].item())
                if len(retention_tensor) > 0
                else 0.0,
            },
            working_memory_estimate=working_memory_estimate,
        )

    def temporal_dependencies(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
        max_lag: int = 10,
    ) -> Dict[str, Any]:
        """Analyze temporal dependencies in latent sequences.

        Works with ANY world model - computes autocorrelation and
        cross-correlation at various time lags.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze.
            max_lag: Maximum time lag.

        Returns:
            Dict with autocorrelation and lag analysis.
        """
        latents = []
        for t in sorted(cache.timesteps):
            try:
                z = cache[component, t]
                if z is not None:
                    latents.append(z.flatten())
            except (KeyError, TypeError):
                pass

        if len(latents) < 3:
            return {
                "autocorrelations": [],
                "dominant_period": 0,
                "is_available": True,
            }

        latents_tensor = torch.stack(latents)
        if latents_tensor.dim() > 2:
            latents_tensor = latents_tensor.flatten(1)

        mean = latents_tensor.mean(dim=0)
        centered = latents_tensor - mean

        autocorrelations = []
        for lag in range(1, min(max_lag + 1, len(latents))):
            if lag >= len(centered):
                break

            c0 = torch.sum(centered[:-lag] * centered[:-lag]).item()
            ck = torch.sum(centered[:-lag] * centered[lag:]).item()

            if c0 > 1e-8:
                autocorrelations.append(ck / c0)
            else:
                autocorrelations.append(0.0)

        dominant_period = 0
        for i, ac in enumerate(autocorrelations):
            if i > 0 and ac > 0.5:
                for j in range(i + 1, len(autocorrelations)):
                    if autocorrelations[j] < 0:
                        dominant_period = j - i
                        break
                if dominant_period > 0:
                    break

        return {
            "autocorrelations": autocorrelations,
            "dominant_period": dominant_period,
            "is_available": True,
        }

    def sequential_patterns(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
        pattern_length: int = 3,
    ) -> Dict[str, Any]:
        """Detect recurring sequential patterns.

        Works with ANY world model - finds repeating motifs in
        latent trajectory sequences.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze.
            pattern_length: Length of patterns to search for.

        Returns:
            Dict with pattern analysis results.
        """
        latents = []
        for t in sorted(cache.timesteps):
            try:
                z = cache[component, t]
                if z is not None:
                    latents.append(z.flatten())
            except (KeyError, TypeError):
                pass

        if len(latents) < pattern_length * 2:
            return {
                "patterns": [],
                "pattern_frequencies": [],
                "is_available": True,
            }

        latents_tensor = torch.stack(latents)
        if latents_tensor.dim() > 2:
            latents_tensor = latents_tensor.flatten(1)

        latents_quantized = torch.zeros_like(latents_tensor)
        for i in range(len(latents_tensor)):
            normalized = (latents_tensor[i] - latents_tensor[i].mean()) / (
                latents_tensor[i].std() + 1e-8
            )
            latents_quantized[i] = (normalized > 0).float()

        pattern_counts = {}
        for i in range(len(latents_quantized) - pattern_length + 1):
            pattern = tuple(latents_quantized[i : i + pattern_length].flatten().long().tolist())
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
        top_patterns = sorted_patterns[:10]

        return {
            "patterns": [list(p[0]) for p in top_patterns],
            "pattern_frequencies": [
                p[1] / (len(latents_quantized) - pattern_length + 1) for p in top_patterns
            ],
            "is_available": True,
        }

    def full_analysis(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
    ) -> Dict[str, Any]:
        """Run complete temporal memory analysis.

        Works with ANY world model.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze.

        Returns:
            Dict with all temporal memory results.
        """
        results = {}

        mem_result = self.memory_retention(cache, component)
        results["memory_retention"] = mem_result.to_dict()

        dep_result = self.temporal_dependencies(cache, component)
        results["temporal_dependencies"] = dep_result

        pattern_result = self.sequential_patterns(cache, component)
        results["sequential_patterns"] = pattern_result

        return results
