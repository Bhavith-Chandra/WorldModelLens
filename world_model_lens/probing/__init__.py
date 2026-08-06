"""Probing tools for latent representations.

This package provides analysis tools that work with both RL and non-RL world models:

- LatentProber: Train probes on latent representations
- GeometryAnalyzer: Analyze latent space geometry and structure
- TemporalMemoryProber: Analyze temporal dynamics and memory retention
- CrossModalProber: Project latents into CLIP space and query concepts in plain English
- CrossModalProjector: Learnable linear projection from latent → CLIP embedding space
"""

from __future__ import annotations

try:
    from world_model_lens.probing.prober import LatentProber
except ImportError:
    LatentProber = None

from world_model_lens.probing.geometry import GeometryAnalyzer
from world_model_lens.probing.temporal_memory import TemporalMemoryProber
from world_model_lens.probing.crossmodal import (
    CrossModalProber,
    CrossModalProjector,
    CrossModalResult,
    ConceptQueryResult,
    align_multimodal,
)

__all__ = [
    "LatentProber",
    "GeometryAnalyzer",
    "TemporalMemoryProber",
    "CrossModalProber",
    "CrossModalProjector",
    "CrossModalResult",
    "ConceptQueryResult",
    "align_multimodal",
]
