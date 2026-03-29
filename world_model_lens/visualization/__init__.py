"""Visualization Module for World Model Interpretability.

Provides visualization tools to make world model internals human-readable.

Key Components:
- LatentTrajectoryPlotter: PCA/t-SNE of latent states over time
- PredictionVisualizer: Compare predictions to ground truth
- InterventionVisualizer: Show effects of patches
- TemporalAttributionMap: Show timestep-to-timestep influence
"""

from world_model_lens.visualization.latent_plots import LatentTrajectoryPlotter, SurpriseHeatmap
from world_model_lens.visualization.prediction_plots import PredictionVisualizer
from world_model_lens.visualization.intervention_plots import InterventionVisualizer
from world_model_lens.visualization.temporal_maps import TemporalAttributionMap
from world_model_lens.visualization.cache_plots import CacheSignalPlotter

__all__ = [
    "LatentTrajectoryPlotter",
    "PredictionVisualizer",
    "InterventionVisualizer",
    "TemporalAttributionMap",
    "SurpriseHeatmap",
    "CacheSignalPlotter",
]
