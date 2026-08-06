"""Core abstractions for backend-agnostic world model interpretability.

This module provides generic abstractions that work with ANY world model:
- WorldState: Generic latent state representation
- WorldTrajectory: Generic trajectory container
- WorldDynamics: Dynamics prediction outputs
- WorldModelConfig: Minimal configuration
- BaseModelAdapter: Abstract interface for any world model

Supported architectures:
- Ha & Schmidhuber World Models (VAE + MDN-RNN)
- PlaNet (latent RSSM from pixels)
- Dreamer family: V1, V2, V3
- TD-MPC / TD-MPC2 (JEPA-style)
- Contrastive/Predictive models (CWM, SPR)
- IRIS-style transformer world models
- Decision/Trajectory Transformers
- Video world models (WorldDreamer-style)
- Autonomous driving world models
- Robotics/embodied world models

RL-specific components (reward, value, action, done) are OPTIONAL.
Non-RL world models (video prediction, unsupervised) are first-class citizens.
"""

from world_model_lens.core.world_state import (
    WorldState,
    WorldDynamics,
    WorldModelOutput,
    ObservationType,
)
from world_model_lens.core.world_trajectory import (
    WorldTrajectory,
    TrajectoryStatistics,
)
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookPoint, HookContext, HookRegistry
from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.cache_query import CacheQuery
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.latent_state import LatentState
from world_model_lens.core.lazy_trajectory import (
    LatentTrajectoryLite,
    TensorStore,
    TrajectoryDataset,
)
from world_model_lens.core.types import (
    LatentType,
    DynamicsType,
    ModelPurpose,
    WorldModelFamily,
    ObservationModality,
)

__all__ = [
    "WorldState",
    "WorldDynamics",
    "WorldModelOutput",
    "WorldTrajectory",
    "TrajectoryStatistics",
    "WorldModelConfig",
    "ObservationType",
    "HookPoint",
    "HookContext",
    "HookRegistry",
    "ActivationCache",
    "LatentTrajectory",
    "LatentState",
    "LatentTrajectoryLite",
    "TensorStore",
    "TrajectoryDataset",
    "LatentType",
    "DynamicsType",
    "ModelPurpose",
    "WorldModelFamily",
    "ObservationModality",
    "CacheQuery",
]
