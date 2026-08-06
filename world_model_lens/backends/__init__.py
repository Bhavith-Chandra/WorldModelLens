"""Backends for world model architectures.

This package contains adapters for various world model types:

RL World Models:
- DreamerV3Adapter, DreamerV2Adapter: Dreamer model family
- IRISAdapter: Iterative reinforcement learning with imagined states
- TDMPC2Adapter: TD-MPC2 model-based RL

Non-RL World Models:
- ToyVideoAdapter: Video prediction world model
- ToyScientificAdapter: Scientific latent dynamics model

Other:
- VideoWorldModelAdapter: Video prediction models
- PlanningAdapter: Planning-based world models
"""

from world_model_lens.backends.base_adapter import BaseModelAdapter
from world_model_lens.backends.registry import BackendRegistry, REGISTRY
from world_model_lens.core.types import WorldModelFamily

from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.backends.dreamerv2 import DreamerV2Adapter
from world_model_lens.backends.iris import IRISAdapter
from world_model_lens.backends.tdmpc2 import TDMPC2Adapter
from world_model_lens.backends.video_adapter import VideoWorldModelAdapter
from world_model_lens.backends.planning_adapter import PlanningAdapter
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.backends.toy_video_model import ToyVideoAdapter, create_toy_video_adapter
from world_model_lens.backends.toy_scientific_model import (
    ToyScientificAdapter,
    create_toy_scientific_adapter,
)

BACKEND_REGISTRY = REGISTRY

REGISTRY.register(
    name="toy_video",
    cls=ToyVideoAdapter,
    family=WorldModelFamily.VIDEO_WORLD_MODEL,
    description="Toy video prediction world model",
    supports_rl=False,
    supports_video=True,
    supports_planning=False,
)

REGISTRY.register(
    name="toy_scientific",
    cls=ToyScientificAdapter,
    family=WorldModelFamily.CUSTOM,
    description="Toy scientific latent dynamics model",
    supports_rl=False,
    supports_video=False,
    supports_planning=True,
)

__all__ = [
    "BaseModelAdapter",
    "DreamerV3Adapter",
    "DreamerV2Adapter",
    "IRISAdapter",
    "TDMPC2Adapter",
    "VideoWorldModelAdapter",
    "PlanningAdapter",
    "IJEPAAdapter",
    "ToyVideoAdapter",
    "ToyScientificAdapter",
    "create_toy_video_adapter",
    "create_toy_scientific_adapter",
    "BackendRegistry",
    "REGISTRY",
    "BACKEND_REGISTRY",
]
