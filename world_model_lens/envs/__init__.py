"""Environment interface for collecting episodes and interacting with RL ecosystems."""

from world_model_lens.envs.base import (
    EnvironmentAdapter,
    EnvironmentCapabilities,
    EnvSpec,
    EnvStepResult,
    normalize_observation,
    normalize_action,
)

from world_model_lens.envs.factory import (
    EnvironmentFactory,
    AdapterInfo,
    FACTORY,
    register,
    create,
)

from world_model_lens.envs.gymnasium_adapter import (
    GymnasiumAdapter,
    GymnasiumCapabilities,
    LegacyGymAdapter,
    make as gym_make,
)

from world_model_lens.envs.env_interface import EpisodeCollector

__all__ = [
    # Base classes
    "EnvironmentAdapter",
    "EnvironmentCapabilities",
    "EnvSpec",
    "EnvStepResult",
    # Factory
    "EnvironmentFactory",
    "AdapterInfo",
    "FACTORY",
    "register",
    "create",
    # Gymnasium
    "GymnasiumAdapter",
    "GymnasiumCapabilities",
    "LegacyGymAdapter",
    "gym_make",
    # Episode collector (legacy, uses Gymnasium directly)
    "EpisodeCollector",
    # Utilities
    "normalize_observation",
    "normalize_action",
]
