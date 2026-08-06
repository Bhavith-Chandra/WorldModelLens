"""Abstract base class for environment adapters.

Provides a unified interface for interacting with different
reinforcement learning environments (Gymnasium, ProcGen, Isaac Lab, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class EnvSpec:
    """Specification of an environment's observation and action spaces."""

    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    obs_dtype: np.dtype
    action_dtype: np.dtype
    action_is_discrete: bool
    max_episode_steps: Optional[int] = None


@dataclass
class EnvStepResult:
    """Result of a single environment step."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class EnvironmentAdapter(ABC):
    """Abstract base class for environment adapters.

    Provides a unified interface for interacting with different
    reinforcement learning environments. All observations and actions
    are standardized to numpy arrays.

    Example:
        adapter = GymnasiumAdapter("CartPole-v1")
        obs, info = adapter.reset()
        result = adapter.step(action)
        adapter.close()
    """

    def __init__(self, env_config: Union[str, Dict[str, Any]]):
        """Initialize the environment adapter.

        Args:
            env_config: Either an env ID string or a config dict.
        """
        self._env = None
        self._spec: Optional[EnvSpec] = None
        self._env_config = env_config

    @property
    def spec(self) -> EnvSpec:
        """Get the environment specification."""
        if self._spec is None:
            self._spec = self._create_spec()
        return self._spec

    @abstractmethod
    def _create_spec(self) -> EnvSpec:
        """Create the environment specification.

        Returns:
            EnvSpec with observation and action space details.
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional seed for reproducibility.

        Returns:
            Tuple of (observation, info_dict).
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> EnvStepResult:
        """Take a step in the environment.

        Args:
            action: Action as numpy array.

        Returns:
            EnvStepResult with observation, reward, done flags, and info.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass

    @property
    @abstractmethod
    def unwrapped(self) -> Any:
        """Get the underlying environment object."""
        pass


class EnvironmentCapabilities:
    """Describes capabilities of an environment adapter."""

    def __init__(
        self,
        supports_vectorization: bool = False,
        supports_render: bool = True,
        supports_async: bool = False,
        supports_seeded_reset: bool = True,
        supports_max_episode_steps: bool = True,
        is_image_based: bool = False,
        is_unity_based: bool = False,
    ):
        self.supports_vectorization = supports_vectorization
        self.supports_render = supports_render
        self.supports_async = supports_async
        self.supports_seeded_reset = supports_seeded_reset
        self.supports_max_episode_steps = supports_max_episode_steps
        self.is_image_based = is_image_based
        self.is_unity_based = is_unity_based


def normalize_observation(obs: Any) -> np.ndarray:
    """Convert observation to numpy array with standardized shape.

    Args:
        obs: Raw observation from environment.

    Returns:
        Numpy array with standardized shape.
    """
    if isinstance(obs, np.ndarray):
        return obs
    if hasattr(obs, "numpy"):
        return obs.numpy()
    if hasattr(obs, "astype"):
        return obs.astype(np.float32)
    return np.array(obs, dtype=np.float32)


def normalize_action(action: Any, is_discrete: bool = False) -> np.ndarray:
    """Convert action to numpy array.

    Args:
        action: Raw action from environment.
        is_discrete: Whether action space is discrete.

    Returns:
        Numpy array action.
    """
    if isinstance(action, np.ndarray):
        return action
    if hasattr(action, "numpy"):
        return action.numpy()
    if hasattr(action, "astype"):
        dtype = np.int64 if is_discrete else np.float32
        return action.astype(dtype)
    return np.array(action)
