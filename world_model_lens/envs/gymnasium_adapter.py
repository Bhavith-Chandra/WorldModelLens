"""Gymnasium environment adapter.

Provides adapter for Gymnasium environments with standardized
numpy array observations and actions.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    gym = None
    spaces = None
    print("[!WARNING] Gymnasium not installed. No Fallback in place, may cause errors in future. Please install it with: pip install gymnasium")

from world_model_lens.envs.base import (
    EnvironmentAdapter,
    EnvironmentCapabilities,
    EnvSpec,
    EnvStepResult,
    normalize_action,
    normalize_observation,
)
from world_model_lens.envs.factory import register


class GymnasiumCapabilities(EnvironmentCapabilities):
    """Capabilities of the Gymnasium adapter."""

    supports_vectorization: bool = True
    supports_render: bool = True
    supports_async: bool = True
    supports_seeded_reset: bool = True
    supports_max_episode_steps: bool = True
    is_image_based: bool = False


@register(
    prefix="gym",
    capabilities=GymnasiumCapabilities(
        supports_vectorization=True,
        supports_render=True,
        supports_async=True,
        supports_seeded_reset=True,
        supports_max_episode_steps=True,
    ),
    description="Gymnasium environments",
    set_default=True,
)
class GymnasiumAdapter(EnvironmentAdapter):
    """Adapter for Gymnasium environments.

    Wraps any Gymnasium-compatible environment and provides
    standardized numpy array interface.

    Example:
        adapter = GymnasiumAdapter("CartPole-v1")
        obs, info = adapter.reset()
        result = adapter.step(action)
        adapter.close()

        # With vectorized environments
        adapter = GymnasiumAdapter("CartPole-v1", n_envs=4, vector_mode="async")
        obs, info = adapter.reset()
        result = adapter.step(actions)  # 4 actions
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        vector_mode: str = "sync",
        render_mode: Optional[str] = None,
        **gym_kwargs,
    ):
        """Initialize Gymnasium adapter.

        Args:
            env_id: Gymnasium environment ID (e.g., "CartPole-v1").
            n_envs: Number of parallel environments (default 1).
            vector_mode: "sync" or "async" for vectorization.
            render_mode: Optional render mode ("human", "rgb_array", etc.).
            **gym_kwargs: Additional kwargs for gym.make().
        """
        if not HAS_GYM:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium")

        self._env_id = env_id
        self._n_envs = n_envs
        self._vector_mode = vector_mode
        self._render_mode = render_mode
        self._gym_kwargs = gym_kwargs

        super().__init__(env_id)

    def _create_spec(self) -> EnvSpec:
        """Create environment specification."""
        temp_env = self._create_env()

        obs_space = temp_env.observation_space
        action_space = temp_env.action_space

        obs_shape = self._get_space_shape(obs_space)
        action_shape = self._get_space_shape(action_space)

        obs_dtype = self._get_space_dtype(obs_space)
        action_dtype = self._get_space_dtype(action_space)

        action_is_discrete = isinstance(action_space, spaces.Discrete)

        max_episode_steps = getattr(temp_env, "_max_episode_steps", None)

        temp_env.reset()
        self._spec = EnvSpec(
            obs_shape=obs_shape,
            action_shape=action_shape,
            obs_dtype=obs_dtype,
            action_dtype=action_dtype,
            action_is_discrete=action_is_discrete,
            max_episode_steps=max_episode_steps,
        )
        return self._spec

    def _create_env(self) -> Any:
        """Create the Gymnasium environment."""
        if self._n_envs > 1:
            if self._vector_mode == "async":
                self._env = gym.vector.AsyncVectorEnv(
                    [self._make_env for _ in range(self._n_envs)],
                    render_mode=self._render_mode,
                )
            else:
                self._env = gym.vector.SyncVectorEnv(
                    [self._make_env for _ in range(self._n_envs)],
                    render_mode=self._render_mode,
                )
        else:
            self._env = gym.make(
                self._env_id,
                render_mode=self._render_mode,
                **self._gym_kwargs,
            )

        return self._env

    @property
    def _make_env(self):
        """Create a single env factory for vector env."""

        def make():
            return gym.make(
                self._env_id,
                render_mode=self._render_mode,
                **self._gym_kwargs,
            )

        return make

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional seed for reproducibility.

        Returns:
            Tuple of (observation, info_dict).
        """
        if self._env is None:
            self._create_env()

        if self._n_envs > 1:
            obs, info = self._env.reset(seed=seed)
        else:
            obs, info = self._env.reset(seed=seed)

        obs = normalize_observation(obs)
        return obs, info

    def step(self, action: np.ndarray) -> EnvStepResult:
        """Take a step in the environment.

        Args:
            action: Action as numpy array.

        Returns:
            EnvStepResult with observation, reward, done flags, and info.
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action = normalize_action(action, self.spec.action_is_discrete)

        if self._n_envs > 1:
            obs, reward, terminated, truncated, info = self._env.step(action)
            obs = normalize_observation(obs)
            return EnvStepResult(
                observation=obs,
                reward=float(reward)
                if not hasattr(reward, "__len__")
                else [float(r) for r in reward],
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        else:
            obs, reward, terminated, truncated, info = self._env.step(action)
            obs = normalize_observation(obs)
            return EnvStepResult(
                observation=obs,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

    def close(self) -> None:
        """Clean up environment resources."""
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def unwrapped(self) -> Any:
        """Get the underlying environment object."""
        return self._env

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", else None.
        """
        if self._env is None:
            return None
        return self._env.render()

    @property
    def action_space(self) -> Any:
        """Get the action space."""
        if self._env is None:
            self._create_env()
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """Get the observation space."""
        if self._env is None:
            self._create_env()
        return self._env.observation_space

    def _get_space_shape(self, space) -> Tuple[int, ...]:
        """Get shape from gymnasium space."""
        if isinstance(space, spaces.Box):
            return space.shape
        elif isinstance(space, spaces.Discrete):
            return (1,)
        elif isinstance(space, spaces.Tuple):
            return tuple(self._get_space_shape(s) for s in space.spaces)
        elif isinstance(space, spaces.Dict):
            return tuple(self._get_space_shape(s) for s in space.spaces.values())
        return ()

    def _get_space_dtype(self, space) -> np.dtype:
        """Get dtype from gymnasium space."""
        if isinstance(space, spaces.Box):
            return np.dtype(space.dtype)
        elif isinstance(space, spaces.Discrete):
            return np.dtype(np.int64)
        elif isinstance(space, spaces.Tuple):
            dtypes = [self._get_space_dtype(s) for s in space.spaces]
            return dtypes[0] if dtypes else np.dtype(np.float32)
        elif isinstance(space, spaces.Dict):
            dtypes = [self._get_space_dtype(s) for s in space.spaces.values()]
            return dtypes[0] if dtypes else np.dtype(np.float32)
        return np.dtype(np.float32)


class LegacyGymAdapter(GymnasiumAdapter):
    """Adapter for legacy OpenAI Gym (pre-Gymnasium).

    Supports older gym versions through shimmy compatibility.
    """

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        vector_mode: str = "sync",
        **kwargs,
    ):
        if not HAS_GYM:
            try:
                import gym

                self._use_gymnasium_shimmy = False
            except ImportError:
                raise ImportError(
                    "Either gymnasium or gym is required. Install gymnasium: pip install gymnasium"
                )

        super().__init__(env_id, n_envs, vector_mode, **kwargs)


def make(env_id: str, **kwargs) -> GymnasiumAdapter:
    """Convenience function to create a GymnasiumAdapter.

    Args:
        env_id: Gymnasium environment ID.
        **kwargs: Additional arguments for GymnasiumAdapter.

    Returns:
        GymnasiumAdapter instance.
    """
    return GymnasiumAdapter(env_id, **kwargs)
