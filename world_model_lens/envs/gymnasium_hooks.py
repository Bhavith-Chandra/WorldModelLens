"""Gymnasium environment hooks for world model integration.

This module provides wrappers to use world models with standard RL environments:
- Gymnasium (formerly Gym)
- DeepMind Control Suite
- Procgen
- Meta-World

Usage:
    env = make_env("CartPole-v1")
    wm = HookedWorldModel(adapter, config)

    # Wrap environment for world model interaction
    wm_env = WorldModelEnv(wm, env)

    obs = wm_env.reset()
    for _ in range(100):
        action = wm_env.action_space.sample()  # Or use wm.imagine() for planning
        obs, reward, done, info = wm_env.step(action)
"""

from typing import Any, Dict, Optional, Tuple, Union, Callable, TYPE_CHECKING, List
import numpy as np
import torch
from dataclasses import dataclass

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces

        GYMNASIUM_AVAILABLE = True
    except ImportError:
        GYMNASIUM_AVAILABLE = False
        gym = None
        spaces = None
except ImportError:
    import gym
    from gym import spaces

    GYMNASIUM_AVAILABLE = True


@dataclass
class EnvConfig:
    """Configuration for world model environment wrapper."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    frame_stack: int = 1
    obs_transform: Optional[Callable] = None
    action_transform: Optional[Callable] = None
    normalize_observations: bool = False
    normalize_rewards: bool = False
    render_mode: Optional[str] = None


class WorldModelEnv:
    """Wrapper to use a world model with Gymnasium environments.

    This allows:
    - Running real environment interactions with world model caching
    - Planning via imagined rollouts
    - Model-based RL with the world model

    Example:
        env = gym.make("CartPole-v1")
        wm = HookedWorldModel(adapter, config)

        wm_env = WorldModelEnv(wm, env)
        obs = wm_env.reset()

        # Use imagination for planning
        imagined = wm.imagine(wm_env.current_state, horizon=10)

        # Take real step
        action = wm_env.action_space.sample()
        obs, reward, done, info = wm_env.step(action)
    """

    def __init__(
        self,
        world_model: Any,
        env: gym.Env,
        config: Optional[EnvConfig] = None,
    ):
        """Initialize world model environment wrapper.

        Args:
            world_model: HookedWorldModel instance
            env: Gymnasium environment
            config: Environment configuration
        """
        self.wm = world_model
        self.env = env
        self.config = config or EnvConfig()

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata

        self._device = torch.device(self.config.device)
        self._current_obs = None
        self._current_state = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._trajectory_cache = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional reset options

        Returns:
            Observation and info dict
        """
        obs, info = self.env.reset(seed=seed, options=options)

        obs = self._transform_obs(obs)
        self._current_obs = obs

        if self.config.normalize_observations:
            obs = self._normalize_obs(obs)

        obs_tensor = torch.from_numpy(obs).float().to(self._device)
        if obs_tensor.dim() == 3:  # Image
            obs_tensor = obs_tensor.permute(2, 0, 1) / 255.0
        elif obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        h, z = self.wm.adapter.initial_state(batch_size=1, device=self._device)

        self._current_state = (h, z)

        self._episode_return = 0.0
        self._episode_length = 0
        self._trajectory_cache = []

        if getattr(self.wm, "config", {}).get("store_logits", True):
            from world_model_lens.core.activation_cache import ActivationCache

            self._cache = ActivationCache()
        else:
            self._cache = None

        return obs, info

    def step(
        self,
        action: Union[int, np.ndarray],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Observation, reward, terminated, truncated, info
        """
        if isinstance(action, int):
            action = np.array(action)

        action = self._transform_action(action)

        if self.config.action_transform:
            action = self.config.action_transform(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._transform_obs(obs)

        if self.config.normalize_observations:
            obs = self._normalize_obs(obs)
        if self.config.normalize_rewards:
            reward = self._normalize_reward(reward)

        self._current_obs = obs

        obs_tensor = torch.from_numpy(obs).float().to(self._device)
        if obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.permute(2, 0, 1) / 255.0
        elif obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        h, z = self._current_state

        if self._cache is not None:
            try:
                traj, cache = self.wm.run_with_cache(
                    obs_tensor.unsqueeze(0),
                    actions=torch.tensor([action]).to(self._device).unsqueeze(0)
                    if action.shape
                    else None,
                )
                self._cache = cache
            except Exception:
                pass

        posterior, obs_encoding = self.wm.adapter.encode(
            obs_tensor.unsqueeze(0), h if h.dim() > 1 else h.unsqueeze(0)
        )

        prior = self.wm.adapter.dynamics(h if h.dim() > 1 else h.unsqueeze(0))

        z = posterior.squeeze(0) if posterior.dim() > 1 else posterior
        h_new = self.wm.adapter.transition(
            h if h.dim() > 1 else h.unsqueeze(0),
            z.unsqueeze(0) if z.dim() == 1 else z,
            torch.tensor(action).to(self._device).unsqueeze(0).unsqueeze(0)
            if action.shape
            else None,
        )

        self._current_state = (h_new.squeeze(0), z)

        self._episode_return += reward
        self._episode_length += 1

        info["world_model_state"] = self._current_state
        info["episode_return"] = self._episode_return
        info["episode_length"] = self._episode_length

        if self._cache is not None:
            info["activation_cache"] = self._cache

        return obs, reward, terminated, truncated, info

    def imagine(
        self,
        horizon: int = 10,
        actions: Optional[np.ndarray] = None,
    ) -> Any:
        """Run imagined rollouts from current state.

        Args:
            horizon: Number of imagination steps
            actions: Optional actions to execute

        Returns:
            Imagined WorldTrajectory
        """
        h, z = self._current_state

        if actions is None:
            actions = (
                np.random.randn(horizon, *self.action_space.shape)
                if hasattr(self.action_space, "shape")
                else None
            )

        if actions is not None:
            actions = torch.from_numpy(actions).float().to(self._device)

        trajectory = self.wm.imagine(
            start_state=None,
            actions=actions,
            horizon=horizon,
        )

        return trajectory

    def plan(
        self,
        planner_fn: Callable,
        horizon: int = 10,
        n_candidates: int = 10,
        **planner_kwargs,
    ) -> np.ndarray:
        """Plan using the world model.

        Args:
            planner_fn: Function that takes imagined trajectory and returns score
            horizon: Planning horizon
            n_candidates: Number of action sequences to evaluate
            **planner_kwargs: Additional arguments for planner

        Returns:
            Best action sequence
        """
        h, z = self._current_state

        best_score = float("-inf")
        best_actions = None

        for _ in range(n_candidates):
            actions = (
                np.random.randn(horizon, *self.action_space.shape)
                if hasattr(self.action_space, "shape")
                else None
            )

            if actions is not None:
                actions_tensor = torch.from_numpy(actions).float().to(self._device)

                trajectory = self.wm.imagine(
                    start_state=None,
                    actions=actions_tensor,
                    horizon=horizon,
                )

                score = planner_fn(trajectory, **planner_kwargs)

                if score > best_score:
                    best_score = score
                    best_actions = actions

        return best_actions[0] if best_actions is not None else None

    def _transform_obs(self, obs: np.ndarray) -> np.ndarray:
        """Transform observation."""
        if self.config.obs_transform:
            return self.config.obs_transform(obs)
        return obs

    def _transform_action(self, action: Union[int, np.ndarray]) -> np.ndarray:
        """Transform action."""
        if self.config.action_transform:
            return self.config.action_transform(action)
        if isinstance(action, int):
            return np.array([action])
        return action

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        if not hasattr(self, "_obs_mean"):
            self._obs_mean = np.zeros_like(obs)
            self._obs_var = np.ones_like(obs)

        self._obs_mean = 0.99 * self._obs_mean + 0.01 * obs
        self._obs_var = 0.99 * self._obs_var + 0.01 * (obs**2)

        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward."""
        if not hasattr(self, "_reward_mean"):
            self._reward_mean = 0.0
            self._reward_var = 1.0

        self._reward_mean = 0.99 * self._reward_mean + 0.01 * reward
        self._reward_var = 0.99 * self._reward_var + 0.01 * (reward**2)

        return (reward - self._reward_mean) / (np.sqrt(self._reward_var) + 1e-8)

    @property
    def current_state(self) -> Any:
        """Get current world model state."""
        return self._current_state

    @property
    def activation_cache(self) -> Any:
        """Get current activation cache."""
        return self._cache

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


def make_world_model_env(
    world_model: Any,
    env_id: str,
    config: Optional[EnvConfig] = None,
    **env_kwargs,
) -> WorldModelEnv:
    """Create a world model environment.

    Args:
        world_model: HookedWorldModel instance
        env_id: Gymnasium environment ID
        config: Environment configuration
        **env_kwargs: Additional arguments for gym.make

    Returns:
        WorldModelEnv wrapper
    """
    env = gym.make(env_id, **env_kwargs)
    return WorldModelEnv(world_model, env, config)


class BatchWorldModelEnv:
    """Vectorized world model environments."""

    def __init__(
        self,
        world_model: Any,
        env_fns: list,
        config: Optional[EnvConfig] = None,
    ):
        """Initialize batched environments.

        Args:
            world_model: HookedWorldModel instance
            env_fns: List of functions that create environments
            config: Environment configuration
        """
        self.wm = world_model
        self.envs = [fn() for fn in env_fns]
        self.config = config or EnvConfig()

        self._num_envs = len(self.envs)

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        obs_list = []
        info_list = []

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        return np.array(obs_list), info_list

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        obs_list = []
        reward_list = []
        term_list = []
        trunc_list = []
        info_list = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)

        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(term_list),
            np.array(trunc_list),
            info_list,
        )

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
