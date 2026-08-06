"""Episode collector for Gymnasium environments."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import gymnasium as gym

    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    gym = None

import torch

from world_model_lens import HookedWorldModel
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.activation_cache import ActivationCache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world_model_lens.core.lazy_trajectory import TrajectoryDataset


class EpisodeCollector:
    """Collects episodes from a Gymnasium environment using a world model.

    Example:
        collector = EpisodeCollector(env, wm, device=torch.device('cuda'))
        traj, cache = collector.collect_episode(max_steps=1000)
        dataset = collector.collect_dataset(n_episodes=100)
    """

    def __init__(
        self,
        env: Any,
        wm: HookedWorldModel,
        device: Optional[torch.device] = None,
    ):
        if not HAS_GYM:
            raise ImportError(
                "gymnasium is required for EpisodeCollector. Install with: pip install gymnasium"
            )

        self.env = env
        self.wm = wm
        self.device = device or torch.device("cpu")

    def collect_episode(
        self,
        policy: Optional[Callable] = None,
        max_steps: int = 1000,
        seed: Optional[int] = None,
        render_fn: Optional[Callable] = None,
    ) -> Tuple[LatentTrajectory, ActivationCache, Dict[str, Any]]:
        """Collect a single episode.

        Args:
            policy: Optional policy taking (obs, h, z) returning action.
                    If None, uses random actions.
            max_steps: Maximum episode length.
            seed: Optional environment seed.
            render_fn: Optional function to render observations.

        Returns:
            Tuple of (LatentTrajectory, ActivationCache, info_dict).
        """
        obs, info = self.env.reset(seed=seed)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        h, z = self.wm.adapter.initial_state(batch_size=1)
        h = h.squeeze(0)
        z_flat = z.squeeze(0).flatten()

        states = []
        obs_list = [obs]
        action_list = []
        reward_list = []
        done = False
        t = 0

        while not done and t < max_steps:
            if policy is not None:
                action = policy(obs, h, z_flat)
            else:
                action = torch.from_numpy(self.env.action_space.sample()).float()

            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()

            obs_next, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            obs_next_t = (
                torch.from_numpy(obs_next).float() if isinstance(obs_next, np.ndarray) else obs_next
            )

            z_posterior_logits, obs_encoding = self.wm.adapter.encode(
                obs.unsqueeze(0).to(self.device), h.unsqueeze(0).to(self.device)
            )
            z_posterior_logits = z_posterior_logits.squeeze(0)
            obs_encoding = obs_encoding.squeeze(0)

            z_posterior = self.wm.adapter.sample_z(
                z_posterior_logits, temperature=1.0, sample=False
            )
            z_prior_logits = self.wm.adapter.dynamics(h.unsqueeze(0).to(self.device)).squeeze(0)
            z_prior = self.wm.adapter.sample_z(z_prior_logits, temperature=1.0, sample=False)

            reward_pred = self.wm.adapter.predict_reward(
                h.unsqueeze(0).to(self.device), z_posterior.flatten().unsqueeze(0).to(self.device)
            ).squeeze(0)

            from world_model_lens.core import LatentState

            state = LatentState(
                h_t=h.clone(),
                z_posterior=z_posterior.clone(),
                z_prior=z_prior.clone(),
                timestep=t,
                action=action.clone(),
                reward_pred=reward_pred.detach().clone(),
                reward_real=torch.tensor(reward, dtype=torch.float32),
                obs_encoding=obs_encoding.detach().clone(),
            )
            states.append(state)
            obs_list.append(obs_next_t)
            action_list.append(action)
            reward_list.append(reward)

            h = self.wm.adapter.transition(
                h.unsqueeze(0).to(self.device),
                z_posterior.flatten().unsqueeze(0).to(self.device),
                action.unsqueeze(0).to(self.device),
            ).squeeze(0)

            obs = obs_next_t
            t += 1

        obs_seq = torch.stack(obs_list, dim=0).to(self.device)
        action_seq = torch.stack(action_list, dim=0).to(self.device)

        traj = LatentTrajectory(
            states=states,
            imagined=False,
            metadata={"env_info": info, "total_reward": sum(reward_list)},
        )

        _, traj, cache = self.wm.run_with_cache(obs_seq, action_seq)

        return traj, cache, {"total_reward": sum(reward_list), "length": t}

    def collect_dataset(
        self,
        n_episodes: int,
        policy: Optional[Callable] = None,
        seed: Optional[int] = None,
        max_steps: int = 1000,
    ) -> "TrajectoryDataset":
        """Collect multiple episodes into a dataset.

        Args:
            n_episodes: Number of episodes to collect.
            policy: Optional policy function.
            seed: Base seed (episodes get seed, seed+1, ...).
            max_steps: Max steps per episode.

        Returns:
            TrajectoryDataset containing all episodes.
        """
        from world_model_lens.hub import TrajectoryDataset

        trajectories = []
        for i in range(n_episodes):
            ep_seed = (seed + i) if seed is not None else None
            traj, _, _ = self.collect_episode(
                policy=policy,
                max_steps=max_steps,
                seed=ep_seed,
            )
            traj.episode_id = i
            trajectories.append(traj)

        return TrajectoryDataset(trajectories)
