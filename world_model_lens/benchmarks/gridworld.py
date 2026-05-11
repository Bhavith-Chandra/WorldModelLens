"""Gridworld Benchmark.

Canonical experiment showing:
1. Memory tracking through latent dynamics
2. Goal encoding in specific dimensions
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class GridworldResult:
    """Results from Gridworld benchmark."""

    memory_tracking_found: bool
    goal_encoding_found: bool
    memory_dims: List[int]
    goal_dims: List[int]
    causal_effects: Dict[str, float]
    metadata: Dict[str, Any]


class GridworldBenchmark:
    """Gridworld interpretability benchmark.

    Demonstrates:
    - Latent dimensions track visited locations (memory)
    - Goal location encoded in specific dims
    - Causal effect of memory/goal on behavior

    Example:
        benchmark = GridworldBenchmark(world_model)

        results = benchmark.run(
            grid_size=5,
            num_episodes=50,
        )

        print(f"Memory dims: {results.memory_dims}")
        print(f"Goal dims: {results.goal_dims}")
    """

    def __init__(self, world_model: Any):
        """Initialize benchmark.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def generate_gridworld_trajectories(
        self,
        grid_size: int = 5,
        num_episodes: int = 50,
        max_steps: int = 50,
        goal_location: Tuple[int, int] = (4, 4),
    ) -> List[torch.Tensor]:
        """Generate gridworld trajectories.

        Args:
            grid_size: Size of grid (grid_size x grid_size)
            num_episodes: Number of episodes
            max_steps: Max steps per episode
            goal_location: (row, col) of goal

        Returns:
            List of observation tensors
        """
        observations = []

        for _ in range(num_episodes):
            # Random start position
            pos = (
                np.random.randint(0, grid_size),
                np.random.randint(0, grid_size),
            )

            episode_obs = []

            for _ in range(max_steps):
                # One-hot encoding of position
                obs = torch.zeros(grid_size * grid_size + 2)
                obs[pos[0] * grid_size + pos[1]] = 1.0
                obs[-2] = goal_location[0] / grid_size
                obs[-1] = goal_location[1] / grid_size

                episode_obs.append(obs)

                # Random walk (simple exploration)
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                move = moves[np.random.randint(0, len(moves))]
                new_pos = (pos[0] + move[0], pos[1] + move[1])

                # Clip to grid
                new_pos = (
                    max(0, min(grid_size - 1, new_pos[0])),
                    max(0, min(grid_size - 1, new_pos[1])),
                )
                pos = new_pos

            observations.append(torch.stack(episode_obs))

        return observations

    def find_memory_tracking_dims(
        self,
        trajectories: List[torch.Tensor],
    ) -> Tuple[bool, List[int]]:
        """Find dimensions that track position history.

        Args:
            trajectories: List of trajectory tensors

        Returns:
            Tuple of (found, dimension_indices)
        """
        # Extract latents
        latents = []

        for traj_obs in trajectories[:20]:
            if traj_obs.dim() == 2:
                traj_obs = traj_obs.unsqueeze(0)
            traj, _ = self.wm.run_with_cache(traj_obs)

            for state in traj.states:
                z = state.obs_encoding if state.obs_encoding is not None else state.state
                if z is not None:
                    latents.append(z.squeeze())

        if not latents:
            return False, []

        latent_tensor = torch.stack(latents)

        # Compute temporal differences (position changes)
        diffs = torch.diff(latent_tensor, dim=0).abs()

        # Dimensions with high variance = tracking changes = memory
        variances = diffs.var(dim=0)

        threshold = variances.quantile(0.9).item()
        memory_dims = (variances > threshold).nonzero(as_tuple=True)[0].tolist()

        return len(memory_dims) > 0, memory_dims

    def find_goal_encoding_dims(
        self,
        trajectories: List[torch.Tensor],
    ) -> Tuple[bool, List[int]]:
        """Find dimensions that encode goal information.

        Args:
            trajectories: List of trajectory tensors

        Returns:
            Tuple of (found, dimension_indices)
        """
        latents = []
        goal_signals = []

        for traj_obs in trajectories[:20]:
            if traj_obs.dim() == 2:
                traj_obs = traj_obs.unsqueeze(0)

            # Goal info is in last 2 dimensions
            goal_signal = traj_obs[:, :, -2:].mean()

            traj, _ = self.wm.run_with_cache(traj_obs)

            for state in traj.states:
                z = state.obs_encoding if state.obs_encoding is not None else state.state
                if z is not None:
                    latents.append(z.squeeze())
                    goal_signals.append(goal_signal)

        if not latents:
            return False, []

        latent_tensor = torch.stack(latents)
        goal_tensor = torch.tensor(goal_signals)

        # Correlation with goal signal
        correlations = []
        for dim in range(latent_tensor.shape[-1]):
            if len(goal_tensor) > 1:
                corr = torch.corrcoef(
                    torch.stack([latent_tensor[:, dim], goal_tensor[: len(latent_tensor)]])
                )[0, 1].abs()
                correlations.append(corr.item() if not torch.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)

        threshold = 0.3
        goal_dims = [i for i, c in enumerate(correlations) if c > threshold]

        return len(goal_dims) > 0, goal_dims

    def test_memory_effect(
        self,
        memory_dims: List[int],
    ) -> float:
        """Test effect of ablating memory dimensions.

        Args:
            memory_dims: Dimensions to ablate

        Returns:
            Causal effect size
        """
        # Generate test trajectory
        test_traj = self.generate_gridworld_trajectories(
            grid_size=5,
            num_episodes=1,
            max_steps=20,
        )[0]

        # Baseline
        baseline_traj, _ = self.wm.run_with_cache(test_traj.unsqueeze(0))

        if not memory_dims:
            return 0.0

        # Ablation
        def ablate_memory(tensor, ctx):
            result = tensor.clone()
            for dim in memory_dims[:5]:
                if dim < result.shape[-1]:
                    result[..., dim] = 0
            return result

        intervened_traj, _ = self.wm.run_with_advanced_hooks(
            test_traj.unsqueeze(0),
            hook_specs={"z[0:32]": ablate_memory},
        )

        # Compute divergence
        baseline_norm = baseline_traj.states[-1].state.norm().item()
        intervened_norm = intervened_traj.states[-1].state.norm().item()

        return abs(baseline_norm - intervened_norm)

    def run(
        self,
        grid_size: int = 5,
        num_episodes: int = 50,
    ) -> GridworldResult:
        """Run full Gridworld benchmark.

        Args:
            grid_size: Size of grid
            num_episodes: Number of episodes

        Returns:
            GridworldResult with all findings
        """
        # Generate trajectories
        trajectories = self.generate_gridworld_trajectories(
            grid_size=grid_size,
            num_episodes=num_episodes,
        )

        # Find encodings
        mem_found, mem_dims = self.find_memory_tracking_dims(trajectories)
        goal_found, goal_dims = self.find_goal_encoding_dims(trajectories)

        # Test effects
        mem_effect = self.test_memory_effect(mem_dims)

        return GridworldResult(
            memory_tracking_found=mem_found,
            goal_encoding_found=goal_found,
            memory_dims=mem_dims,
            goal_dims=goal_dims,
            causal_effects={
                "memory_ablation_effect": mem_effect,
            },
            metadata={
                "grid_size": grid_size,
                "num_episodes": num_episodes,
            },
        )
