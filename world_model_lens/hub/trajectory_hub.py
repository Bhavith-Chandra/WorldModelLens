"""Trajectory hub for managing trajectory datasets."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence, cast
from dataclasses import dataclass

import os
import pickle
from pathlib import Path

from world_model_lens.core import LatentTrajectory


@dataclass
class TrajectoryStats:
    """Statistics for a trajectory dataset."""

    n_trajectories: int
    total_timesteps: int
    mean_length: float
    std_length: float
    mean_reward: Optional[float] = None
    std_reward: Optional[float] = None


class TrajectoryDataset:
    """Dataset wrapper for a collection of LatentTrajectory objects.

    Supports filtering, mapping, and splitting.
    """

    def __init__(self, trajectories: List[LatentTrajectory]):
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> LatentTrajectory:
        return self.trajectories[index]

    def filter(self, fn: Callable[[LatentTrajectory], bool]) -> "TrajectoryDataset":
        """Filter trajectories by a predicate.

        Args:
            fn: Function that returns True to keep a trajectory.

        Returns:
            New TrajectoryDataset with filtered trajectories.
        """
        return TrajectoryDataset([t for t in self.trajectories if fn(t)])

    def map(self, fn: Callable[[LatentTrajectory], LatentTrajectory]) -> "TrajectoryDataset":
        """Apply a function to all trajectories.

        Args:
            fn: Function to apply to each trajectory.

        Returns:
            New TrajectoryDataset with transformed trajectories.
        """
        return TrajectoryDataset([fn(t) for t in self.trajectories])

    def split(self, ratio: float) -> Tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Split dataset into train/test.

        Args:
            ratio: Fraction for first split (e.g., 0.8 for 80/20).

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        n = int(len(self.trajectories) * ratio)
        train = TrajectoryDataset(self.trajectories[:n])
        test = TrajectoryDataset(self.trajectories[n:])
        return train, test

    def statistics(self) -> TrajectoryStats:
        """Compute dataset statistics.

        Returns:
            TrajectoryStats with dataset-level statistics.
        """
        lengths = [t.length for t in self.trajectories]
        rewards = [
            t.rewards_real.mean().item() for t in self.trajectories if t.rewards_real is not None
        ]

        return TrajectoryStats(
            n_trajectories=len(self.trajectories),
            total_timesteps=sum(lengths),
            mean_length=sum(lengths) / len(lengths) if lengths else 0,
            std_length=self._std([float(l) for l in lengths]),
            mean_reward=sum(rewards) / len(rewards) if rewards else None,
            std_reward=self._std(rewards) if rewards else None,
        )

    @staticmethod
    def _std(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return cast(float, variance**0.5)




class TrajectoryHub:
    """Local storage for trajectory datasets.

    Datasets are saved as .pkl files in the local cache directory.
    """

    def __init__(self, root_dir: str = "~/.cache/world_model_lens/trajectories") -> None:
        """Initialize TrajectoryHub.

        Args:
            root_dir: Root directory for storing trajectory datasets.
        """
        self.root_dir = Path(root_dir).expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, dataset: TrajectoryDataset, name: str) -> str:
        """Save a trajectory dataset to the hub.

        Args:
            dataset: TrajectoryDataset to save.
            name: Name to register under.

        Returns:
            Path where the dataset was saved.
        """
        path = self.root_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(dataset, f)
        return str(path)

    def load(self, name: str) -> TrajectoryDataset:
        """Load a saved trajectory dataset.

        Args:
            name: Name of the saved dataset.

        Returns:
            TrajectoryDataset.

        Raises:
            FileNotFoundError: If name not found.
        """
        path = self.root_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No dataset named '{name}' found at {path}")

        with open(path, "rb") as f:
            return cast(TrajectoryDataset, pickle.load(f))

    def list_saved(self) -> List[str]:
        """List all saved dataset names."""
        pkl_files = self.root_dir.glob("*.pkl")
        return [f.stem for f in pkl_files]

    def delete(self, name: str) -> None:
        """Delete a saved dataset.

        Args:
            name: Name of the dataset to delete.

        Raises:
            FileNotFoundError: If dataset not found.
        """
        path = self.root_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No dataset named '{name}' found at {path}")
        path.unlink()
