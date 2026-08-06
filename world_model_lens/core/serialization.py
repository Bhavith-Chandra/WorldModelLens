"""Trajectory serialization for save/load functionality.

This module provides tools to serialize and deserialize trajectories,
activation caches, and world model states for debugging and analysis.
"""

from __future__ import annotations

import json
import pickle
import gzip
from pathlib import Path
from typing import Any, Optional
import torch
import numpy as np


class TrajectorySerializer:
    """Serialize and deserialize world model trajectories."""

    @staticmethod
    def save(
        trajectory: Any,
        path: str | Path,
        format: str = "pkl",
        compress: bool = True,
    ) -> None:
        """Save trajectory to file.

        Args:
            trajectory: WorldTrajectory to save
            path: Output path
            format: Format ("pkl", "json", "numpy")
            compress: Whether to compress
        """
        path = Path(path)

        if format == "pkl":
            with gzip.open(path, "wb") if compress else open(path, "wb") as f:
                pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == "json":
            data = TrajectorySerializer._trajectory_to_dict(trajectory)
            with gzip.open(path, "wt") if compress else open(path, "w") as f:
                json.dump(data, f, indent=2, default=_json_serializer)
        elif format == "numpy":
            data = TrajectorySerializer._trajectory_to_dict(trajectory)
            np.savez_compressed(path, **data) if compress else np.savez(path, **data)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def load(
        path: str | Path,
        format: str = "pkl",
    ) -> Any:
        """Load trajectory from file.

        Args:
            path: Input path
            format: Format ("pkl", "json", "numpy")

        Returns:
            WorldTrajectory
        """
        path = Path(path)

        if format == "pkl":
            with gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as f:
                return pickle.load(f)
        elif format == "json":
            with gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r") as f:
                data = json.load(f)
                return TrajectorySerializer._dict_to_trajectory(data)
        elif format == "numpy":
            data = np.load(path)
            return TrajectorySerializer._dict_to_trajectory(dict(data))
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _trajectory_to_dict(trajectory: Any) -> dict[str, Any]:
        """Convert trajectory to dictionary."""
        from world_model_lens import WorldState

        states_data = []
        for state in trajectory.states:
            state_dict = {
                "timestep": state.timestep,
                "state": state.state.cpu().numpy() if state.state is not None else None,
            }
            if state.action is not None:
                state_dict["action"] = state.action.cpu().numpy()
            if state.reward is not None:
                state_dict["reward"] = (
                    state.reward.item() if hasattr(state.reward, "item") else state.reward
                )
            if state.done is not None:
                state_dict["done"] = state.done
            if state.metadata:
                state_dict["metadata"] = state.metadata

            states_data.append(state_dict)

        return {
            "states": states_data,
            "source": getattr(trajectory, "source", "unknown"),
            "length": len(trajectory.states),
        }

    @staticmethod
    def _dict_to_trajectory(data: dict[str, Any]) -> Any:
        """Convert dictionary to trajectory."""
        from world_model_lens import WorldState, WorldTrajectory

        states = []
        for state_data in data["states"]:
            state = WorldState(
                state=torch.from_numpy(state_data["state"])
                if state_data["state"] is not None
                else None,
                timestep=state_data["timestep"],
                action=torch.from_numpy(state_data["action"])
                if state_data.get("action") is not None
                else None,
                reward=torch.tensor(state_data["reward"])
                if state_data.get("reward") is not None
                else None,
                done=state_data.get("done"),
                metadata=state_data.get("metadata", {}),
            )
            states.append(state)

        return WorldTrajectory(states=states, source=data.get("source", "loaded"))


class ActivationCacheSerializer:
    """Serialize and deserialize activation caches."""

    @staticmethod
    def save(
        cache: Any,
        path: str | Path,
        compress: bool = True,
    ) -> None:
        """Save activation cache to file.

        Args:
            cache: ActivationCache to save
            path: Output path
            compress: Whether to compress
        """
        path = Path(path)
        data = ActivationCacheSerializer._cache_to_dict(cache)

        with gzip.open(path, "wb") if compress else open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> Any:
        """Load activation cache from file.

        Args:
            path: Input path

        Returns:
            ActivationCache
        """
        path = Path(path)

        with gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as f:
            data = pickle.load(f)

        return ActivationCacheSerializer._dict_to_cache(data)

    @staticmethod
    def _cache_to_dict(cache: Any) -> dict[str, Any]:
        """Convert cache to dictionary."""
        data = {
            "activations": {},
            "component_names": list(cache.component_names)
            if hasattr(cache, "component_names")
            else [],
            "timesteps": list(cache.timesteps) if hasattr(cache, "timesteps") else [],
        }

        for key in cache.keys():
            val = cache[key]
            if isinstance(val, torch.Tensor):
                data["activations"][str(key)] = val.cpu().numpy()
            else:
                data["activations"][str(key)] = val

        return data

    @staticmethod
    def _dict_to_cache(data: dict[str, Any]) -> Any:
        """Convert dictionary to cache."""
        from world_model_lens import ActivationCache

        cache = ActivationCache()

        for key_str, val in data["activations"].items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            cache[key_str] = val

        return cache


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for custom types."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)
