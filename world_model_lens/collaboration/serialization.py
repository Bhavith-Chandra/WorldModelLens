"""Artifact serialization for efficient storage."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import zarr
import numcodecs

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.lazy_trajectory import (
    LatentTrajectoryLite,
    TensorStore,
    TrajectoryDataset,
)


class ArtifactSerializer:
    """Base class for artifact serialization."""

    def to_disk(self, path: str, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def from_disk(cls, path: str, **kwargs) -> Any:
        raise NotImplementedError


class ActivationCacheSerializer:
    """Serialize ActivationCache to disk with efficient compression."""

    def to_disk(
        self,
        cache: ActivationCache,
        path: str,
        format: str = "zarr",
        compress: bool = True,
    ) -> None:
        """Save activation cache to disk.

        Args:
            cache: ActivationCache to save.
            path: Output path.
            format: 'zarr' (default) or 'hdf5'.
            compress: Whether to compress (zarr only).
        """
        if format == "zarr":
            self._to_zarr(cache, path, compress)
        elif format == "hdf5":
            self._to_hdf5(cache, path)
        elif format == "numpy":
            self._to_numpy(cache, path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_zarr(
        self,
        cache: ActivationCache,
        path: str,
        compress: bool,
    ) -> None:
        compressor = numcodecs.Zstd(level=3) if compress else None
        root = zarr.open_group(path, mode="w")

        metadata = {
            "component_names": cache.component_names,
            "timesteps": cache.timesteps,
        }
        root.attrs["metadata"] = metadata

        # Iterate via public API to avoid accessing internals.
        for name in cache.component_names:
            for t in cache.timesteps:
                val = cache.get(name, t, None)
                if isinstance(val, torch.Tensor):
                    key = f"{name}_{t}"
                    root.array(
                        key,
                        data=val.cpu().numpy(),
                        compressor=compressor,
                    )

    def _to_hdf5(self, cache: ActivationCache, path: str) -> None:
        import h5py

        with h5py.File(path, "w") as f:
            metadata = {
                "component_names": cache.component_names,
                "timesteps": cache.timesteps,
            }
            f.attrs["metadata"] = json.dumps(metadata)

            for name in cache.component_names:
                for t in cache.timesteps:
                    val = cache.get(name, t, None)
                    if isinstance(val, torch.Tensor):
                        key = f"{name}_{t}"
                        f.create_dataset(key, data=val.cpu().numpy())

    def _to_numpy(self, cache: ActivationCache, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

        metadata = {
            "component_names": cache.component_names,
            "timesteps": cache.timesteps,
        }
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f)

        for name in cache.component_names:
            for t in cache.timesteps:
                val = cache.get(name, t, None)
                if isinstance(val, torch.Tensor):
                    np.save(f"{path}/{name}_{t}.npy", val.cpu().numpy())

    @classmethod
    def from_disk(
        cls,
        path: str,
        format: str = "zarr",
    ) -> ActivationCache:
        """Load activation cache from disk."""
        if format == "zarr":
            return cls._from_zarr(path)
        elif format == "hdf5":
            return cls._from_hdf5(path)
        elif format == "numpy":
            return cls._from_numpy(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def _from_zarr(cls, path: str) -> ActivationCache:
        root = zarr.open_group(path, mode="r")
        cache = ActivationCache()

        for key in root.array_keys():
            name, t_str = key.rsplit("_", 1)
            t = int(t_str)
            tensor = torch.from_numpy(root[key][:])
            cache[name, t] = tensor

        return cache

    @classmethod
    def _from_hdf5(cls, path: str) -> ActivationCache:
        import h5py

        cache = ActivationCache()
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if key.startswith("metadata"):
                    continue
                name, t_str = key.rsplit("_", 1)
                t = int(t_str)
                tensor = torch.from_numpy(f[key][:])
                cache[name, t] = tensor

        return cache

    @classmethod
    def _from_numpy(cls, path: str) -> ActivationCache:
        cache = ActivationCache()
        metadata_path = Path(path) / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                _ = json.load(f)

        for np_file in Path(path).glob("*.npy"):
            name, t_str = np_file.stem.rsplit("_", 1)
            t = int(t_str)
            tensor = torch.from_numpy(np.load(np_file))
            cache[name, t] = tensor

        return cache


class TrajectoryDatasetSerializer:
    """Serialize TrajectoryDataset to disk."""

    def to_disk(
        self,
        dataset: TrajectoryDataset,
        path: str,
        format: str = "zarr",
        save_metadata: bool = True,
    ) -> None:
        """Save trajectory dataset to disk.

        Args:
            dataset: TrajectoryDataset to save.
            path: Output path.
            format: 'zarr' (default) or 'hdf5'.
            save_metadata: Whether to save episode metadata.
        """
        if format == "zarr":
            self._to_zarr(dataset, path, save_metadata)
        elif format == "hdf5":
            self._to_hdf5(dataset, path, save_metadata)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_zarr(
        self,
        dataset: TrajectoryDataset,
        path: str,
        save_metadata: bool,
    ) -> None:
        root = zarr.open_group(path, mode="w")

        if save_metadata:
            episode_metas = []
            for traj in dataset.trajectories:
                episode_metas.append(traj.metadata)
            root.attrs["episode_metadata"] = json.dumps(episode_metas)

        for key, tensor in dataset.store.tensors.items():
            if tensor is not None:
                root.array(
                    key,
                    data=tensor.cpu().numpy(),
                    compressor=numcodecs.Zstd(level=3),
                )

    def _to_hdf5(
        self,
        dataset: TrajectoryDataset,
        path: str,
        save_metadata: bool,
    ) -> None:
        import h5py

        with h5py.File(path, "w") as f:
            if save_metadata:
                episode_metas = []
                for traj in dataset.trajectories:
                    episode_metas.append(traj.metadata)
                f.attrs["episode_metadata"] = json.dumps(episode_metas)

            for key, tensor in dataset.store.tensors.items():
                if tensor is not None:
                    f.create_dataset(key, data=tensor.cpu().numpy())

    @classmethod
    def from_disk(
        cls,
        path: str,
        format: str = "zarr",
    ) -> TrajectoryDataset:
        """Load trajectory dataset from disk."""
        if format == "zarr":
            return cls._from_zarr(path)
        elif format == "hdf5":
            return cls._from_hdf5(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def _from_zarr(cls, path: str) -> TrajectoryDataset:
        root = zarr.open_group(path, mode="r")
        store = TensorStore()

        for key in root.array_keys():
            store.tensors[key] = torch.from_numpy(root[key][:])

        episode_metadata = []
        if "episode_metadata" in root.attrs:
            episode_metadata = json.loads(root.attrs["episode_metadata"])

        if not episode_metadata:
            length = store.tensors["h"].shape[0] if "h" in store.tensors else 0
            episode_metadata = [{"length": length, "episode_id": i} for i in range(1)]

        return TrajectoryDataset.from_store(store, episode_metadata)

    @classmethod
    def _from_hdf5(cls, path: str) -> TrajectoryDataset:
        import h5py

        store = TensorStore()
        with h5py.File(path, "r") as f:
            for key in f.keys():
                store.tensors[key] = torch.from_numpy(f[key][:])

            episode_metadata = []
            if "episode_metadata" in f.attrs:
                episode_metadata = json.loads(f.attrs["episode_metadata"])

            if not episode_metadata:
                length = store.tensors["h"].shape[0] if "h" in store.tensors else 0
                episode_metadata = [{"length": length, "episode_id": i} for i in range(1)]

        return TrajectoryDataset.from_store(store, episode_metadata)


def serialize_cache(
    cache: ActivationCache,
    path: str,
    format: str = "zarr",
) -> None:
    """Convenience function to serialize activation cache."""
    serializer = ActivationCacheSerializer()
    serializer.to_disk(cache, path, format)


def deserialize_cache(
    path: str,
    format: str = "zarr",
) -> ActivationCache:
    """Convenience function to deserialize activation cache."""
    serializer = ActivationCacheSerializer()
    return serializer.from_disk(path, format)


def serialize_dataset(
    dataset: TrajectoryDataset,
    path: str,
    format: str = "zarr",
) -> None:
    """Convenience function to serialize trajectory dataset."""
    serializer = TrajectoryDatasetSerializer()
    serializer.to_disk(dataset, path, format)


def deserialize_dataset(
    path: str,
    format: str = "zarr",
) -> TrajectoryDataset:
    """Convenience function to deserialize trajectory dataset."""
    serializer = TrajectoryDatasetSerializer()
    return serializer.from_disk(path, format)
