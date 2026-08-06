"""Memory-efficient trajectory storage with lazy loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator, Sequence
import torch
import numpy as np


@dataclass
class LatentTrajectoryLite:
    """Memory-efficient trajectory that stores indices into a shared tensor store.

    Only stores metadata + indices, not full tensors. Tensors are loaded on-demand
    from a shared store, enabling efficient batching across large trajectory datasets.

    Example:
        # Create shared tensor store
        store = TensorStore()
        store.add("h", torch.randn(1000, 512))  # 1000 timesteps, 512 dim
        store.add("z_posterior", torch.randn(1000, 32, 32))

        # Create lite trajectory referencing store indices
        traj = LatentTrajectoryLite(
            episode_id=0,
            start_idx=0,
            length=100,
            store=store,
            metadata={"env_name": "cartpole", "total_reward": 500.0},
        )

        # Access is lazy - tensors only loaded when needed
        h_seq = traj.h_sequence  # Loads from store on demand
    """

    episode_id: int
    start_idx: int
    length: int
    store: "TensorStore"
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cached_h: Optional[torch.Tensor] = field(default=None, repr=False)
    _cached_z_posterior: Optional[torch.Tensor] = field(default=None, repr=False)
    _cached_z_prior: Optional[torch.Tensor] = field(default=None, repr=False)
    _cached_actions: Optional[torch.Tensor] = field(default=None, repr=False)
    _cached_rewards: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def env_name(self) -> str:
        return self.metadata.get("env_name", "unknown")

    @property
    def imagined(self) -> bool:
        return self.metadata.get("imagined", False)

    @property
    def fork_point(self) -> Optional[int]:
        return self.metadata.get("fork_point", None)

    @property
    def indices(self) -> range:
        """Range of indices into the tensor store."""
        return range(self.start_idx, self.start_idx + self.length)

    def _load(self, key: str) -> Optional[torch.Tensor]:
        """Load tensor from store if available."""
        if key in self.store.tensors:
            tensor = self.store.tensors[key]
            return tensor[self.start_idx : self.start_idx + self.length]
        return None

    @property
    def h_sequence(self) -> torch.Tensor:
        """Get hidden state sequence [T, d_h], loaded lazily."""
        if self._cached_h is None:
            self._cached_h = self._load("h")
            if self._cached_h is None and "h" in self.store.tensors:
                self._cached_h = self._load("h")
        return self._cached_h

    @property
    def z_posterior_sequence(self) -> Optional[torch.Tensor]:
        """Get posterior latent sequence [T, n_cat, n_cls]."""
        if self._cached_z_posterior is None:
            self._cached_z_posterior = self._load("z_posterior")
        return self._cached_z_posterior

    @property
    def z_prior_sequence(self) -> Optional[torch.Tensor]:
        """Get prior latent sequence [T, n_cat, n_cls]."""
        if self._cached_z_prior is None:
            self._cached_z_prior = self._load("z_prior")
        return self._cached_z_prior

    @property
    def actions(self) -> Optional[torch.Tensor]:
        """Get action sequence [T, d_action]."""
        if self._cached_actions is None:
            self._cached_actions = self._load("actions")
        return self._cached_actions

    @property
    def rewards_real(self) -> Optional[torch.Tensor]:
        """Get real rewards [T]."""
        if self._cached_rewards is None:
            self._cached_rewards = self._load("rewards")
        return self._cached_rewards

    @property
    def rewards_pred(self) -> Optional[torch.Tensor]:
        """Get predicted rewards [T]."""
        return self._load("reward_pred")

    @property
    def kl_sequence(self) -> Optional[torch.Tensor]:
        """Get KL divergence sequence [T]."""
        return self._load("kl")

    def slice(self, start: int, end: int) -> "LatentTrajectoryLite":
        """Create a slice referencing the same store."""
        return LatentTrajectoryLite(
            episode_id=self.episode_id,
            start_idx=self.start_idx + start,
            length=end - start,
            store=self.store,
            metadata=self.metadata.copy(),
        )

    def to_device(self, device: torch.device) -> "LatentTrajectoryLite":
        """Move to device (loads all tensors)."""
        new_store = self.store.to_device(device)
        return LatentTrajectoryLite(
            episode_id=self.episode_id,
            start_idx=self.start_idx,
            length=self.length,
            store=new_store,
            metadata=self.metadata.copy(),
        )

    def cache(self) -> None:
        """Load all tensors into memory for faster access."""
        _ = self.h_sequence
        _ = self.z_posterior_sequence
        _ = self.z_prior_sequence
        _ = self.actions
        _ = self.rewards_real

    def uncache(self) -> None:
        """Clear cached tensors to free memory."""
        self._cached_h = None
        self._cached_z_posterior = None
        self._cached_z_prior = None
        self._cached_actions = None
        self._cached_rewards = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single timestep as dict."""
        result = {}
        for key in self.store.tensors:
            tensor = self.store.tensors[key]
            if tensor is not None and self.start_idx + idx < tensor.shape[0]:
                result[key] = tensor[self.start_idx + idx]
        return result


class TensorStore:
    """Shared tensor storage for efficient memory usage across trajectories.

    Stores large tensors that multiple LatentTrajectoryLite objects can index into.
    Supports lazy loading, device transfer, and HDF5/zarr serialization.
    """

    def __init__(self):
        self.tensors: Dict[str, Optional[torch.Tensor]] = {}
        self._device = torch.device("cpu")

    def add(self, key: str, tensor: torch.Tensor) -> None:
        """Add a named tensor to the store."""
        self.tensors[key] = tensor

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor by name."""
        return self.tensors.get(key)

    def keys(self) -> List[str]:
        return list(self.tensors.keys())

    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {k: v.shape for k, v in self.tensors.items() if v is not None}

    def to_device(self, device: torch.device) -> "TensorStore":
        """Move all tensors to device."""
        new_store = TensorStore()
        for key, tensor in self.tensors.items():
            if tensor is not None:
                new_store.tensors[key] = tensor.to(device)
            else:
                new_store.tensors[key] = None
        return new_store

    def to_disk(self, path: str, format: str = "zarr") -> None:
        """Serialize to disk.

        Args:
            path: Output path.
            format: 'zarr' or 'hdf5'.
        """
        if format == "zarr":
            try:
                import zarr
                import numcodecs
            except ImportError:
                raise ImportError("zarr required: pip install zarr")

            root = zarr.open_group(path, mode="w")
            for key, tensor in self.tensors.items():
                if tensor is not None:
                    root.array(
                        key,
                        data=tensor.cpu().numpy(),
                        compressor=numcodecs.Zstd(level=3),
                    )
        elif format == "hdf5":
            try:
                import h5py
            except ImportError:
                raise ImportError("h5py required: pip install h5py")

            with h5py.File(path, "w") as f:
                for key, tensor in self.tensors.items():
                    if tensor is not None:
                        f.create_dataset(key, data=tensor.cpu().numpy())
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def from_disk(cls, path: str, format: str = "zarr") -> "TensorStore":
        """Load from disk."""
        store = cls()

        if format == "zarr":
            try:
                import zarr
            except ImportError:
                raise ImportError("zarr required: pip install zarr")

            root = zarr.open_group(path, mode="r")
            for key in root.array_keys():
                store.tensors[key] = torch.from_numpy(root[key][:])
        elif format == "hdf5":
            try:
                import h5py
            except ImportError:
                raise ImportError("h5py required: pip install h5py")

            with h5py.File(path, "r") as f:
                for key in f.keys():
                    store.tensors[key] = torch.from_numpy(f[key][:])
        else:
            raise ValueError(f"Unknown format: {format}")

        return store

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def total_bytes(self) -> int:
        """Total memory usage in bytes."""
        total = 0
        for tensor in self.tensors.values():
            if tensor is not None:
                total += tensor.element_size() * tensor.nelement()
        return total

    def __repr__(self) -> str:
        return f"TensorStore({self.shapes()}, {self.total_bytes / 1e9:.2f} GB)"


class TrajectoryDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for trajectory collections.

    Supports lazy loading, batching, and efficient memory usage via TensorStore.

    Example:
        # From list of trajectories
        dataset = TrajectoryDataset(trajectories)

        # From TensorStore + metadata
        dataset = TrajectoryDataset.from_store(store, episode_metadata)

        # Standard PyTorch usage
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch in loader:
            h = batch["h"]  # [B, T, d_h]
    """

    def __init__(
        self,
        trajectories: Optional[List[LatentTrajectoryLite]] = None,
        store: Optional[TensorStore] = None,
        episode_metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        if trajectories is not None:
            self.trajectories = trajectories
            self.store = trajectories[0].store if trajectories else None
        elif store is not None and episode_metadata is not None:
            self.store = store
            self.trajectories = []
            for i, meta in enumerate(episode_metadata):
                length = meta.get("length", store.tensors["h"].shape[0])
                traj = LatentTrajectoryLite(
                    episode_id=meta.get("episode_id", i),
                    start_idx=meta.get("start_idx", 0),
                    length=length,
                    store=store,
                    metadata=meta,
                )
                self.trajectories.append(traj)
        else:
            raise ValueError("Must provide either trajectories or store+metadata")

    @classmethod
    def from_store(
        cls,
        store: TensorStore,
        episode_metadata: List[Dict[str, Any]],
    ) -> "TrajectoryDataset":
        """Create dataset from TensorStore and episode metadata."""
        return cls(store=store, episode_metadata=episode_metadata)

    @classmethod
    def from_disk(
        cls,
        path: str,
        format: str = "zarr",
        metadata_path: Optional[str] = None,
    ) -> "TrajectoryDataset":
        """Load dataset from zarr/HDF5 on disk.

        Args:
            path: Path to zarr/HDF5 file.
            format: 'zarr' or 'hdf5'.
            metadata_path: Optional path to JSON metadata file with episode info.

        Returns:
            Loaded TrajectoryDataset.
        """
        import json
        import os

        store = TensorStore.from_disk(path, format)

        metadata = []
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            total_timesteps = store.tensors["h"].shape[0] if "h" in store.tensors else 0
            if total_timesteps > 0:
                episode_starts = [0]
                if "episode_boundaries" in store.tensors:
                    boundaries = store.tensors["episode_boundaries"]
                    episode_starts = boundaries.tolist()
                for i, start in enumerate(episode_starts):
                    end = episode_starts[i + 1] if i + 1 < len(episode_starts) else total_timesteps
                    metadata.append(
                        {
                            "episode_id": i,
                            "start_idx": start,
                            "length": end - start,
                        }
                    )

        return cls(store=store, episode_metadata=metadata)

    def to_disk(
        self,
        path: str,
        format: str = "zarr",
        metadata_path: Optional[str] = None,
    ) -> None:
        """Save dataset to disk.

        Args:
            path: Output path.
            format: 'zarr' or 'hdf5'.
            metadata_path: Optional path to save episode metadata JSON.
        """
        import json
        import os

        self.store.to_disk(path, format)

        if metadata_path:
            metadata = [
                {
                    "episode_id": t.episode_id,
                    "start_idx": t.start_idx,
                    "length": t.length,
                    "env_name": t.env_name,
                    "imagined": t.imagined,
                    **t.metadata,
                }
                for t in self.trajectories
            ]
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get trajectory as dict of tensors."""
        traj = self.trajectories[idx]
        result = {
            "episode_id": traj.episode_id,
            "length": traj.length,
            "metadata": traj.metadata,
        }

        for key in traj.store.tensors:
            tensor = traj.store.tensors[key]
            if tensor is not None:
                result[key] = tensor[traj.start_idx : traj.start_idx + traj.length]

        return result

    def statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        lengths = [t.length for t in self.trajectories]
        rewards = []
        for t in self.trajectories:
            if t.rewards_real is not None:
                rewards.extend(t.rewards_real.tolist())

        return {
            "n_trajectories": len(self.trajectories),
            "total_timesteps": sum(lengths),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "mean_reward": np.mean(rewards) if rewards else None,
            "std_reward": np.std(rewards) if rewards else None,
        }

    def filter(self, fn: Callable[[LatentTrajectoryLite], bool]) -> "TrajectoryDataset":
        """Filter trajectories by predicate."""
        filtered = [t for t in self.trajectories if fn(t)]
        return TrajectoryDataset(trajectories=filtered)

    def split(self, ratio: float) -> Tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Split into train/test."""
        n = int(len(self.trajectories) * ratio)
        return (
            TrajectoryDataset(trajectories=self.trajectories[:n]),
            TrajectoryDataset(trajectories=self.trajectories[n:]),
        )


def collate_trajectories(
    batch: List[Dict[str, Any]],
    max_length: Optional[int] = None,
    pad_axis: int = 0,
) -> Dict[str, Any]:
    """Collate function for variable-length trajectories.

    Pads sequences to max_length in batch.

    Args:
        batch: List of trajectory dicts from Dataset.__getitem__.
        max_length: Optional max length for padding.
        pad_axis: Axis to pad.

    Returns:
        Batched dict with padded tensors.
    """
    result = {}
    max_len = max_length or max(b["length"] for b in batch)

    for key in batch[0].keys():
        if key in ("episode_id", "length", "metadata"):
            result[key] = [b[key] for b in batch]
            continue

        first_val = batch[0][key]
        if not isinstance(first_val, torch.Tensor):
            result[key] = [b[key] for b in batch]
            continue

        # Pad variable-length tensors
        padded = []
        for b in batch:
            tensor = b[key]
            if tensor.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - tensor.shape[0],
                    *tensor.shape[1:],
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, padding], dim=0)
            padded.append(tensor)

        result[key] = torch.stack(padded, dim=0)

    return result


def create_data_loader(
    dataset: TrajectoryDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_length: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create DataLoader with proper collate_fn."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_trajectories(b, max_length=max_length),
    )
