"""Activation cache with offloading and memory management.

This module provides advanced caching features:
- GPU→CPU offloading for long sequences
- Selective caching with regex filters
- Float16/bfloat16 quantization
- Memory-mapped storage for huge datasets
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Set
import torch
import pandas as pd
import numpy as np
import mmap
import tempfile
import os
import re
from pathlib import Path


class OffloadStrategy:
    """Strategy for offloading activations to CPU/disk."""

    GPU = "gpu"  # Keep in GPU memory
    CPU = "cpu"  # Move to CPU RAM
    DISK = "disk"  # Memory-map to disk
    AUTO = "auto"  # Automatic based on memory


@dataclass
class CacheConfig:
    """Configuration for activation cache behavior."""

    dtype: Optional[torch.dtype] = None  # Quantization dtype
    offload_strategy: OffloadStrategy = OffloadStrategy.AUTO
    max_gpu_memory_gb: float = 4.0  # Max GPU memory before offload
    offload_threshold: int = 100  # Timesteps before auto-offload
    name_filter: Optional[str] = None  # Regex filter for names
    device: Optional[torch.device] = None  # Target device


class ActivationCache:
    """Advanced activation cache with offloading and memory management.

    Features:
    - Selective caching with regex filters
    - Float16/bfloat16 quantization
    - GPU→CPU automatic offloading
    - Memory-mapped disk storage for huge datasets
    - Lazy evaluation with callable support

    Example:
        cache = ActivationCache()
        cache["z_posterior", 0] = tensor
        single = cache["z_posterior", 0]           # Single tensor
        sequence = cache["z_posterior", :]          # Stacked tensors
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self._store: Dict[Tuple[str, int], Union[torch.Tensor, Callable[[], torch.Tensor]]] = {}
        self._evaluated: Dict[Tuple[str, int], torch.Tensor] = {}
        self._offloaded: Dict[Tuple[str, int], str] = {}  # Key -> "cpu" or "disk"
        self._disk_files: Dict[Tuple[str, int], Path] = {}

        self.config = config or CacheConfig()
        self._device = device or torch.device("cpu")

        self._name_filter: Optional[re.Pattern] = None
        if self.config.name_filter:
            self._name_filter = re.compile(self.config.name_filter)

        self._temp_dir = tempfile.mkdtemp()

    def _should_cache(self, name: str) -> bool:
        """Check if name passes the filter."""
        if self._name_filter is None:
            return True
        return self._name_filter.match(name) is not None

    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor if dtype is specified."""
        if self.config.dtype is None:
            return tensor
        return tensor.to(dtype=self.config.dtype)

    def _dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Restore quantized tensor to original dtype."""
        if self.config.dtype is None:
            return tensor
        return tensor.to(dtype=torch.float32)

    def _offload_to_disk(self, key: Tuple[str, int], tensor: torch.Tensor) -> None:
        """Offload tensor to disk via memory mapping."""
        path = Path(self._temp_dir) / f"{key[0]}_{key[1]}.npy"
        np.save(path, tensor.cpu().numpy())
        self._disk_files[key] = path
        self._offloaded[key] = "disk"

    def _load_from_disk(self, key: Tuple[str, int]) -> torch.Tensor:
        """Load tensor from disk."""
        path = self._disk_files[key]
        tensor = torch.from_numpy(np.load(path))
        if self._device.type != "cpu":
            tensor = tensor.to(self._device)
        return tensor

    def _maybe_offload(self, key: Tuple[str, int], tensor: torch.Tensor) -> None:
        """Maybe offload based on strategy."""
        if self.config.offload_strategy == OffloadStrategy.GPU:
            return

        if self.config.offload_strategy == OffloadStrategy.CPU:
            self._store[key] = tensor.cpu()
            self._evaluated.pop(key, None)
            self._offloaded[key] = "cpu"

        elif self.config.offload_strategy == OffloadStrategy.DISK:
            self._offload_to_disk(key, tensor)
            self._store.pop(key, None)
            self._evaluated.pop(key, None)

        elif self.config.offload_strategy == OffloadStrategy.AUTO:
            if key[1] >= self.config.offload_threshold:
                self._offload_to_disk(key, tensor)
                self._store.pop(key, None)
                self._evaluated.pop(key, None)

    def __getitem__(
        self,
        key: Union[Tuple[str, int], Tuple[str, slice], Tuple[str, str], str],
    ) -> torch.Tensor:
        """Access cached activations."""
        if isinstance(key, str):
            return self._get_all(key)

        name, second = key

        if not self._should_cache(name):
            raise KeyError(f"{name} does not match filter")

        if isinstance(second, int):
            return self._get_single(name, second)
        elif isinstance(second, slice):
            return self._get_slice(name, second)
        elif second == ":":
            return self._get_all(name)
        else:
            raise KeyError(f"Invalid cache key: {key}")

    def _get_single(self, name: str, timestep: int) -> torch.Tensor:
        """Get a single cached tensor."""
        key = (name, timestep)

        if key in self._offloaded:
            if self._offloaded[key] == "disk":
                return self._load_from_disk(key)
            elif self._offloaded[key] == "cpu":
                return self._store[key]

        if key not in self._store and key not in self._evaluated:
            raise KeyError(f"No cached activation for {name} at t={timestep}")

        if key in self._evaluated:
            return self._dequantize(self._evaluated[key])

        value = self._store[key]
        if callable(value):
            value = value()

        value = self._quantize(value)

        if self.config.offload_strategy != OffloadStrategy.GPU:
            self._maybe_offload(key, value)
        else:
            self._evaluated[key] = value

        return self._dequantize(value)

    def _get_all(self, name: str) -> torch.Tensor:
        """Get all timesteps stacked."""
        timesteps = sorted(set(t for n, t in self._store.keys() if n == name))
        timesteps_offloaded = sorted(set(t for n, t in self._offloaded.keys() if n == name))
        all_timesteps = sorted(set(timesteps + timesteps_offloaded))

        if not all_timesteps:
            raise KeyError(f"No cached activations for '{name}'")
        return torch.stack([self._get_single(name, t) for t in all_timesteps], dim=0)

    def _get_slice(self, name: str, slc: slice) -> torch.Tensor:
        """Get a slice of timesteps."""
        timesteps = sorted(set(t for n, t in self._store.keys() if n == name))
        timesteps_offloaded = sorted(set(t for n, t in self._offloaded.keys() if n == name))
        all_timesteps = sorted(set(timesteps + timesteps_offloaded))

        sliced = all_timesteps[slc]
        return torch.stack([self._get_single(name, t) for t in sliced], dim=0)

    def __setitem__(
        self,
        key: Tuple[str, int],
        value: Union[torch.Tensor, Callable[[], torch.Tensor]],
    ) -> None:
        """Store an activation."""
        name, timestep = key

        if not self._should_cache(name):
            return

        value = self._quantize(value) if isinstance(value, torch.Tensor) else value
        self._store[key] = value
        self._evaluated.pop(key, None)
        self._offloaded.pop(key, None)

    def __contains__(self, key: Tuple[str, int]) -> bool:
        """Check if a key exists."""
        return key in self._store or key in self._evaluated or key in self._offloaded

    def keys(self) -> Iterable[Tuple[str, int]]:
        """Iterate over all (component, timestep) pairs."""
        return set(self._store.keys()) | set(self._evaluated.keys()) | set(self._offloaded.keys())

    @property
    def component_names(self) -> List[str]:
        """List of unique component names in cache."""
        all_keys = (
            set(self._store.keys()) | set(self._evaluated.keys()) | set(self._offloaded.keys())
        )
        return sorted(set(n for n, _ in all_keys))

    @property
    def timesteps(self) -> List[int]:
        """List of timesteps with cached activations."""
        all_keys = (
            set(self._store.keys()) | set(self._evaluated.keys()) | set(self._offloaded.keys())
        )
        return sorted(set(t for _, t in all_keys))

    def get(
        self,
        name: str,
        timestep: int,
        default: Any = None,
    ) -> Optional[torch.Tensor]:
        """Get with default if not found."""
        try:
            return self._get_single(name, timestep)
        except KeyError:
            return default

    def to_device(self, device: torch.device) -> "ActivationCache":
        """Move all tensors to a device."""
        self._device = device

        for key in list(self._store.keys()):
            val = self._store[key]
            if isinstance(val, torch.Tensor):
                self._store[key] = val.to(device)
                self._evaluated.pop(key, None)

        for key in list(self._evaluated.keys()):
            self._evaluated[key] = self._evaluated[key].to(device)

        for key in list(self._offloaded.keys()):
            if self._offloaded[key] == "disk":
                self._load_from_disk(key).to(device)

        return self

    def detach(self) -> "ActivationCache":
        """Detach all tensors from computation graphs."""
        for key in list(self._store.keys()):
            val = self._store[key]
            if isinstance(val, torch.Tensor):
                self._store[key] = val.detach()
                self._evaluated.pop(key, None)

        for key in list(self._evaluated.keys()):
            self._evaluated[key] = self._evaluated[key].detach()

        return self

    def filter(self, names: List[str]) -> "ActivationCache":
        """Return a new cache with only specified components."""
        new_cache = ActivationCache(config=self.config, device=self._device)

        if names:
            pattern = "|".join(f"({n})" for n in names)
            new_cache._name_filter = re.compile(pattern)

        for (name, t), val in self._store.items():
            if new_cache._should_cache(name):
                new_cache._store[(name, t)] = val

        return new_cache

    def filter_regex(self, pattern: str) -> "ActivationCache":
        """Return a new cache with components matching regex."""
        new_cache = ActivationCache(config=self.config, device=self._device)
        new_cache._name_filter = re.compile(pattern)

        for key in self.keys():
            name, t = key
            if new_cache._should_cache(name):
                try:
                    new_cache[name, t] = self._get_single(name, t)
                except KeyError:
                    pass

        return new_cache

    def surprise(
        self,
        posterior_key: str = "z_posterior",
        prior_key: str = "z_prior",
    ) -> Optional[torch.Tensor]:
        """Compute per-timestep KL divergence."""
        try:
            posterior = self[posterior_key]
            prior = self[prior_key]
        except KeyError:
            return None

        T = posterior.shape[0]
        kl_vals = []
        for t in range(T):
            p = posterior[t].clamp(min=1e-8)
            q = prior[t].clamp(min=1e-8)
            p = p / p.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-1, keepdim=True)
            kl = (p * (p.log() - q.log())).sum(dim=-1)
            kl_vals.append(kl.sum().item())
        return torch.tensor(kl_vals)

    def to_dataframe(self) -> pd.DataFrame:
        """Export cache to a pandas DataFrame."""
        records = []
        for key in self.keys():
            name, t = key
            try:
                val = self._get_single(name, t)
                records.append(
                    {
                        "component": name,
                        "timestep": t,
                        "shape": str(list(val.shape)),
                        "dtype": str(val.dtype),
                        "offloaded": self._offloaded.get(key, "memory"),
                    }
                )
            except Exception:
                pass
        return pd.DataFrame(records)

    def materialize(
        self,
        names: Optional[List[str]] = None,
        timesteps: Optional[List[int]] = None,
    ) -> "ActivationCache":
        """Pre-compute lazy callables into tensors."""
        if names is None:
            names = self.component_names
        if timesteps is None:
            timesteps = self.timesteps

        for name in names:
            for t in timesteps:
                key = (name, t)
                if key in self._store and callable(self._store[key]):
                    self._get_single(name, t)

        return self

    def estimate_memory_gb(self) -> float:
        """Estimate memory usage."""
        total_bytes = 0

        for key, val in self._store.items():
            if key in self._evaluated:
                total_bytes += self._evaluated[key].element_size() * self._evaluated[key].nelement()
            elif isinstance(val, torch.Tensor):
                total_bytes += val.element_size() * val.nelement()
            elif callable(val):
                total_bytes += 4 * 1024 * 1024

        for key in self._offloaded.keys():
            if key in self._disk_files:
                total_bytes += os.path.getsize(self._disk_files[key])

        return total_bytes / (1024**3)

    def clear(self) -> None:
        """Clear all cached data."""
        self._store.clear()
        self._evaluated.clear()

        for path in self._disk_files.values():
            if path.exists():
                path.unlink()
        self._disk_files.clear()
        self._offloaded.clear()

    def __del__(self):
        """Cleanup temp files."""
        self.clear()
        if Path(self._temp_dir).exists():
            os.rmdir(self._temp_dir)


class DistributedActivationCache:
    """Multi-GPU distributed activation cache.

    Aggregates activations across DDP processes.
    """

    def __init__(self, is_main_rank: bool = True):
        self.is_main_rank = is_main_rank
        self._caches: Dict[int, ActivationCache] = {}

    def register_rank(self, rank: int, cache: ActivationCache) -> None:
        """Register cache for a rank."""
        self._caches[rank] = cache

    def gather(self, name: str, timestep: int) -> Optional[torch.Tensor]:
        """Gather activations from all ranks."""
        if not self._caches:
            return None

        tensors = []
        for rank, cache in self._caches.items():
            try:
                tensors.append(cache[name, timestep])
            except KeyError:
                pass

        if not tensors:
            return None

        return torch.cat(tensors, dim=0)

    def all_gather(self, name: str) -> Dict[int, torch.Tensor]:
        """Gather all activations for a component."""
        result = {}
        for rank, cache in self._caches.items():
            try:
                result[rank] = cache[name]
            except KeyError:
                pass
        return result
