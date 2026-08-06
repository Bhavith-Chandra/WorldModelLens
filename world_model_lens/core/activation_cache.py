from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
import torch
import torch.distributions as dist


class ActivationCache:
    """Storage for activations keyed by (component_name, timestep).

    Supports single-item access, slicing, lazy evaluation, and export.
    Now supports storing torch.distributions.Distribution objects and dicts
    with distribution parameters for uncertainty analysis.

    Key Features:
    - Store tensors, distributions, dicts, or lazy callables
    - Automatic mean extraction from distributions for backward compatibility
    - Enhanced KL divergence computation for uncertainty analysis
    - Distribution parameter access for variance analysis

    Example:
        cache = ActivationCache()

        # Store tensor (backward compatible)
        cache["z_posterior", 0] = torch.randn(64)

        # Store distribution for uncertainty analysis
        import torch.distributions as dist
        cache["z_posterior", 0] = dist.Normal(mean, std)

        # Access (returns mean for backward compatibility)
        posterior = cache["z_posterior", 0]  # tensor

        # Check if stored value is a distribution
        if cache.is_distribution("z_posterior", 0):
            params = cache.get_distribution_params("z_posterior", 0)
            variance = params["variance"]  # access uncertainty

        # Slicing works with all types
        sequence = cache["z_posterior", :]  # stacked means
    """

    def __init__(self):
        self._store: dict[
            tuple[str, int],
            torch.Tensor | Callable[[], torch.Tensor] | dist.Distribution | dict[str, Any],
        ] = {}
        self._evaluated: dict[
            tuple[str, int], torch.Tensor | dist.Distribution | dict[str, Any]
        ] = {}

    def __getitem__(
        self, key: str | tuple[str, int] | tuple[str, slice] | tuple[str, str]
    ) -> torch.Tensor:
        """Access cached activations.

        For backward compatibility, when distributions are stored, returns the mean tensor.
        Use get_distribution_params() to access full distribution parameters.

        Args:
            key: Either:
                - (name, timestep): Single tensor (mean if distribution stored)
                - (name, slice): Tensor sequence via slice
                - (name, ':'): All timesteps stacked
                - name: All timesteps stacked

        Returns:
            torch.Tensor: Requested activation(s). For distributions, returns the mean.
        """
        if isinstance(key, str):
            return self._get_all(key)

        name, second = key

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
        if key not in self._store and key not in self._evaluated:
            raise KeyError(f"No cached activation for {name} at t={timestep}")

        if key in self._evaluated:
            value = self._evaluated[key]
        else:
            value = self._store[key]
            if callable(value):
                value = value()
                self._evaluated[key] = value

        # Handle different types of stored values
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, dist.Distribution):
            return value.mean
        elif isinstance(value, dict):
            # Return tensor/mean if available
            if "tensor" in value:
                return value["tensor"]
            elif "mean" in value:
                return value["mean"]
            else:
                # If no tensor available, return a placeholder or raise error
                # For backward compatibility, assume dict contains the data
                raise ValueError(
                    f"Dict stored for {name},{timestep} does not contain 'tensor' or 'mean' key"
                )
        else:
            raise ValueError(f"Unsupported stored value type: {type(value)}")

    def _get_all(self, name: str) -> torch.Tensor:
        """Get all timesteps stacked."""
        timesteps = sorted({t for n, t in self._store.keys() if n == name})
        if not timesteps:
            raise KeyError(f"No cached activations for '{name}'")
        return torch.stack([self._get_single(name, t) for t in timesteps], dim=0)

    def _get_slice(self, name: str, slc: slice) -> torch.Tensor:
        """Get a slice of timesteps."""
        timesteps = sorted({t for n, t in self._store.keys() if n == name})
        sliced = timesteps[slc]
        return torch.stack([self._get_single(name, t) for t in sliced], dim=0)

    def __setitem__(
        self,
        key: tuple[str, int],
        value: torch.Tensor | Callable[[], torch.Tensor] | dist.Distribution | dict[str, Any],
    ) -> None:
        """Store an activation.

        Args:
            key: Tuple of (component_name, timestep).
            value: Activation to store. Can be:
                - torch.Tensor: Raw activation tensor
                - Callable: Lazy evaluation function returning tensor/distribution/dict
                - torch.distributions.Distribution: Full distribution for uncertainty analysis
                - dict: Dictionary with distribution parameters (e.g., {"mean": tensor, "std": tensor})
        """
        name, timestep = key
        self._store[key] = value
        self._evaluated.pop(key, None)

    def store(self, name: str, timestep: int, value: torch.Tensor) -> None:
        """Public helper to store a concrete tensor for (name, timestep)."""
        self[name, timestep] = value

    def __contains__(self, key: tuple[str, int]) -> bool:
        """Check if a key exists in the cache."""
        return key in self._store or key in self._evaluated

    def keys(self) -> Iterable[tuple[str, int]]:
        """Iterate over all (component, timestep) pairs."""
        return self._store.keys()

    @property
    def component_names(self) -> list[str]:
        """List of unique component names in cache."""
        return sorted({n for n, _ in self._store.keys()})

    @property
    def timesteps(self) -> list[int]:
        """List of timesteps with cached activations."""
        return sorted({t for _, t in self._store.keys()})

    def get(self, name: str, timestep: int, default: Any = None) -> torch.Tensor | None:
        """Get with default if not found."""
        try:
            return self._get_single(name, timestep)
        except KeyError:
            return default

    def to_device(self, device: torch.device) -> "ActivationCache":
        """Move all tensors to a device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        for key in list(self._store.keys()):
            val = self._store[key]
            if isinstance(val, torch.Tensor):
                self._store[key] = val.to(device)
                self._evaluated.pop(key, None)
            elif isinstance(val, dist.Distribution):
                # Distributions don't have to_device, but their parameters might
                pass  # For now, assume distributions are handled elsewhere
        for key in list(self._evaluated.keys()):
            val = self._evaluated[key]
            if isinstance(val, torch.Tensor):
                self._evaluated[key] = val.to(device)
            elif isinstance(val, dist.Distribution):
                pass
        return self

    def detach(self) -> "ActivationCache":
        """Detach all tensors from the computation graph."""
        for key in list(self._store.keys()):
            val = self._store[key]
            if isinstance(val, torch.Tensor):
                self._store[key] = val.detach()
                self._evaluated.pop(key, None)
            elif isinstance(val, dist.Distribution):
                # For distributions, we can't detach them directly, but their parameters
                # should be detached when evaluated. For now, leave as is.
                pass
        for key in list(self._evaluated.keys()):
            val = self._evaluated[key]
            if isinstance(val, torch.Tensor):
                self._evaluated[key] = val.detach()
            elif isinstance(val, dist.Distribution):
                pass
        return self

    def filter(self, names: list[str]) -> "ActivationCache":
        """Return a new cache with only specified components.

        Args:
            names: Components to keep.

        Returns:
            New ActivationCache with filtered contents.
        """
        new_cache = ActivationCache()
        for (name, t), val in self._store.items():
            if name in names:
                new_cache._store[(name, t)] = val
        return new_cache

    def surprise(self) -> torch.Tensor:
        """Compute per-timestep KL divergence (surprise) between z_posterior and z_prior.

        Supports both tensor and distribution inputs. When distributions are stored,
        uses torch.distributions.kl_divergence() for accurate computation.

        Returns:
            Tensor of shape [T] with KL values per timestep. Higher values indicate
            more surprising predictions (larger variance spikes).
        """
        try:
            posterior_vals = []
            prior_vals = []
            timesteps = sorted({t for n, t in self._store.keys() if n == "z_posterior"})
            if not timesteps:
                raise KeyError("No z_posterior found")

            for t in timesteps:
                post_key = ("z_posterior", t)
                prior_key = ("z_prior", t)

                if post_key in self._store:
                    post_val = self._store[post_key]
                elif post_key in self._evaluated:
                    post_val = self._evaluated[post_key]
                else:
                    raise KeyError(f"No z_posterior at t={t}")

                if prior_key in self._store:
                    prior_val = self._store[prior_key]
                elif prior_key in self._evaluated:
                    prior_val = self._evaluated[prior_key]
                else:
                    raise KeyError(f"No z_prior at t={t}")

                posterior_vals.append(post_val)
                prior_vals.append(prior_val)

            kl_vals = []
            for p_val, q_val in zip(posterior_vals, prior_vals):
                if isinstance(p_val, dist.Distribution) and isinstance(q_val, dist.Distribution):
                    # Use torch.distributions KL divergence
                    kl = dist.kl_divergence(p_val, q_val).sum(dim=-1)
                    kl_vals.append(kl.item())
                elif isinstance(p_val, dist.Distribution) and isinstance(q_val, torch.Tensor):
                    # posterior is distribution, prior is tensor
                    p_probs = p_val.mean  # Use mean as approximation
                    kl = (p_probs * (p_probs.log() - q_val.log())).sum(dim=-1)
                    kl_vals.append(kl.item())
                elif isinstance(p_val, torch.Tensor) and isinstance(q_val, dist.Distribution):
                    # posterior is tensor, prior is distribution
                    q_probs = q_val.mean  # Use mean as approximation
                    kl = (p_val * (p_val.log() - q_probs.log())).sum(dim=-1)
                    kl_vals.append(kl.item())
                else:
                    # Fallback to manual KL calculation for tensors
                    p = (
                        p_val.clamp(min=1e-8)
                        if isinstance(p_val, torch.Tensor)
                        else p_val.mean.clamp(min=1e-8)
                    )
                    q = (
                        q_val.clamp(min=1e-8)
                        if isinstance(q_val, torch.Tensor)
                        else q_val.mean.clamp(min=1e-8)
                    )
                    p = p / p.sum(dim=-1, keepdim=True)
                    q = q / q.sum(dim=-1, keepdim=True)
                    kl = (p * (p.log() - q.log())).sum(dim=-1)
                    kl_vals.append(kl.sum().item())

            return torch.tensor(kl_vals)
        except KeyError as err:
            raise KeyError("Cache must contain z_posterior and z_prior for surprise()") from err

    def stacked(self, name: str) -> torch.Tensor:
        """Explicitly return all timesteps for `name` stacked along dim=0.

        This is equivalent to accessing `cache[name]` or `cache[name, ':']` but
        provides a more discoverable API.
        """
        return self._get_all(name)

    def diff(
        self, other: "ActivationCache", names: list[str] | None = None, absolute: bool = True
    ) -> "ActivationCache":
        """Compute per-key tensor differences between this cache and `other`.

        For any (component, timestep) present in both caches the result will
        contain `self - other` (or absolute difference when `absolute=True`).

        Args:
            other: Other ActivationCache to compare against.
            names: Optional list of component names to restrict comparison.
            absolute: If True, store the absolute difference.

        Returns:
            ActivationCache containing difference tensors for matching keys.
        """
        new_cache = ActivationCache()
        if names is None:
            names = sorted(set(self.component_names) & set(other.component_names))

        for name in names:
            # get sorted timesteps present in both
            timesteps = sorted(
                {t for n, t in self._store.keys() if n == name}
                & {t for n, t in other._store.keys() if n == name}
            )
            for t in timesteps:
                a = self._get_single(name, t)
                b = other._get_single(name, t)
                diff = a - b
                if absolute:
                    diff = diff.abs()
                # store concrete tensor
                new_cache._store[(name, t)] = diff

        return new_cache

    def temporal_variability(self, name: str, p: float = 2.0) -> torch.Tensor:
        """Compute normed differences between consecutive timesteps for `name`.

        Returns a tensor of shape [T-1] where each entry is the p-norm of the
        difference between timestep t and t-1. Index i corresponds to change
        from timestep i to i+1 in the stacked ordering.
        """
        seq = self._get_all(name)  # [T, ...]
        if seq.shape[0] < 2:
            return torch.tensor([])
        diffs = seq[1:] - seq[:-1]
        flat = diffs.reshape(diffs.shape[0], -1)
        return torch.norm(flat, p=p, dim=1)

    def most_variable_timesteps(self, name: str, top_k: int = 5) -> list[int]:
        """Return the timesteps with the largest temporal changes for `name`.

        The returned timesteps correspond to the later timestep of each change
        (i.e. change between t-1 and t is reported as t). If `top_k` is larger
        than available changes, all candidate timesteps are returned.
        """
        vari = self.temporal_variability(name)
        if vari.numel() == 0:
            return []
        k = min(top_k, vari.numel())
        vals, idx = torch.topk(vari, k)
        # map indices (0..T-2) to actual timestep numbers (use stored timesteps)
        timesteps = sorted({t for n, t in self._store.keys() if n == name})
        result = [timesteps[i + 1] for i in idx.tolist()]
        return result

    def timesteps_exceeding_surprise(self, threshold: float) -> list[int]:
        """Return timesteps where surprise() exceeds `threshold`.

        Uses the same ordering as `surprise()` / `stacked('z_posterior')`.
        """
        kl = self.surprise()
        timesteps = sorted(
            {t for _, t in self._store.keys() if _ == "z_posterior" or _ == "z_prior"}
        )
        # surprise() raises if z_prior/posterior missing; map indices to sorted timesteps
        exceed_idx = (kl > threshold).nonzero(as_tuple=True)[0].tolist()
        return [timesteps[i] for i in exceed_idx]

    def compare_summary(
        self, other: "ActivationCache", names: list[str] | None = None, p: float = 2.0
    ) -> pd.DataFrame:
        """Produce a DataFrame summarizing normed differences per (component,timestep).

        Columns: component, timestep, diff_norm
        """
        rows = []
        if names is None:
            names = sorted(set(self.component_names) & set(other.component_names))
        for name in names:
            timesteps = sorted(
                {t for n, t in self._store.keys() if n == name}
                & {t for n, t in other._store.keys() if n == name}
            )
            for t in timesteps:
                a = self._get_single(name, t)
                b = other._get_single(name, t)
                diff = (a - b).reshape(-1).norm(p=p).item()
                rows.append({"component": name, "timestep": t, "diff_norm": diff})
        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Export cache to a pandas DataFrame.

        Returns:
            DataFrame with columns: component, timestep, shape.
        """
        records = []
        for (name, t), val in self._store.items():
            if isinstance(val, torch.Tensor):
                records.append(
                    {
                        "component": name,
                        "timestep": t,
                        "shape": str(list(val.shape)),
                        "dtype": str(val.dtype),
                    }
                )
        return pd.DataFrame(records)

    def materialize(
        self, names: list[str] | None = None, timesteps: list[int] | None = None
    ) -> "ActivationCache":
        """Pre-compute a subset of lazy callables into tensors.

        Args:
            names: Component names to materialize (None = all).
            timesteps: Timesteps to materialize (None = all).

        Returns:
            Self for chaining.
        """
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

    def set_prior_equivalent(
        self,
        timestep: int,
        logits: torch.Tensor,
        probs: torch.Tensor,
        names_filter: list[str] | None = None,
    ) -> None:
        """Store prior-equivalent entries for timestep when posterior==prior.

        Writes both ``z_prior.logits`` and ``z_prior`` for convenience. If
        *names_filter* is provided, only writes components present in the
        filter.
        """
        if names_filter is None or "z_prior.logits" in names_filter:
            self["z_prior.logits", timestep] = logits
        if names_filter is None or "z_prior" in names_filter:
            self["z_prior", timestep] = probs

    def get_distribution_params(self, name: str, timestep: int) -> dict[str, Any]:
        """Get distribution parameters for uncertainty analysis.

        Args:
            name: Component name
            timestep: Timestep

        Returns:
            Dict with keys like 'mean', 'std', 'variance', etc.
        """
        key = (name, timestep)
        if key not in self._store and key not in self._evaluated:
            raise KeyError(f"No cached activation for {name} at t={timestep}")

        if key in self._evaluated:
            value = self._evaluated[key]
        else:
            value = self._store[key]
            if callable(value):
                value = value()
                self._evaluated[key] = value

        if isinstance(value, dist.Distribution):
            params = {"mean": value.mean}
            if hasattr(value, "stddev"):
                params["std"] = value.stddev
            if hasattr(value, "variance"):
                params["variance"] = value.variance
            return params
        elif isinstance(value, dict):
            return value
        else:
            # For tensors, assume it's the mean
            return {"mean": value}

    # KV helper methods -------------------------------------------------
    def set_kv(self, layer: int, kind: str, timestep: int, tensor: torch.Tensor) -> None:
        """Set a KV-style entry for a transformer-like adapter.

        Args:
            layer: Layer index (int).
            kind: Either 'k' or 'v' (key or value).
            timestep: Timestep index.
            tensor: Tensor to store.
        """
        name = f"kv/l{layer}/{kind}"
        self[name, timestep] = tensor

    def get_kv(self, layer: int, kind: str, timestep: int, default: Any = None) -> Any:
        """Get a KV entry if present, otherwise return default.

        Returns the stored tensor (or distribution mean) following the same
        semantics as __getitem__/_get_single.
        """
        try:
            return self._get_single(f"kv/l{layer}/{kind}", timestep)
        except KeyError:
            return default

    def delete_kv(self, layer: int, kind: str, timestep: int) -> None:
        """Delete a KV entry if present.

        This removes both stored and evaluated entries for the (name,timestep)
        key so subsequent reads behave as-if the entry never existed.
        """
        name = f"kv/l{layer}/{kind}"
        key = (name, timestep)
        self._store.pop(key, None)
        self._evaluated.pop(key, None)

    def is_distribution(self, name: str, timestep: int) -> bool:
        """Check if the cached value for (name, timestep) is a distribution.

        Args:
            name: Component name
            timestep: Timestep

        Returns:
            True if the stored value is a torch.distributions.Distribution
        """
        key = (name, timestep)
        if key in self._evaluated:
            return isinstance(self._evaluated[key], dist.Distribution)
        elif key in self._store:
            val = self._store[key]
            if callable(val):
                # Don't evaluate callable
                return False
            return isinstance(val, dist.Distribution)
        return False

    def estimate_memory_gb(self) -> float:
        """Predict RAM usage before materializing all callables.

        Returns:
            Estimated memory in gigabytes.
        """
        total_bytes = 0
        for key, val in self._store.items():
            if key in self._evaluated:
                eval_val = self._evaluated[key]
                if isinstance(eval_val, torch.Tensor):
                    total_bytes += eval_val.element_size() * eval_val.nelement()
            elif isinstance(val, torch.Tensor):
                total_bytes += val.element_size() * val.nelement()
            elif isinstance(val, dist.Distribution):
                # Estimate based on distribution parameters
                # This is approximate
                total_bytes += 4 * 1024 * 1024
            elif callable(val):
                estimated_size = 4 * 1024 * 1024
                total_bytes += estimated_size
        return total_bytes / (1024**3)


# Re-export CacheQuery for backward compatibility. Import after ActivationCache
# is defined to avoid a circular import (cache_query imports ActivationCache).
from .cache_query import CacheQuery  # re-export for backward compatibility

__all__ = ["ActivationCache", "CacheQuery"]
