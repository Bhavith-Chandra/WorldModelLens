"""Analysis-oriented query helpers for ActivationCache.

This module provides `CacheQuery`, a thin, discoverable layer for common
analysis patterns (stacking, differencing, top-k timesteps, correlations).
CacheQuery delegates to ActivationCache's public API and keeps logic focused
on returning simple numeric types useful in notebooks and plots.
"""

from typing import List, Optional

import torch

from .activation_cache import ActivationCache


class CacheQuery:
    """Convenience query helpers built on top of ActivationCache.

    This keeps common analysis patterns in one place so callers do not need
    to iterate timesteps manually.
    """

    def __init__(self, cache: ActivationCache):
        self.cache = cache

    def stack(self, name: str) -> torch.Tensor:
        """Return all timesteps for `name` stacked as (T, *activation_shape).

        Raises KeyError if the component is missing.
        """
        return self.cache.stacked(name)

    def diff(self, other: ActivationCache, name: str, align: str = "intersection") -> torch.Tensor:
        """Element-wise difference between two caches for the same activation.

        Aligns timesteps by `align` strategy. Currently only 'intersection' is
        supported (default). Returns tensor of shape (T_common, *activation_shape)
        where T_common is the number of timesteps present in both caches for
        `name`.
        """
        if align != "intersection":
            raise ValueError("Only 'intersection' align mode is supported currently")

        t_self = sorted(t for n, t in self.cache.keys() if n == name)
        t_other = sorted(t for n, t in other.keys() if n == name)
        common = [t for t in t_self if t in t_other]
        if not common:
            raise KeyError(f"No common timesteps for component '{name}'")
        parts = []
        for t in common:
            a = self.cache.get(name, t)
            b = other.get(name, t)
            if a is None or b is None:
                raise KeyError(f"Missing timestep {t} for component '{name}' in one of the caches")
            parts.append(a - b)
        return torch.stack(parts, dim=0)

    def top_k_timesteps(self, name: str, k: int, reduce: str = "norm") -> List[int]:
        """Return timesteps with largest activation magnitude for `name`.

        reduce: one of {"norm", "sum", "max", "mean"} and defines the scalar
        metric computed per-timestep from the activation tensor. Returns a list
        of timesteps ordered by descending metric value (length <= k).
        """
        if k <= 0:
            return []
        timesteps = sorted(t for n, t in self.cache.keys() if n == name)
        if not timesteps:
            raise KeyError(f"No timesteps for component '{name}'")
        metrics = []
        for t in timesteps:
            val = self.cache.get(name, t)
            if val is None:
                continue
            flat = val.reshape(-1)
            if reduce == "norm":
                m = float(flat.norm().item())
            elif reduce == "sum":
                m = float(flat.abs().sum().item())
            elif reduce == "max":
                m = float(flat.abs().max().item())
            elif reduce == "mean":
                m = float(flat.mean().item())
            else:
                raise ValueError(f"Unknown reduce '{reduce}'")
            metrics.append(m)
        vals = torch.tensor(metrics)
        k2 = min(k, vals.numel())
        _, topk_idx = torch.topk(vals, k2)
        return [timesteps[i] for i in topk_idx.tolist()]

    def correlation(
        self, name_a: str, name_b: str, reduce: str = "mean", per_dim: bool = False
    ) -> torch.Tensor:
        """Pearson correlation between two activation streams across timesteps.

        If `per_dim` is False (default) returns a scalar correlation between
        the reduced time series. If `per_dim` is True, computes correlation for
        each activation dimension (requires activations to have the same shape
        across timesteps) and returns a 1D tensor of correlations.
        """
        t_a = sorted(t for n, t in self.cache.keys() if n == name_a)
        t_b = sorted(t for n, t in self.cache.keys() if n == name_b)
        common = [t for t in t_a if t in t_b]
        if len(common) < 2:
            raise ValueError("Need at least two common timesteps to compute correlation")

        series_a = []
        series_b = []
        per_dim_series_a = []
        per_dim_series_b = []
        for t in common:
            a = self.cache.get(name_a, t)
            b = self.cache.get(name_b, t)
            if a is None or b is None:
                raise KeyError(f"Missing timestep {t} for components")
            a_flat = a.reshape(-1)
            b_flat = b.reshape(-1)
            if per_dim:
                per_dim_series_a.append(a_flat)
                per_dim_series_b.append(b_flat)
            else:
                if reduce == "mean":
                    series_a.append(float(a_flat.mean().item()))
                    series_b.append(float(b_flat.mean().item()))
                elif reduce == "norm":
                    series_a.append(float(a_flat.norm().item()))
                    series_b.append(float(b_flat.norm().item()))
                elif reduce == "sum":
                    series_a.append(float(a_flat.abs().sum().item()))
                    series_b.append(float(b_flat.abs().sum().item()))
                else:
                    raise ValueError(f"Unknown reduce '{reduce}'")

        if per_dim:
            # stack to [T, D]
            A = torch.stack(per_dim_series_a, dim=0)
            B = torch.stack(per_dim_series_b, dim=0)
            # zero-mean
            A = A - A.mean(dim=0)
            B = B - B.mean(dim=0)
            num = (A * B).sum(dim=0)
            den = torch.sqrt((A * A).sum(dim=0) * (B * B).sum(dim=0))
            corr = num / den
            corr[den == 0] = float("nan")
            return corr

        xa = torch.tensor(series_a)
        xb = torch.tensor(series_b)
        xa = xa - xa.mean()
        xb = xb - xb.mean()
        num = (xa * xb).sum()
        den = torch.sqrt((xa * xa).sum() * (xb * xb).sum())
        if den == 0:
            return torch.tensor(float("nan"))
        return num / den
