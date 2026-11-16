"""Activation patching for causal analysis."""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.hooks import HookContext, HookPoint
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.world_trajectory import WorldTrajectory
from world_model_lens.hooked_world_model import HookedWorldModel


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PatchResult:
    """Result of a single patch experiment."""

    metric_clean: float
    metric_corrupted: float
    metric_patched: float
    recovery_rate: float
    component: str
    timestep: int
    patch_mode: str
    original_value: Optional[Tensor] = None
    patched_value: Optional[Tensor] = None

    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Component: {self.component}, t={self.timestep}\n"
            f"Clean: {self.metric_clean:.4f}, "
            f"Corrupted: {self.metric_corrupted:.4f}, "
            f"Patched: {self.metric_patched:.4f}\n"
            f"Recovery: {self.recovery_rate:.2%}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_clean": self.metric_clean,
            "metric_corrupted": self.metric_corrupted,
            "metric_patched": self.metric_patched,
            "recovery_rate": self.recovery_rate,
            "component": self.component,
            "timestep": self.timestep,
            "patch_mode": self.patch_mode,
        }


@dataclass
class PatchingSweepResult:
    """Result of full patching sweep."""

    results: Dict[Tuple[str, int], PatchResult]
    components: List[str]
    timesteps: List[int]
    device: torch.device = field(default_factory=_get_device)

    def recovery_matrix(self) -> Tensor:
        """Get recovery rate matrix [n_components, T]."""
        matrix = torch.zeros(len(self.components), len(self.timesteps), device=self.device)
        for i, comp in enumerate(self.components):
            for j, t in enumerate(self.timesteps):
                key = (comp, t)
                if key in self.results:
                    matrix[i, j] = self.results[key].recovery_rate
        return matrix

    def top_k_patches(self, k: int = 10) -> List[PatchResult]:
        """Get top-k patches by recovery rate, skipping unmeasurable (NaN) ones."""
        measured = [r for r in self.results.values() if not math.isnan(r.recovery_rate)]
        return sorted(measured, key=lambda r: r.recovery_rate, reverse=True)[:k]

    def get_component_rankings(self) -> Dict[str, float]:
        """Get average recovery rate per component."""
        component_scores: Dict[str, List[float]] = {}
        for (comp, _), result in self.results.items():
            if not math.isnan(result.recovery_rate):
                component_scores.setdefault(comp, []).append(result.recovery_rate)
        return {comp: sum(scores) / len(scores) for comp, scores in component_scores.items()}

    def get_timestep_rankings(self) -> Dict[int, float]:
        """Get average recovery rate per timestep."""
        timestep_scores: Dict[int, List[float]] = {}
        for (_, t), result in self.results.items():
            if not math.isnan(result.recovery_rate):
                timestep_scores.setdefault(t, []).append(result.recovery_rate)
        return {t: sum(scores) / len(scores) for t, scores in timestep_scores.items()}

    def heatmap(self, title: str = "Patching Recovery", figsize: Tuple[int, int] = (12, 8)):
        """Plot heatmap of recovery rates."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
            return None

        matrix = self.recovery_matrix().cpu().numpy()
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

        ax.set_yticks(range(len(self.components)))
        ax.set_yticklabels(self.components)
        ax.set_xticks(range(len(self.timesteps)))
        ax.set_xticklabels(self.timesteps)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Component")
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label="Recovery Rate")
        return fig


class TemporalPatcher:
    """Activation patching experiments.

    Tests causal roles by replacing activations and measuring recovery.
    """

    def __init__(self, wm: HookedWorldModel, device: Optional[torch.device] = None):
        """Initialize the patcher.

        Args:
            wm: The hooked world model.
            device: Optional device override.
        """
        self.wm = wm
        self.device = device or _get_device()

    def _ensure_device(self, tensor: Tensor) -> Tensor:
        """Move tensor to device if needed."""
        return tensor.to(self.device) if tensor.device != self.device else tensor

    def compute_metric_from_cache(
        self,
        cache: ActivationCache,
        metric_fn: Callable[[ActivationCache], float],
    ) -> float:
        """Compute a metric from an activation cache."""
        return float(metric_fn(cache))

    def compute_metric_from_trajectory(
        self,
        traj: WorldTrajectory,
        metric_fn: Callable[[WorldTrajectory], float],
    ) -> float:
        """Compute a metric from a trajectory."""
        return float(metric_fn(traj))

    def _run_with_patches(
        self, obs_seq: Tensor, action_seq: Tensor, hooks: List[HookPoint]
    ) -> Tuple[WorldTrajectory, ActivationCache]:
        with self.wm.hook_registry.temp_hooks(hooks):
            traj, _, cache = self.wm.run_with_cache(
                self._ensure_device(obs_seq), self._ensure_device(action_seq)
            )
        return traj, cache

    def patch_state(
        self,
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        patch_component: str,
        patch_at_timestep: int,
        metric_fn: Callable[[ActivationCache], float],
        clean_obs_seq: Optional[Tensor] = None,
        clean_action_seq: Optional[Tensor] = None,
        patch_value: Optional[Tensor] = None,
        corrupted_obs_seq: Optional[Tensor] = None,
        corrupted_action_seq: Optional[Tensor] = None,
    ) -> PatchResult:
        """Patch one component at one timestep and measure the effect.

        With corrupted inputs, this denoises: the corrupted inputs are run with
        the clean activation patched in. Without them, this noises: the clean
        inputs are run with the corrupted activation patched in.

        The recovery rate is ``(patched - corrupted) / (clean - corrupted)`` of
        ``metric_fn``, unclipped, and NaN when the clean and corrupted metrics
        are equal.

        Args:
            clean_cache: Cache from the clean run.
            corrupted_cache: Cache from the corrupted run.
            patch_component: Component to patch.
            patch_at_timestep: Timestep to patch.
            metric_fn: Maps an ActivationCache to a scalar.
            clean_obs_seq: Clean observations.
            clean_action_seq: Clean actions.
            patch_value: Value to patch in instead of the cached one.
            corrupted_obs_seq: Corrupted observations, to denoise.
            corrupted_action_seq: Corrupted actions; defaults to the clean actions.

        Returns:
            PatchResult with the recovery rate.
        """
        if clean_obs_seq is None or clean_action_seq is None:
            raise ValueError("clean_obs_seq and clean_action_seq are required")

        denoise = corrupted_obs_seq is not None
        if denoise:
            obs_seq = corrupted_obs_seq
            action_seq = clean_action_seq if corrupted_action_seq is None else corrupted_action_seq
            source_cache = clean_cache
        else:
            obs_seq, action_seq = clean_obs_seq, clean_action_seq
            source_cache = corrupted_cache

        if patch_value is None:
            patch_value = source_cache[patch_component, patch_at_timestep]
        patch_value = self._ensure_device(patch_value.detach().clone())

        recorded: Dict[str, Tensor] = {}

        def replace(tensor: Tensor, ctx: HookContext) -> Tensor:
            recorded["original"] = tensor.detach().clone()
            return patch_value

        hook = HookPoint(name=patch_component, fn=replace, timestep=patch_at_timestep)
        _, patched_cache = self._run_with_patches(obs_seq, action_seq, [hook])

        clean_metric = self.compute_metric_from_cache(clean_cache, metric_fn)
        corrupted_metric = self.compute_metric_from_cache(corrupted_cache, metric_fn)
        patched_metric = self.compute_metric_from_cache(patched_cache, metric_fn)

        return PatchResult(
            metric_clean=clean_metric,
            metric_corrupted=corrupted_metric,
            metric_patched=patched_metric,
            recovery_rate=_recovery(clean_metric, corrupted_metric, patched_metric),
            component=patch_component,
            timestep=patch_at_timestep,
            patch_mode="denoise" if denoise else "noise",
            original_value=recorded.get("original"),
            patched_value=patch_value,
        )

    def patch_state_with_corruption(
        self,
        clean_obs_seq: Tensor,
        clean_action_seq: Tensor,
        corruption_fn: Callable[[Tensor], Tensor],
        patch_component: str,
        patch_at_timestep: int,
        metric_fn: Callable[[WorldTrajectory], float],
    ) -> PatchResult:
        """Corrupt a component at every timestep, then restore it at one timestep.

        The corrupted run applies ``corruption_fn`` to ``patch_component`` at
        every timestep. The patched run does the same, except that the clean
        value is restored at ``patch_at_timestep``.

        Args:
            clean_obs_seq: Clean observations.
            clean_action_seq: Clean actions.
            corruption_fn: Maps an activation to its corrupted version.
            patch_component: Component to corrupt and restore.
            patch_at_timestep: Timestep at which the clean value is restored.
            metric_fn: Maps a WorldTrajectory to a scalar.

        Returns:
            PatchResult with the recovery rate.
        """
        clean_traj, clean_cache = self._run_with_patches(clean_obs_seq, clean_action_seq, [])
        clean_value = clean_cache[patch_component, patch_at_timestep].detach().clone()

        def corrupt(tensor: Tensor, ctx: HookContext) -> Tensor:
            return corruption_fn(tensor)

        def restore(tensor: Tensor, ctx: HookContext) -> Tensor:
            return clean_value

        corrupt_hook = HookPoint(name=patch_component, fn=corrupt)
        restore_hook = HookPoint(name=patch_component, fn=restore, timestep=patch_at_timestep)

        corrupted_traj, _ = self._run_with_patches(
            clean_obs_seq, clean_action_seq, [corrupt_hook]
        )
        patched_traj, _ = self._run_with_patches(
            clean_obs_seq, clean_action_seq, [corrupt_hook, restore_hook]
        )

        clean_metric = self.compute_metric_from_trajectory(clean_traj, metric_fn)
        corrupted_metric = self.compute_metric_from_trajectory(corrupted_traj, metric_fn)
        patched_metric = self.compute_metric_from_trajectory(patched_traj, metric_fn)

        return PatchResult(
            metric_clean=clean_metric,
            metric_corrupted=corrupted_metric,
            metric_patched=patched_metric,
            recovery_rate=_recovery(clean_metric, corrupted_metric, patched_metric),
            component=patch_component,
            timestep=patch_at_timestep,
            patch_mode="auto_corrupt",
            patched_value=clean_value,
        )

    def full_sweep(
        self,
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        components: List[str],
        metric_fn: Callable[[ActivationCache], float],
        t_range: Optional[List[int]] = None,
        clean_obs_seq: Optional[Tensor] = None,
        clean_action_seq: Optional[Tensor] = None,
        parallel: bool = False,
        corrupted_obs_seq: Optional[Tensor] = None,
        corrupted_action_seq: Optional[Tensor] = None,
    ) -> PatchingSweepResult:
        """Run patch_state for every component and timestep.

        Args:
            clean_cache: Cache from the clean run.
            corrupted_cache: Cache from the corrupted run.
            components: Components to patch.
            metric_fn: Maps an ActivationCache to a scalar.
            t_range: Timesteps to patch; defaults to all cached timesteps.
            clean_obs_seq: Clean observations.
            clean_action_seq: Clean actions.
            parallel: Unused; kept for compatibility.
            corrupted_obs_seq: Corrupted observations, to denoise.
            corrupted_action_seq: Corrupted actions; defaults to the clean actions.

        Returns:
            PatchingSweepResult.
        """
        if t_range is None:
            t_range = list(clean_cache.timesteps)

        results: Dict[Tuple[str, int], PatchResult] = {}
        progress = tqdm(total=len(components) * len(t_range), desc="Patching sweep") if tqdm else None

        for comp in components:
            for t in t_range:
                results[(comp, t)] = self.patch_state(
                    clean_cache,
                    corrupted_cache,
                    comp,
                    t,
                    metric_fn,
                    clean_obs_seq,
                    clean_action_seq,
                    corrupted_obs_seq=corrupted_obs_seq,
                    corrupted_action_seq=corrupted_action_seq,
                )
                if progress:
                    progress.update(1)

        if progress:
            progress.close()

        return PatchingSweepResult(
            results=results,
            components=components,
            timesteps=t_range,
            device=self.device,
        )


def _recovery(clean: float, corrupted: float, patched: float) -> float:
    if clean == corrupted:
        return float("nan")
    return (patched - corrupted) / (clean - corrupted)


class CorruptedCacheFactory:
    """Factory for creating corrupted activation caches."""

    @staticmethod
    def random_noise(
        cache: ActivationCache,
        noise_level: float = 0.5,
        components: Optional[List[str]] = None,
    ) -> ActivationCache:
        """Add random noise to activations.

        Args:
            cache: Source cache.
            noise_level: Standard deviation of noise.
            components: Optional list of components to corrupt.

        Returns:
            Corrupted cache.
        """
        device = cache.device if hasattr(cache, "device") else _get_device()
        corrupted = ActivationCache()

        target_components = components or list(cache.component_names)

        for key in target_components:
            for t in cache.timesteps:
                original = cache.get(key, t, None)
                if original is not None:
                    original = original.to(device)
                    noise = torch.randn_like(original) * noise_level
                    corrupted[key, t] = original + noise

        return corrupted

    @staticmethod
    def zero_out(
        cache: ActivationCache,
        components: Optional[List[str]] = None,
    ) -> ActivationCache:
        """Zero out activations.

        Args:
            cache: Source cache.
            components: Optional list of components.

        Returns:
            Corrupted cache.
        """
        device = cache.device if hasattr(cache, "device") else _get_device()
        corrupted = ActivationCache()

        target_components = components or list(cache.component_names)

        for key in target_components:
            for t in cache.timesteps:
                original = cache.get(key, t, None)
                if original is not None:
                    original = original.to(device)
                    corrupted[key, t] = torch.zeros_like(original)

        return corrupted

    @staticmethod
    def shuffle(
        cache: ActivationCache,
        components: Optional[List[str]] = None,
    ) -> ActivationCache:
        """Shuffle each component's activations across timesteps.

        Args:
            cache: Source cache.
            components: Optional list of components.

        Returns:
            Corrupted cache.
        """
        corrupted = ActivationCache()

        for key in components or list(cache.component_names):
            timesteps = [t for t in cache.timesteps if cache.get(key, t) is not None]
            if not timesteps:
                continue
            permutation = torch.randperm(len(timesteps)).tolist()
            for t, source in zip(timesteps, permutation):
                corrupted[key, t] = cache[key, timesteps[source]]

        return corrupted


def default_metric_fn(cache: ActivationCache) -> float:
    """Default metric: sum of z_posterior L2 norm.

    Args:
        cache: Activation cache.

    Returns:
        Metric value.
    """
    if "z_posterior" in cache.component_names and 0 in cache.timesteps:
        z = cache["z_posterior", 0]
        return float(z.abs().sum().item())
    return 0.0


def reconstruction_metric_fn(cache: ActivationCache) -> float:
    """Metric based on reconstruction quality.

    Args:
        cache: Activation cache.

    Returns:
        Negative MSE (higher is better).
    """
    if "reconstruction" in cache.component_names and "observation" in cache.component_names:
        recon = cache["reconstruction", 0]
        obs = cache["observation", 0]
        mse = float(((recon - obs) ** 2).mean().item())
        return -mse
    return 0.0


def surprise_metric_fn(cache: ActivationCache) -> float:
    """Metric based on surprise (KL divergence).

    Args:
        cache: Activation cache.

    Returns:
        Sum of KL divergences.
    """
    if "kl" in cache.component_names:
        return float(cache["kl", 0].sum().item() if cache["kl", 0].numel() > 0 else 0.0)
    return 0.0
