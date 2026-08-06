"""Activation patching for causal analysis."""

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
        """Get top-k patches by recovery rate."""
        sorted_results = sorted(self.results.values(), key=lambda x: x.recovery_rate, reverse=True)
        return sorted_results[:k]

    def get_component_rankings(self) -> Dict[str, float]:
        """Get average recovery rate per component."""
        component_scores: Dict[str, List[float]] = {}
        for (comp, _), result in self.results.items():
            if comp not in component_scores:
                component_scores[comp] = []
            component_scores[comp].append(result.recovery_rate)
        return {comp: sum(scores) / len(scores) for comp, scores in component_scores.items()}

    def get_timestep_rankings(self) -> Dict[int, float]:
        """Get average recovery rate per timestep."""
        timestep_scores: Dict[int, List[float]] = {}
        for (_, t), result in self.results.items():
            if t not in timestep_scores:
                timestep_scores[t] = []
            timestep_scores[t].append(result.recovery_rate)
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
        component: str,
        metric_fn: Callable[[ActivationCache], float],
    ) -> float:
        """Compute metric from activation cache.

        Args:
            cache: Activation cache.
            metric_fn: Function to compute metric.

        Returns:
            Metric value.
        """
        return metric_fn(cache[component]) if component in list(cache.keys()) else 0.0

    def compute_metric_from_trajectory(
        self,
        traj: WorldTrajectory,
        metric_fn: Callable[[WorldTrajectory], float],
    ) -> float:
        """Compute metric from trajectory.

        Args:
            traj: World trajectory.
            metric_fn: Function to compute metric.

        Returns:
            Metric value.
        """
        return metric_fn(traj)

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
    ) -> PatchResult:
        """Patch a single component at a timestep.

        Args:
            clean_cache: Clean activation cache.
            corrupted_cache: Corrupted activation cache.
            patch_component: Component to patch.
            patch_at_timestep: Timestep to patch.
            metric_fn: Function computing metric from cache.
            clean_obs_seq: Clean observations (for patched run).
            clean_action_seq: Clean actions (for patched run).
            patch_value: Optional specific value to patch with.

        Returns:
            PatchResult with recovery rate.
        """
        clean_metric = self.compute_metric_from_cache(clean_cache, patch_component, metric_fn)
        corrupted_metric = self.compute_metric_from_cache(
            corrupted_cache, patch_component, metric_fn
        )

        if patch_value is None:
            patch_value = self._ensure_device(
                clean_cache[patch_component, patch_at_timestep].clone()
            )

        original_value: Optional[Tensor] = None
        patched_value: Optional[Tensor] = None

        def patch_hook(tensor: Tensor, ctx: HookContext) -> Tensor:
            nonlocal original_value, patched_value
            if ctx.component == patch_component and ctx.timestep == patch_at_timestep:
                original_value = tensor.detach().clone()
                patched_value = patch_value.detach().clone()
                return patched_value
            return tensor

        hook = HookPoint(
            name=patch_component,
            fn=patch_hook,
            stage="post",
            timestep=patch_at_timestep,
        )
        self.wm.add_hook(hook)

        try:
            if clean_obs_seq is not None and clean_action_seq is not None:
                obs_seq = self._ensure_device(clean_obs_seq)
                action_seq = self._ensure_device(clean_action_seq)
                patched_traj, _, patched_cache = self.wm.run_with_cache(obs_seq, action_seq)
                patched_metric = self.compute_metric_from_cache(
                    patched_cache, patch_component, metric_fn
                )
            else:
                raise ValueError("clean_obs_seq and clean_action_seq required")

        except Exception as e:
            raise RuntimeError(f"Patching failed: {e}")
        finally:
            self.wm.clear_hooks()

        if abs(clean_metric - corrupted_metric) < 1e-8:
            recovery = 1.0
        else:
            recovery = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
        recovery = max(0.0, min(1.0, recovery))

        return PatchResult(
            metric_clean=clean_metric,
            metric_corrupted=corrupted_metric,
            metric_patched=patched_metric,
            recovery_rate=recovery,
            component=patch_component,
            timestep=patch_at_timestep,
            patch_mode="single",
            original_value=original_value,
            patched_value=patched_value,
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
        """Patch with automatic corruption generation.

        Args:
            clean_obs_seq: Clean observations.
            clean_action_seq: Clean actions.
            corruption_fn: Function to corrupt activations.
            patch_component: Component to patch.
            patch_at_timestep: Timestep to patch.
            metric_fn: Metric function from trajectory.

        Returns:
            PatchResult.
        """
        clean_obs = self._ensure_device(clean_obs_seq)
        clean_actions = self._ensure_device(clean_action_seq)

        clean_traj, _, clean_cache = self.wm.run_with_cache(clean_obs, clean_actions)
        clean_metric = self.compute_metric_from_trajectory(clean_traj, metric_fn)

        corrupted_cache = ActivationCache()
        for key in clean_cache.component_names:
            for t in clean_cache.timesteps:
                original = clean_cache.get(key, t, None)
                if original is not None:
                    original = self._ensure_device(original)
                    corrupted = corruption_fn(original)
                    corrupted_cache[key, t] = corrupted

        def create_corrupted_hook(comp: str, t: int, corr_val: Tensor):
            def hook(tensor: Tensor, ctx: HookContext) -> Tensor:
                if ctx.component == comp and ctx.timestep == t:
                    return corr_val
                return tensor

            return hook

        hook = HookPoint(
            name=patch_component,
            fn=create_corrupted_hook(
                patch_component,
                patch_at_timestep,
                corrupted_cache[patch_component, patch_at_timestep],
            ),
            stage="post",
            timestep=patch_at_timestep,
        )
        self.wm.add_hook(hook)

        try:
            corrupted_traj, _, _ = self.wm.run_with_cache(clean_obs, clean_actions)
            corrupted_metric = self.compute_metric_from_trajectory(corrupted_traj, metric_fn)
        finally:
            self.wm.clear_hooks()

        patch_value = self._ensure_device(clean_cache[patch_component, patch_at_timestep].clone())
        hook2 = HookPoint(
            name=patch_component,
            fn=lambda t, c: patch_value,
            stage="post",
            timestep=patch_at_timestep,
        )
        self.wm.add_hook(hook2)

        try:
            patched_traj, _, _ = self.wm.run_with_cache(clean_obs, clean_actions)
            patched_metric = self.compute_metric_from_trajectory(patched_traj, metric_fn)
        finally:
            self.wm.clear_hooks()

        if abs(clean_metric - corrupted_metric) < 1e-8:
            recovery = 1.0
        else:
            recovery = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
        recovery = max(0.0, min(1.0, recovery))

        return PatchResult(
            metric_clean=clean_metric,
            metric_corrupted=corrupted_metric,
            metric_patched=patched_metric,
            recovery_rate=recovery,
            component=patch_component,
            timestep=patch_at_timestep,
            patch_mode="auto_corrupt",
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
    ) -> PatchingSweepResult:
        """Run full patching sweep across components and timesteps.

        Args:
            clean_cache: Clean activation cache.
            corrupted_cache: Corrupted activation cache.
            components: List of components to test.
            metric_fn: Metric function.
            t_range: Optional list of timesteps.
            clean_obs_seq: Clean observations for patched runs.
            clean_action_seq: Clean actions for patched runs.
            parallel: Whether to run in parallel.

        Returns:
            PatchingSweepResult.
        """
        if t_range is None:
            if clean_cache.timesteps:
                t_range = list(range(max(clean_cache.timesteps) + 1))
            else:
                t_range = list(range(10))

        results: Dict[Tuple[str, int], PatchResult] = {}

        total_experiments = len(components) * len(t_range)
        iterator = tqdm(total=total_experiments, desc="Patching sweep") if tqdm else None

        for comp in components:
            for t in t_range:
                result = self.patch_state(
                    clean_cache,
                    corrupted_cache,
                    comp,
                    t,
                    metric_fn,
                    clean_obs_seq,
                    clean_action_seq,
                )
                results[(comp, t)] = result
                if iterator:
                    iterator.update(1)

        if iterator:
            iterator.close()

        return PatchingSweepResult(
            results=results,
            components=components,
            timesteps=t_range,
            device=self.device,
        )


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
        """Shuffle activations across timesteps.

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
            all_tensors = []
            for t in cache.timesteps:
                val = cache.get(key, t, None)
                if val is not None:
                    all_tensors.append(val.to(device))

            if all_tensors:
                shuffled = torch.cat(all_tensors)[torch.randperm(len(all_tensors))]
                for i, t in enumerate(cache.timesteps):
                    if i < len(all_tensors):
                        start = i * all_tensors[0].shape[0]
                        end = start + all_tensors[0].shape[0]
                        corrupted[key, t] = shuffled[start:end]

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
