"""Parallelization primitives for large-scale analysis."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator, TYPE_CHECKING
import numpy as np
import torch
from tqdm import tqdm

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.lazy_trajectory import TrajectoryDataset, LatentTrajectoryLite

if TYPE_CHECKING:
    from world_model_lens.hooked_world_model import HookedWorldModel
    from world_model_lens.patching.patcher import PatchingSweepResult, PatchResult
    from world_model_lens.patching.patcher import TemporalPatcher


def run_with_cache_parallel(
    dataset: TrajectoryDataset,
    wm_factory: Callable[[], "HookedWorldModel"],
    n_workers: int = 4,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    names_filter: Optional[List[str]] = None,
    use_gpu: bool = True,
) -> List[Tuple[ActivationCache, LatentTrajectoryLite]]:
    """Run analysis in parallel across trajectory dataset.

    Shards trajectories across CPU/GPU workers and collects activation caches.

    Args:
        dataset: TrajectoryDataset with trajectories to process.
        wm_factory: Factory function to create HookedWorldModel per worker.
        n_workers: Number of parallel workers.
        batch_size: Batch size per worker.
        device: Target device (None = auto).
        names_filter: Components to cache.
        use_gpu: Whether to use GPU workers (requires CUDA).

    Returns:
        List of (ActivationCache, trajectory) tuples.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    if n_workers == 1:
        return _run_single_worker(dataset, wm_factory, device, names_filter)

    results = []
    chunks = np.array_split(range(len(dataset)), n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            future = executor.submit(
                _worker_process_chunk,
                list(chunk),
                wm_factory,
                device,
                names_filter,
                dataset,
            )
            futures.append(future)

        for future in tqdm(futures, desc="Parallel processing"):
            chunk_results = future.result()
            results.extend(chunk_results)

    return results


def _worker_process_chunk(
    indices: List[int],
    wm_factory: Callable[[], "HookedWorldModel"],
    device: torch.device,
    names_filter: Optional[List[str]],
    dataset: TrajectoryDataset,
) -> List[Tuple[ActivationCache, int]]:
    """Process a chunk of trajectories in a worker."""
    wm = wm_factory()
    wm = wm.to(device)
    results = []

    for idx in indices:
        traj_data = dataset[idx]
        obs = traj_data["observation"]
        actions = traj_data.get("actions")

        traj, cache = wm.run_with_cache(
            obs.to(device),
            actions.to(device) if actions is not None else None,
            names_filter=names_filter,
        )
        results.append((cache, idx))

    return results


def _run_single_worker(
    dataset: TrajectoryDataset,
    wm_factory: Callable[[], "HookedWorldModel"],
    device: torch.device,
    names_filter: Optional[List[str]],
) -> List[Tuple[ActivationCache, LatentTrajectoryLite]]:
    """Process all trajectories in single thread."""
    wm = wm_factory()
    wm = wm.to(device)
    results = []

    for traj in tqdm(dataset.trajectories, desc="Processing trajectories"):
        obs = traj.h_sequence.to(device)
        actions = traj.actions.to(device) if traj.actions is not None else None

        traj_out, cache = wm.run_with_cache(
            obs,
            actions,
            names_filter=names_filter,
        )
        results.append((cache, traj))

    return results


@dataclass
class PatchingSweepConfig:
    """Configuration for parallel patching sweeps."""

    components: List[str]
    timesteps: List[int]
    metric_fn: Callable[[torch.Tensor], float]
    clean_cache: Optional[ActivationCache] = None
    corrupted_cache: Optional[ActivationCache] = None
    obs_seq: Optional[torch.Tensor] = None
    action_seq: Optional[torch.Tensor] = None
    n_workers: int = 4
    chunk_size: int = 10


def patching_sweep_parallel(
    patcher: "TemporalPatcher",
    config: PatchingSweepConfig,
) -> "PatchingSweepResult":
    """Run patching sweep in parallel over components and timesteps.

    Args:
        patcher: Initialized TemporalPatcher.
        config: Sweep configuration.

    Returns:
        PatchingSweepResult with all patch results.
    """
    from world_model_lens.patching.patcher import PatchingSweepResult

    tasks = []
    for comp in config.components:
        for t in config.timesteps:
            tasks.append((comp, t))

    chunks = np.array_split(tasks, config.n_workers)

    with ThreadPoolExecutor(max_workers=config.n_workers) as executor:
        futures = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            future = executor.submit(
                _worker_patching_chunk,
                patcher,
                chunk,
                config,
            )
            futures.append(future)

        results = {}
        for future in futures:
            chunk_results = future.result()
            results.update(chunk_results)

    return PatchingSweepResult(
        results=results,
        components=config.components,
        timesteps=config.timesteps,
    )


def _worker_patching_chunk(
    patcher: "TemporalPatcher",
    tasks: List[Tuple[str, int]],
    config: PatchingSweepConfig,
) -> Dict[Tuple[str, int], "PatchResult"]:
    """Process a chunk of patching tasks."""
    results = {}
    # Use a static description for the progress bar; per-item names aren't available
    for comp, t in tqdm(tasks, desc="Patching chunk"):
        result = patcher.patch_state(
            config.clean_cache,
            config.corrupted_cache,
            comp,
            t,
            config.metric_fn,
            config.obs_seq,
            config.action_seq,
        )
        results[(comp, t)] = result
    return results


@dataclass
class ParallelConfig:
    """Configuration for parallel world model analysis."""

    n_workers: int = 4
    batch_size: int = 8
    use_gpu: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        if self.mixed_precision and self.dtype is None:
            self.dtype = torch.float16


def create_parallel_loader(
    dataset: TrajectoryDataset,
    config: ParallelConfig,
) -> torch.utils.data.DataLoader:
    """Create DataLoader optimized for parallel world model analysis.

    Args:
        dataset: TrajectoryDataset.
        config: Parallel configuration.

    Returns:
        Configured DataLoader.
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.n_workers,
        pin_memory=config.pin_memory and config.use_gpu,
        prefetch_factor=config.prefetch_factor,
    )
    return loader


class GradientCheckpointedModel:
    """Wrapper that adds gradient checkpointing to a world model.

    Useful for saliency computation and SAE training where backward
    passes are needed but memory is limited.
    """

    def __init__(
        self,
        wm: "HookedWorldModel",
        checkpoint_layers: Optional[List[str]] = None,
    ):
        self.wm = wm
        self.checkpoint_layers = checkpoint_layers or ["encoder", "dynamics"]

    def run_with_cache(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        names_filter: Optional[List[str]] = None,
    ) -> Tuple[Any, ActivationCache]:
        """Run with gradient checkpointing enabled."""

        def checkpointee(module, *args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                module,
                *args,
                **kwargs,
                use_reentrant=False,
            )

        return self.wm.run_with_cache(
            observations,
            actions,
            names_filter,
        )

    def compute_saliency(
        self,
        observations: torch.Tensor,
        target: torch.Tensor,
        method: str = "grad",
    ) -> Dict[str, torch.Tensor]:
        """Compute saliency with gradient checkpointing."""
        self.wm.adapter.train()
        observations.requires_grad_(True)

        if method == "grad":
            output = self.wm.run_with_cache(observations)[0]
            loss = (output - target).pow(2).mean()
            loss.backward()
            saliency = observations.grad.abs()
        elif method == "integrated_grad":
            saliency = self._integrated_gradients(observations, target)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.wm.adapter.eval()
        return saliency

    def _integrated_gradients(
        self,
        observations: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Compute integrated gradients."""
        baseline = torch.zeros_like(observations)
        scaled_inputs = [
            baseline + (i / n_steps) * (observations - baseline) for i in range(n_steps + 1)
        ]

        grads = []
        for inp in scaled_inputs:
            inp.requires_grad_(True)
            out = self.wm.run_with_cache(inp.unsqueeze(0))[0]
            loss = (out - target).pow(2).mean()
            grads.append(inp.grad)

        avg_grad = torch.stack(grads).mean(dim=0)
        saliency = (observations - baseline) * avg_grad
        return saliency.abs().sum(dim=0)


class MixedPrecisionContext:
    """Context for mixed-precision execution."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        self.scaler = torch.cuda.amp.GradScaler()
        return self

    def __exit__(self, *args):
        pass

    def run(
        self,
        fn: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run function with automatic mixed precision."""
        with torch.cuda.amp.autocast(dtype=self.dtype):
            return fn(*args, **kwargs)


def run_with_mixed_precision(
    fn: Callable,
    *args,
    dtype: torch.dtype = torch.float16,
    **kwargs,
) -> Any:
    """Run function with mixed precision.

    Args:
        fn: Function to run.
        *args: Positional arguments.
        dtype: Precision dtype (float16, bfloat16).
        **kwargs: Keyword arguments.

    Returns:
        Function output.
    """
    with torch.cuda.amp.autocast(dtype=dtype):
        return fn(*args, **kwargs)


def shard_across_gpus(
    model_fn: Callable[[], "HookedWorldModel"],
    dataset: TrajectoryDataset,
    n_gpus: int,
) -> Iterator[Tuple[int, "HookedWorldModel", List[int]]]:
    """Shard dataset across available GPUs.

    Args:
        model_fn: Factory for world model.
        dataset: TrajectoryDataset to shard.
        n_gpus: Number of GPUs to use.

    Yields:
        Tuples of (gpu_id, model, trajectory_indices).
    """
    indices = list(range(len(dataset)))
    chunks = np.array_split(indices, n_gpus)

    for gpu_id, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        device = torch.device(f"cuda:{gpu_id}")
        model = model_fn().to(device)
        yield gpu_id, model, list(chunk)
