"""Experiment tracking and collaboration: WandB/MLflow integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from world_model_lens.core.world_trajectory import WorldTrajectory
import numpy as np
import torch

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.lazy_trajectory import TrajectoryDataset, LatentTrajectoryLite
from world_model_lens.probing.prober import ProbeResult, SweepResult
from world_model_lens.patching.patcher import PatchResult, PatchingSweepResult


@dataclass
class ReproConfig:
    """Reproducibility configuration.

    Captures all random seeds and parameters for exact reproduction.

    Example:
        config = ReproConfig(
            model_seed=42,
            env_seed=123,
            analysis_seed=456,
            model_params={"lr": 1e-4, "hidden_dim": 512},
            env_params={"max_steps": 1000},
        )
        config.log_to_wandb(project="my_project")
    """

    model_seed: int = 42
    env_seed: int = 0
    analysis_seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42
    python_hash_seed: int = 0

    model_params: Dict[str, Any] = field(default_factory=dict)
    env_params: Dict[str, Any] = field(default_factory=dict)
    analysis_params: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        self._apply_seeds()

    def _apply_seeds(self):
        """Apply all seeds."""
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
        os.environ["PYTHONHASHSEED"] = str(self.python_hash_seed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_seed": self.model_seed,
            "env_seed": self.env_seed,
            "analysis_seed": self.analysis_seed,
            "torch_seed": self.torch_seed,
            "numpy_seed": self.numpy_seed,
            "python_hash_seed": self.python_hash_seed,
            "model_params": self.model_params,
            "env_params": self.env_params,
            "analysis_params": self.analysis_params,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReproConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ExperimentTracker:
    """Base class for experiment tracking."""

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_artifact(self, name: str, path: str) -> None:
        pass

    def finish(self) -> None:
        pass


class WandBTracker(ExperimentTracker):
    """Weights & Biases integration for WorldModelLens.

    Example:
        tracker = WandBTracker(
            project="world_model_interp",
            entity="my_team",
            name="dreamer_atari_analysis",
        )
        tracker.log_params(config.to_dict())
        tracker.log_metrics({"accuracy": 0.95}, step=100)
        tracker.log_artifact("trajectory_cache", "cache.pt")
    """

    def __init__(
        self,
        project: str = "world_model_lens",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[ReproConfig] = None,
        mode: str = "online",
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config
        self.mode = mode
        self._run = None

    def _ensure_init(self):
        if self._run is None:
            try:
                import wandb
            except ImportError:
                raise ImportError("wandb required: pip install wandb")

            wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config.to_dict() if self.config else None,
                mode=self.mode,
            )
            self._run = wandb.run

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._ensure_init()
        import wandb

        wandb.log(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._ensure_init()
        import wandb

        wandb.config.update(params)

    def log_artifact(self, name: str, path: str) -> None:
        self._ensure_init()
        import wandb

        artifact = wandb.Artifact(name, type="dataset")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def log_trajectory(self, traj: "WorldTrajectory", name: str = "trajectory") -> None:
        """Log trajectory summary."""
        self._ensure_init()
        import wandb

        summary = {
            f"{name}_length": traj.length,
            f"{name}_mean_reward": traj.rewards_real.mean().item()
            if traj.rewards_real is not None
            else None,
        }
        wandb.log(summary)

    def log_activation_cache(
        self,
        cache: ActivationCache,
        name: str = "activation_cache",
    ) -> None:
        """Log activation cache summary."""
        self._ensure_init()
        import wandb

        summary = {
            f"{name}_n_components": len(cache.component_names),
            f"{name}_n_timesteps": len(cache.timesteps),
        }
        wandb.log(summary)

    def log_probe_result(self, result: ProbeResult, prefix: str = "probe") -> None:
        """Log probe result."""
        self._ensure_init()
        import wandb

        metrics = {
            f"{prefix}_accuracy": result.accuracy,
            f"{prefix}_r2": result.r2 or 0.0,
            f"{prefix}_n_samples": result.training_samples + result.test_samples,
        }
        wandb.log(metrics)

    def log_patching_result(self, result: PatchResult, prefix: str = "patch") -> None:
        """Log patching result."""
        self._ensure_init()
        import wandb

        metrics = {
            f"{prefix}_recovery": result.recovery_rate,
            f"{prefix}_clean": result.metric_clean,
            f"{prefix}_patched": result.metric_patched,
        }
        wandb.log(metrics)

    def log_benchmark_result(self, result: Dict[str, Any]) -> None:
        """Log benchmark results."""
        self._ensure_init()
        import wandb

        for name, data in result.items():
            if isinstance(data, dict) and "score" in data:
                wandb.log({f"benchmark_{name}": data["score"]})

    def finish(self) -> None:
        if self._run is not None:
            import wandb

            wandb.finish()


class MLflowTracker(ExperimentTracker):
    """MLflow integration for WorldModelLens.

    Example:
        tracker = MLflowTracker(
            experiment_name="world_model_interp",
            tracking_uri="http://localhost:5000",
        )
        tracker.log_params(config.to_dict())
        tracker.log_metrics({"accuracy": 0.95}, step=100)
    """

    def __init__(
        self,
        experiment_name: str = "world_model_lens",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self._run = None

    def _ensure_init(self):
        if self._run is None:
            try:
                import mlflow
            except ImportError:
                raise ImportError("mlflow required: pip install mlflow")

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name)
            self._mlflow = mlflow

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._ensure_init()
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._ensure_init()
        self._mlflow.log_params(params)

    def log_artifact(self, name: str, path: str) -> None:
        self._ensure_init()
        self._mlflow.log_artifact(path, artifact_path=name)

    def log_model(self, model: torch.nn.Module, name: str) -> None:
        """Log PyTorch model."""
        self._ensure_init()
        self._mlflow.pytorch.log_model(model, name)

    def finish(self) -> None:
        if self._run is not None:
            self._mlflow.end_run()


def auto_log_to_wandb(
    project: str,
    entity: Optional[str] = None,
    config: Optional[ReproConfig] = None,
    mode: str = "online",
) -> WandBTracker:
    """Create WandB tracker with auto-logging for common artifacts."""
    return WandBTracker(project=project, entity=entity, config=config, mode=mode)


def auto_log_to_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    config: Optional[ReproConfig] = None,
) -> MLflowTracker:
    """Create MLflow tracker with auto-logging."""
    return MLflowTracker(experiment_name=experiment_name, tracking_uri=tracking_uri)


class CompositeTracker(ExperimentTracker):
    """Log to multiple backends simultaneously."""

    def __init__(self, trackers: List[ExperimentTracker]):
        self.trackers = trackers

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def log_params(self, params: Dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_artifact(self, name: str, path: str) -> None:
        for tracker in self.trackers:
            tracker.log_artifact(name, path)

    def finish(self) -> None:
        for tracker in self.trackers:
            tracker.finish()


def create_tracker(
    backend: str = "wandb",
    **kwargs,
) -> ExperimentTracker:
    """Factory for creating experiment trackers."""
    if backend == "wandb":
        return WandBTracker(**kwargs)
    elif backend == "mlflow":
        return MLflowTracker(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
