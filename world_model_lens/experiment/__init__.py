"""Experiment tracking with wandb and HuggingFace Hub integration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
import json
import os
import torch

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.lazy_trajectory import TrajectoryDataset


@dataclass
class ExperimentTracker:
    """Base experiment tracker for logging experiments."""

    experiment_name: str
    config: Dict[str, Any] = field(default_factory=dict)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        raise NotImplementedError

    def log_parameters(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_artifact(self, name: str, artifact: Any) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


class NoOpTracker(ExperimentTracker):
    """No-op tracker that does nothing (for when no tracking is desired)."""

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        pass

    def log_parameters(self, params: Dict[str, Any]) -> None:
        pass

    def log_artifact(self, name: str, artifact: Any) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker.

    Requires: pip install wandb
    """

    def __init__(
        self,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        super().__init__(experiment_name, config or {})
        self.project = project or "world_model_lens"
        self.entity = entity
        self.tags = tags or []
        self.notes = notes or ""
        self._run = None

    def start(self) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb required: pip install wandb")

        wandb.init(
            name=self.experiment_name,
            project=self.project,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
        )
        self._run = wandb.run

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        if self._run is None:
            self.start()

        import wandb

        wandb.log({name: value}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._run is None:
            self.start()

        import wandb

        wandb.log(metrics, step=step)

    def log_parameters(self, params: Dict[str, Any]) -> None:
        if self._run is None:
            self.start()

        import wandb

        wandb.config.update(params)

    def log_artifact(
        self,
        name: str,
        artifact: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._run is None:
            self.start()

        import wandb

        if isinstance(artifact, dict):
            artifact = wandb.Artifact(name, type="dataset", metadata=metadata or {})
            for key, value in artifact.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                artifact.add(value, key)
        elif isinstance(artifact, str):
            if os.path.isfile(artifact):
                artifact = wandb.Artifact(name, type="file")
                artifact.add_file(artifact)
            else:
                artifact = wandb.Artifact(name, type="dataset", metadata=metadata)

        self._run.log_artifact(artifact)

    def log_table(self, name: str, data: Any, columns: Optional[List[str]] = None) -> None:
        if self._run is None:
            self.start()

        import wandb
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            table = wandb.Table(dataframe=data)
        elif isinstance(data, list):
            table = wandb.Table(data=data, columns=columns)
        else:
            table = wandb.Table(data=[[data]], columns=[name])

        wandb.log({name: table})

    def finish(self) -> None:
        if self._run is not None:
            import wandb

            wandb.finish()
            self._run = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.finish()


@dataclass
class HuggingFaceHub:
    """HuggingFace Hub integration for model and dataset sharing.

    Requires: pip install huggingface_hub
    """

    repo_id: str
    token: Optional[str] = None
    private: bool = False

    def push_trajectory_dataset(
        self,
        dataset: TrajectoryDataset,
        commit_message: str = "Add trajectory dataset",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Push TrajectoryDataset to HuggingFace Hub.

        Args:
            dataset: TrajectoryDataset to push.
            commit_message: Git commit message.
            metadata: Optional metadata dict.

        Returns:
            Repository URL.
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            raise ImportError("huggingface_hub required: pip install huggingface_hub")

        api = HfApi(token=self.token)

        try:
            create_repo(self.repo_id, repo_type="dataset", private=self.private, exist_ok=True)
        except Exception:
            pass

        cache_dir = f"/tmp/wml_hf_{self.repo_id.replace('/', '_')}"
        os.makedirs(cache_dir, exist_ok=True)

        dataset_path = os.path.join(cache_dir, "dataset.zarr")
        metadata_path = os.path.join(cache_dir, "metadata.json")

        dataset.to_disk(dataset_path, metadata_path=metadata_path)

        api.upload_folder(
            folder_path=cache_dir,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

        return f"https://huggingface.co/datasets/{self.repo_id}"

    @classmethod
    def from_disk(cls, repo_id: str, token: Optional[str] = None) -> TrajectoryDataset:
        """Load TrajectoryDataset from HuggingFace Hub.

        Args:
            repo_id: Repository ID (e.g., 'user/my-dataset').
            token: Optional HF token.

        Returns:
            Loaded TrajectoryDataset.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("huggingface_hub required: pip install huggingface_hub")

        cache_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        metadata_path = os.path.join(cache_dir, "metadata.json")
        dataset_path = cache_dir

        return TrajectoryDataset.from_disk(dataset_path, metadata_path=metadata_path)

    def push_model_card(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> None:
        """Create and push a model card to the repository.

        Args:
            metrics: Evaluation metrics.
            config: Model configuration.
            tags: Optional list of tags.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("huggingface_hub required: pip install huggingface_hub")

        api = HfApi(token=self.token)

        card_content = f"""---
license: apache-2.0
tags:
{chr(10).join(f"- {t}" for t in (tags or ["world-model", "interpretability"]))}
---

# World Model Analysis

## Metrics
{chr(10).join(f"- {k}: {v:.4f}" for k, v in metrics.items())}

## Configuration
```json
{json.dumps(config, indent=2)}
```
"""

        card_path = "/tmp/model_card.md"
        with open(card_path, "w") as f:
            f.write(card_content)

        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="model",
        )


@contextmanager
def wandb_init(
    experiment_name: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """Context manager for wandb experiment tracking.

    Example:
        with wandb_init("my_experiment", project="world_model_analysis") as tracker:
            tracker.log_metric("loss", 0.5)
            # ... run experiment
    """
    tracker = WandbTracker(
        experiment_name=experiment_name,
        project=project,
        entity=entity,
        config=config or {},
    )
    try:
        tracker.start()
        yield tracker
    finally:
        tracker.finish()


class ExperimentLogger:
    """Centralized experiment logger with multiple backend support."""

    def __init__(
        self,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        use_hf: bool = False,
        hf_repo_id: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        self.tracker: ExperimentTracker = NoOpTracker("default")

        if use_wandb and wandb_config:
            self.tracker = WandbTracker(**wandb_config)
        elif log_dir:
            self.tracker = LocalTracker(log_dir=log_dir)

        self.hf = HuggingFaceHub(repo_id=hf_repo_id) if use_hf and hf_repo_id else None
        self.log_dir = log_dir

    def start(self) -> None:
        if hasattr(self.tracker, "start"):
            self.tracker.start()

    def log(
        self,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        if metrics:
            for name, value in metrics.items():
                self.tracker.log_metric(name, value, step)

        if params:
            self.tracker.log_parameters(params)

    def log_cache(
        self,
        cache: ActivationCache,
        name: str = "activation_cache",
    ) -> None:
        if self.log_dir:
            cache_path = os.path.join(self.log_dir, f"{name}.pt")
            cache_data = {}
            for name_k in cache.component_names:
                for t in cache.timesteps:
                    val = cache.get(name_k, t, None)
                    cache_data[(name_k, t)] = val if isinstance(val, torch.Tensor) else None
            torch.save(cache_data, cache_path)
            self.tracker.log_artifact(name, cache_path)

    def log_trajectory(
        self,
        trajectory: LatentTrajectory,
        name: str = "trajectory",
    ) -> None:
        if self.log_dir:
            traj_path = os.path.join(self.log_dir, f"{name}.pt")
            torch.save(
                {
                    "h_sequence": trajectory.h_sequence,
                    "z_posterior": trajectory.z_posterior_sequence,
                    "metadata": trajectory.metadata,
                },
                traj_path,
            )
            self.tracker.log_artifact(name, traj_path)

    def push_dataset(
        self,
        dataset: TrajectoryDataset,
        commit_message: str = "Add dataset",
    ) -> Optional[str]:
        if self.hf:
            return self.hf.push_trajectory_dataset(dataset, commit_message)
        return None

    def finish(self) -> None:
        self.tracker.finish()


class LocalTracker(ExperimentTracker):
    """Local file-based experiment tracker."""

    def __init__(self, experiment_name: str, log_dir: str = "./logs"):
        super().__init__(experiment_name, {})
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")
        self.step = 0

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1

        record = {"step": self.step, "name": name, "value": value}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_parameters(self, params: Dict[str, Any]) -> None:
        params_path = os.path.join(self.log_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

    def log_artifact(self, name: str, artifact: Any) -> None:
        if isinstance(artifact, str) and os.path.isfile(artifact):
            import shutil

            dest = os.path.join(self.log_dir, name)
            shutil.copy(artifact, dest)

    def finish(self) -> None:
        pass


def create_logger(
    tracker_type: str = "local",
    **kwargs,
) -> ExperimentLogger:
    """Factory function to create experiment logger.

    Args:
        tracker_type: Type of tracker ('local', 'wandb', 'none').
        **kwargs: Additional arguments passed to tracker.

    Returns:
        Configured ExperimentLogger.
    """
    if tracker_type == "wandb":
        return ExperimentLogger(use_wandb=True, wandb_config=kwargs)
    elif tracker_type == "local":
        return ExperimentLogger(log_dir=kwargs.get("log_dir", "./logs"))
    else:
        return ExperimentLogger()
