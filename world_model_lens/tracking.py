"""Experiment Tracking Integration.

Provides hooks for experiment tracking systems (W&B, MLFlow, etc.)
"""

from typing import Any, Dict, Optional, Callable
import json
import logging
from pathlib import Path


class ExperimentTracker:
    """Base experiment tracker.

    Can be extended for W&B, MLFlow, TensorBoard, etc.
    """

    def __init__(self, name: str = "world_model_lens"):
        self.name = name
        self.experiment_name: Optional[str] = None
        self.run_id: Optional[str] = None
        self._metrics: Dict[str, list] = {}
        self._logged_artifacts: Dict[str, Any] = {}

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append({"value": value, "step": step})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dict of metric names to values
            step: Optional step number
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, name: str, data: Any) -> None:
        """Log an artifact (file, image, etc.).

        Args:
            name: Artifact name
            data: Artifact data
        """
        self._logged_artifacts[name] = data

    def get_metrics(self) -> Dict[str, list]:
        """Get all logged metrics."""
        return self._metrics.copy()

    def summary(self) -> Dict[str, float]:
        """Get summary of metrics (latest values)."""
        summary = {}
        for key, values in self._metrics.items():
            if values:
                summary[key] = values[-1]["value"]
        return summary


class WandBTracker(ExperimentTracker):
    """Weights & Biases tracker."""

    def __init__(self, project: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self._run = None

    def start_run(self, name: Optional[str] = None, **kwargs) -> None:
        """Start a new run."""
        try:
            import wandb

            self._run = wandb.init(project=self.project, name=name, **kwargs)
            self.experiment_name = name
            self.run_id = self._run.id if self._run else None
        except ImportError:
            print("[WARNING] wandb not installed. Using basic tracker.")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to W&B."""
        super().log_metric(key, value, step)
        if self._run:
            try:
                import wandb

                self._run.log({key: value, "step": step} if step else {key: value})
            except Exception:
                pass

    def log_artifact(self, name: str, data: Any) -> None:
        """Log artifact to W&B."""
        super().log_artifact(name, data)
        if self._run:
            try:
                import wandb

                if isinstance(data, (str, Path)):
                    artifact = wandb.Artifact(name, type="data")
                    artifact.add_file(str(data))
                    self._run.log_artifact(artifact)
                elif hasattr(data, "save"):
                    data.save(f"{name}.png")
                    artifact = wandb.Image(f"{name}.png")
                    self._run.log({name: artifact})
            except Exception:
                pass

    def end_run(self) -> None:
        """End the run."""
        if self._run:
            try:
                self._run.finish()
            except Exception:
                pass


class MLFlowTracker(ExperimentTracker):
    """MLFlow tracker."""

    def __init__(self, tracking_uri: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.tracking_uri = tracking_uri
        self._run = None

    def start_run(self, name: Optional[str] = None, **kwargs) -> None:
        """Start a new run."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            self._run = mlflow.start_run(run_name=name, **kwargs)
            self.experiment_name = name
            self.run_id = self._run.info.run_id if self._run else None
        except ImportError:
            print("[WARNING] mlflow not installed. Using basic tracker.")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to MLFlow."""
        super().log_metric(key, value, step)
        if self._run:
            try:
                import mlflow

                mlflow.log_metric(key, value, step or 0)
            except Exception:
                pass

    def log_artifact(self, name: str, data: Any) -> None:
        """Log artifact to MLFlow."""
        super().log_artifact(name, data)
        if self._run:
            try:
                import mlflow

                if isinstance(data, (str, Path)):
                    mlflow.log_artifact(str(data))
            except Exception:
                pass

    def end_run(self) -> None:
        """End the run."""
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass


def create_tracker(backend: str = "auto", **kwargs) -> ExperimentTracker:
    """Create an experiment tracker.

    Args:
        backend: "wandb", "mlflow", or "auto" (auto-detect)
        **kwargs: Additional arguments

    Returns:
        ExperimentTracker instance
    """
    if backend == "wandb":
        return WandBTracker(**kwargs)
    elif backend == "mlflow":
        return MLFlowTracker(**kwargs)
    elif backend == "auto":
        # Try W&B first, then MLFlow, then basic
        try:
            import wandb

            return WandBTracker(**kwargs)
        except ImportError:
            try:
                import mlflow

                return MLFlowTracker(**kwargs)
            except ImportError:
                return ExperimentTracker(**kwargs)
    else:
        return ExperimentTracker(**kwargs)


class Logger:
    """Internal logger for WorldModelLens."""

    def __init__(self, name: str = "world_model_lens", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)


# Global logger instance
_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """Get global logger."""
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger
