"""Weights & Biases integration for experiment tracking.

This module provides W&B integration for:
- Logging probing results
- Logging belief/surprise analysis
- Logging geometry analysis
- Logging safety reports
- Configuration tracking
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache
    from world_model_lens.core.world_trajectory import WorldTrajectory


class WandbLogger:
    """Logger for W&B integration.

    Example:
        logger = WandbLogger(project="world-model-analysis")

        # Log probing results
        logger.log_probe_results(probe_result)

        # Log surprise timeline
        logger.log_surprise(cache)

        # Log config
        logger.log_config(model_config)
    """

    def __init__(
        self,
        project: str = "world-model-lens",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        mode: str = "online",
    ):
        """Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity (team/organization)
            name: Run name
            config: Initial config to log
            tags: Tags for the run
            mode: 'online', 'offline', or 'disabled'
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Install with: pip install wandb")

        self.project = project
        self.entity = entity
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            mode=mode,
        )

    def log_config(self, config: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log configuration.

        Args:
            config: Configuration dictionary
            step: Optional step number
        """
        self.run.config.update(config, step=step)

    def log_probe_results(
        self,
        results: Any,
        step: Optional[int] = None,
    ) -> None:
        """Log probing results.

        Args:
            results: ProbeResult or SweepResult
            step: Optional step number
        """
        if hasattr(results, "accuracy"):
            self.run.log(
                {
                    f"probing/accuracy_{results.concept_name}": results.accuracy,
                    f"probing/r2_{results.concept_name}": results.r2,
                },
                step=step,
            )
        elif hasattr(results, "results"):
            for key, result in results.results.items():
                self.run.log(
                    {
                        f"probing/accuracy_{key}": result.accuracy,
                    },
                    step=step,
                )

    def log_surprise(
        self,
        cache: "ActivationCache",
        step: Optional[int] = None,
    ) -> None:
        """Log surprise/KL timeline.

        Args:
            cache: ActivationCache with surprise data
            step: Optional step number
        """
        surprise = cache.surprise()
        if surprise is not None:
            self.run.log(
                {
                    "surprise/mean": surprise.mean().item(),
                    "surprise/max": surprise.max().item(),
                    "surprise/std": surprise.std().item(),
                },
                step=step,
            )

    def log_geometry(
        self,
        results: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Log geometry analysis results.

        Args:
            results: Geometry analysis results dict
            step: Optional step number
        """
        logs = {}

        if "pca" in results:
            pca = results["pca"]
            if pca.get("pca_explained_variance") is not None:
                logs["geometry/pca_variance_top1"] = pca["pca_explained_variance"][0].item()

        if "trajectory" in results:
            traj = results["trajectory"]
            logs["geometry/mean_distance"] = traj.get("mean_trajectory_distance", 0)
            logs["geometry/temporal_coherence"] = traj.get("temporal_coherence", 0)

        if logs:
            self.run.log(logs, step=step)

    def log_safety_report(
        self,
        report: Any,
        step: Optional[int] = None,
    ) -> None:
        """Log safety audit report.

        Args:
            report: SafetyReport
            step: Optional step number
        """
        logs = {}

        if hasattr(report, "findings"):
            logs["safety/finding_count"] = len(report.findings)

        if hasattr(report, "overall_level"):
            logs["safety/level"] = (
                report.overall_level.value
                if hasattr(report.overall_level, "value")
                else report.overall_level
            )

        if hasattr(report, "metrics"):
            logs.update({f"safety/{k}": v for k, v in report.metrics.items()})

        if logs:
            self.run.log(logs, step=step)

    def log_trajectory_stats(
        self,
        trajectory: "WorldTrajectory",
        step: Optional[int] = None,
    ) -> None:
        """Log trajectory statistics.

        Args:
            trajectory: WorldTrajectory
            step: Optional step number
        """
        logs = {
            "trajectory/length": trajectory.length,
        }

        if trajectory.total_reward is not None:
            logs["trajectory/total_reward"] = trajectory.total_reward.item()

        if trajectory.mean_reward is not None:
            logs["trajectory/mean_reward"] = trajectory.mean_reward.item()

        self.run.log(logs, step=step)

    def log_activation_heatmap(
        self,
        cache: "ActivationCache",
        component: str,
        title: str = "Activation Heatmap",
        step: Optional[int] = None,
    ) -> None:
        """Log activation heatmap.

        Args:
            cache: ActivationCache
            component: Component name
            title: Plot title
            step: Optional step number
        """
        try:
            activations = cache[component]
            if activations.dim() == 2:
                self.run.log(
                    {
                        f"activations/{component}": wandb.Image(
                            np.array(activations.cpu().numpy().T)
                        )
                    },
                    step=step,
                )
        except (KeyError, Exception):
            pass

    def log_latent_trajectory(
        self,
        cache: "ActivationCache",
        components: List[str] = None,
        title: str = "Latent Trajectory PCA",
        step: Optional[int] = None,
    ) -> None:
        """Log 2D PCA projection of latent trajectory.

        Args:
            cache: ActivationCache
            components: Components to visualize
            title: Plot title
            step: Optional step number
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        components = components or ["z_posterior", "z"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for comp in components:
            try:
                data = cache[comp]
                if data.dim() == 2:
                    pca = PCA(n_components=2)
                    proj = pca.fit_transform(data.cpu().numpy())
                    ax.scatter(proj[:, 0], proj[:, 1], label=comp, alpha=0.5)
            except (KeyError, Exception):
                continue

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend()

        self.run.log({f"latent/{title}": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def finish(self) -> None:
        """Finish the W&B run."""
        self.run.finish()


def get_logger(
    project: str = "world-model-lens",
    **kwargs,
) -> Optional[WandbLogger]:
    """Get W&B logger if available.

    Returns None if wandb not installed.

    Args:
        project: Project name
        **kwargs: Additional arguments for WandbLogger

    Returns:
        WandbLogger or None
    """
    if not WANDB_AVAILABLE:
        return None
    return WandbLogger(project=project, **kwargs)
