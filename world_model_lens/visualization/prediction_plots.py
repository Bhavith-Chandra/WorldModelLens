"""Prediction vs Ground Truth Visualization.

Compare model predictions to actual observations.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class PredictionComparison:
    """Comparison between prediction and ground truth."""

    predictions: List[torch.Tensor]
    targets: List[torch.Tensor]
    errors: List[float]
    mean_error: float
    max_error: float
    timestep_errors: np.ndarray


class PredictionVisualizer:
    """Visualize predictions vs ground truth.

    Example:
        viz = PredictionVisualizer(world_model)

        # Get predictions and ground truth
        obs = torch.randn(10, 3, 64, 64)
        traj, cache = world_model.run_with_cache(obs)

        # Compare reconstructions
        comp = viz.compare_reconstructions(traj, cache, observations=obs)

        # Error heatmap
        errors = viz.error_heatmap(traj, cache)

        # Per-frame comparison
        frames = viz.plot_frame_comparison(traj, cache, frame_idx=5)
    """

    def __init__(self, world_model: Any):
        """Initialize visualizer.

        Args:
            world_model: HookedWorldModel instance
        """
        self.wm = world_model

    def compare_reconstructions(
        self,
        trajectory: Any,
        cache: Any,
        observations: Optional[torch.Tensor] = None,
    ) -> PredictionComparison:
        """Compare reconstructions to observations.

        Args:
            trajectory: WorldTrajectory
            cache: ActivationCache
            observations: Original observations (if available)

        Returns:
            PredictionComparison with errors
        """
        predictions = []
        targets = []
        errors = []

        for t in range(len(trajectory.states)):
            recon = cache.get("reconstruction", t)

            if recon is None:
                continue

            if observations is not None and t < len(observations):
                target = observations[t]
            else:
                target = recon  # Self-reconstruction

            pred_flat = recon.flatten(start_dim=1) if recon.dim() > 2 else recon
            tgt_flat = target.flatten(start_dim=1) if target.dim() > 2 else target

            error = torch.nn.functional.mse_loss(pred_flat, tgt_flat).item()

            predictions.append(recon)
            targets.append(target)
            errors.append(error)

        timestep_errors = np.array(errors) if errors else np.array([0.0])

        return PredictionComparison(
            predictions=predictions,
            targets=targets,
            errors=errors,
            mean_error=np.mean(errors) if errors else 0.0,
            max_error=np.max(errors) if errors else 0.0,
            timestep_errors=timestep_errors,
        )

    def error_heatmap(
        self,
        trajectory: Any,
        cache: Any,
        observations: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute per-pixel error heatmap.

        Args:
            trajectory: WorldTrajectory
            cache: ActivationCache
            observations: Original observations

        Returns:
            2D array of per-pixel errors
        """
        errors = []

        for t in range(len(trajectory.states)):
            recon = cache.get("reconstruction", t)

            if recon is None:
                continue

            if observations is not None and t < len(observations):
                target = observations[t]
            else:
                target = recon

            # Per-pixel squared error
            pixel_error = (recon - target) ** 2

            if recon.dim() == 4:  # [B, C, H, W]
                pixel_error = pixel_error.mean(dim=1)  # [B, H, W]

            errors.append(pixel_error.mean(dim=0))

        if errors:
            return torch.stack(errors).mean(dim=0).numpy()

        return np.zeros((64, 64))

    def plot_frame_comparison(
        self,
        trajectory: Any,
        cache: Any,
        frame_idx: int,
        observations: Optional[torch.Tensor] = None,
    ) -> dict:
        """Get frame for side-by-side comparison.

        Args:
            trajectory: WorldTrajectory
            cache: ActivationCache
            frame_idx: Which frame to compare
            observations: Original observations

        Returns:
            Dict with 'prediction', 'target', 'error', 'diff'
        """
        recon = cache.get("reconstruction", frame_idx)

        if recon is None:
            return {}

        if observations is not None and frame_idx < len(observations):
            target = observations[frame_idx]
        else:
            target = recon

        diff = (recon - target).abs()

        if recon.dim() == 4 and recon.shape[0] > 1:
            recon = recon[0]
            target = target[0] if target.dim() > 3 else target
            diff = diff[0]

        return {
            "prediction": recon.detach().cpu(),
            "target": target.detach().cpu(),
            "error": diff.detach().cpu(),
        }

    def reward_timeline(
        self,
        trajectory: Any,
    ) -> dict:
        """Get reward predictions over time.

        Args:
            trajectory: WorldTrajectory

        Returns:
            Dict with 'timesteps', 'predicted', 'ground_truth'
        """
        predicted = []
        ground_truth = []
        timesteps = []

        for t, state in enumerate(trajectory.states):
            # Real LatentState fields: reward_pred (model prediction) and
            # reward_real (ground truth from the environment).
            pred = state.reward_pred
            gt = state.reward_real

            if pred is not None:
                if isinstance(pred, torch.Tensor):
                    predicted.append(pred.item())
                else:
                    predicted.append(float(pred))
                timesteps.append(t)

            if gt is not None:
                if isinstance(gt, torch.Tensor):
                    ground_truth.append(gt.item())
                else:
                    ground_truth.append(float(gt))

        return {
            "timesteps": np.array(timesteps),
            "predicted": np.array(predicted) if predicted else np.array([]),
            "ground_truth": np.array(ground_truth) if ground_truth else np.array([]),
        }

    def latent_distribution(
        self,
        trajectory: Any,
        dim: Optional[int] = None,
    ) -> dict:
        """Get latent value distribution.

        Args:
            trajectory: WorldTrajectory
            dim: Specific dimension (None = all)

        Returns:
            Dict with distribution stats
        """
        latents = []

        for state in trajectory.states:
            # Use the correct LatentState API: z_posterior for the stochastic
            # latent.  flatten() handles the [n_cat, n_cls] shape.
            z = state.z_posterior.flatten()

            if dim is not None:
                val = z[dim].item() if z.shape[0] > dim else 0.0
                latents.append(val)
            else:
                latents.append(z.tolist())

        if dim is not None:
            values = np.array(latents)
        else:
            values = np.array([v for vs in latents for v in vs])

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "values": values,
        }

    def state_trajectory_3d(
        self,
        trajectory: Any,
        components: List[int] = [0, 1, 2],
    ) -> dict:
        """Get 3D trajectory for 3D plotting.

        Args:
            trajectory: WorldTrajectory
            components: Which dimensions to use

        Returns:
            Dict with x, y, z arrays
        """
        x = []
        y = []
        z = []

        for state in trajectory.states:
            state_vec = state.state

            if state_vec.dim() > 1:
                state_vec = state_vec.squeeze(0)

            n_dims = len(components)
            if state_vec.shape[-1] < n_dims:
                continue

            for i, comp in enumerate(components):
                val = state_vec[..., comp].item()
                if i == 0:
                    x.append(val)
                elif i == 1:
                    y.append(val)
                else:
                    z.append(val)

        return {
            "x": np.array(x),
            "y": np.array(y),
            "z": np.array(z),
        }
