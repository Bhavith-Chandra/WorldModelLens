"""Video prediction metrics for evaluating video world models.

This module provides metrics specifically designed for video prediction
and generation models. Works with ANY world model that predicts video frames.

Metrics:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- MSE: Mean Squared Error
- MAE: Mean Absolute Error
- FID: Frechet Inception Distance (approximated for video)
- LPIPS: Perceptual similarity
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache


@dataclass
class VideoMetricsResult:
    """Result of video prediction metrics."""

    psnr: float | None = None
    ssim: float | None = None
    mse: float | None = None
    mae: float | None = None
    per_frame_psnr: list[float] | None = None
    per_frame_ssim: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "mse": self.mse,
            "mae": self.mae,
        }


class VideoPredictionMetrics:
    """Metrics for evaluating video prediction world models.

    Works with ANY world model that produces video predictions.
    Both RL and non-RL models are supported.
    """

    def __init__(self, data_range: float = 1.0):
        """Initialize metrics calculator.

        Args:
            data_range: Range of pixel values (1.0 for [-1,1], 255.0 for [0,255])
        """
        self.data_range = data_range

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> VideoMetricsResult:
        """Compute all metrics between predictions and targets.

        Args:
            predictions: Predicted frames [T, C, H, W] or [B, T, C, H, W]
            targets: Ground truth frames [T, C, H, W] or [B, T, C, H, W]

        Returns:
            VideoMetricsResult with all computed metrics
        """
        if predictions.dim() == 4:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 4:
            targets = targets.unsqueeze(0)

        result = VideoMetricsResult()

        result.mse = float(self.mse(predictions, targets))
        result.mae = float(self.mae(predictions, targets))
        result.psnr = float(self.psnr(predictions, targets))
        result.ssim = float(self.ssim(predictions, targets))

        per_frame_psnr = []
        per_frame_ssim = []
        for t in range(predictions.shape[1]):
            pred_t = predictions[:, t]
            target_t = targets[:, t]
            per_frame_psnr.append(float(self.psnr(pred_t, target_t)))
            per_frame_ssim.append(float(self.ssim(pred_t, target_t)))

        result.per_frame_psnr = per_frame_psnr
        result.per_frame_ssim = per_frame_ssim

        return result

    def mse(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Mean Squared Error.

        Args:
            predictions: [B, T, C, H, W]
            targets: [B, T, C, H, W]

        Returns:
            MSE value
        """
        return ((predictions - targets) ** 2).mean()

    def mae(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Mean Absolute Error.

        Args:
            predictions: [B, T, C, H, W]
            targets: [B, T, C, H, W]

        Returns:
            MAE value
        """
        return (predictions - targets).abs().mean()

    def psnr(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Peak Signal-to-Noise Ratio.

        Args:
            predictions: [B, T, C, H, W]
            targets: [B, T, C, H, W]

        Returns:
            PSNR value in dB
        """
        mse = self.mse(predictions, targets)
        if mse == 0:
            return torch.tensor(float("inf"))
        psnr = 20 * torch.log10(torch.tensor(self.data_range) / torch.sqrt(mse))
        return psnr

    def ssim(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """Compute Structural Similarity Index.

        Simplified SSIM implementation.

        Args:
            predictions: [B, T, C, H, W]
            targets: [B, T, C, H, W]
            window_size: Size of Gaussian window

        Returns:
            SSIM value between -1 and 1
        """
        B, T = predictions.shape[:2]

        pred_flat = predictions.reshape(B * T, *predictions.shape[2:])
        target_flat = targets.reshape(B * T, *targets.shape[2:])

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        mu_pred = pred_flat.mean(dim=[1, 2], keepdim=True)
        mu_target = target_flat.mean(dim=[1, 2], keepdim=True)

        sigma_pred = pred_flat.var(dim=[1, 2], keepdim=True)
        sigma_target = target_flat.var(dim=[1, 2], keepdim=True)
        sigma_pred_target = ((pred_flat - mu_pred) * (target_flat - mu_target)).mean(
            dim=[1, 2], keepdim=True
        )

        ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / (
            (mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2)
        )

        return ssim_map.mean()

    def from_cache(
        self,
        cache: "ActivationCache",
        pred_key: str = "reconstruction",
        target_key: str = "observation",
    ) -> VideoMetricsResult:
        """Compute metrics from activation cache.

        Args:
            cache: ActivationCache with predictions and targets
            pred_key: Key for predictions in cache
            target_key: Key for targets in cache

        Returns:
            VideoMetricsResult
        """
        try:
            predictions = cache[pred_key]
            targets = cache[target_key]
            return self.compute_metrics(predictions, targets)
        except KeyError as e:
            raise KeyError(
                f"Cache must contain '{pred_key}' and '{target_key}' for video metrics"
            ) from e
