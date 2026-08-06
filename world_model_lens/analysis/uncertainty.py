"""Uncertainty quantification for world model predictions.

This module provides tools to measure uncertainty in world model predictions.
Works with ANY world model type - both RL and non-RL.

Methods:
- Ensemble: Use multiple forward passes with different seeds/dropout
- Bayesian: Use dropout as Bayesian approximation
- Confidence: Measure prediction confidence from logits/probabilities
- Disagreement: Measure disagreement across multiple samples
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from world_model_lens import HookedWorldModel


@dataclass
class UncertaintyResult:
    """Result of uncertainty analysis."""

    mean_prediction: Optional[torch.Tensor] = None
    variance: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    epistemic_uncertainty: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None
    total_uncertainty: Optional[float] = None
    ood_scores: Optional[torch.Tensor] = None
    is_ood_fitted: bool = False


class UncertaintyQuantifier:
    """Quantify uncertainty in world model predictions.

    Works with ANY world model type. Can use multiple methods:
    - Ensemble-based
    - Dropout-based (Bayesian)
    - Confidence-based
    - Disagreement-based
    - Latent-based OOD detection (fit distribution on predictor latents)
    """

    def __init__(self, wm: Optional["HookedWorldModel"] = None):
        """Initialize quantifier.

        Args:
            wm: Optional HookedWorldModel instance
        """
        self.wm = wm

    def ensemble_uncertainty(
        self,
        predict_fn: Callable,
        inputs: torch.Tensor,
        n_samples: int = 10,
    ) -> UncertaintyResult:
        """Compute uncertainty using ensemble of predictions.

        Args:
            predict_fn: Function that takes inputs and returns predictions
            inputs: Input tensor
            n_samples: Number of ensemble samples

        Returns:
            UncertaintyResult with variance, entropy, etc.
        """
        predictions = []

        for _ in range(n_samples):
            pred = predict_fn(inputs)
            predictions.append(pred)

        predictions = torch.stack(predictions)

        result = UncertaintyResult()
        result.mean_prediction = predictions.mean(dim=0)
        result.variance = predictions.var(dim=0)

        if predictions.dim() > 1:
            probs = torch.softmax(predictions, dim=-1)
            result.entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=0)
            result.confidence = probs.max(dim=-1)[0].mean(dim=0)

        result.total_uncertainty = (
            result.variance.mean().item() if result.variance is not None else None
        )

        return result

    def dropout_uncertainty(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        n_samples: int = 10,
        dropout_rate: float = 0.1,
    ) -> UncertaintyResult:
        """Compute uncertainty using dropout as Bayesian approximation.

        Args:
            model: World model adapter
            inputs: Input tensor
            n_samples: Number of dropout samples
            dropout_rate: Dropout probability

        Returns:
            UncertaintyResult
        """
        original_training = model.training

        try:
            model.train()

            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = dropout_rate

            predictions = []
            for _ in range(n_samples):
                with torch.no_grad():
                    pred = model(inputs)
                predictions.append(pred)

            predictions = torch.stack(predictions)

            result = UncertaintyResult()
            result.mean_prediction = predictions.mean(dim=0)
            result.variance = predictions.var(dim=0)
            result.total_uncertainty = (
                result.variance.mean().item() if result.variance is not None else None
            )

        finally:
            model.train(original_training)

        return result

    def confidence_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute confidence from logits/probabilities.

        Args:
            logits: Prediction logits [..., n_classes]

        Returns:
            Confidence scores [...]
        """
        probs = torch.softmax(logits, dim=-1)
        confidence, _ = probs.max(dim=-1)
        return confidence

    def entropy_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entropy from logits/probabilities.

        Args:
            logits: Prediction logits [..., n_classes]

        Returns:
            Entropy values [...]
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -probs * torch.log(probs + 1e-8)
        return entropy.sum(dim=-1)

    def decompose_uncertainty(
        self,
        ensemble_predictions: torch.Tensor,
    ) -> dict[str, float]:
        """Decompose total uncertainty into epistemic and aleatoric.

        Epistemic: uncertainty from model ignorance (reduces with more data)
        Aleatoric: inherent uncertainty in the data (irreducible)

        Args:
            ensemble_predictions: [n_samples, ...] predictions from different models

        Returns:
            Dict with epistemic, aleatoric, and total uncertainty
        """
        if ensemble_predictions.dim() == 1:
            ensemble_predictions = ensemble_predictions.unsqueeze(0)

        mean_pred = ensemble_predictions.mean(dim=0)
        ensemble_predictions.var(dim=0)

        mean_prob = torch.softmax(mean_pred, dim=-1)
        entropy_mean = (-mean_prob * torch.log(mean_prob + 1e-8)).sum(dim=-1)

        probs = torch.softmax(ensemble_predictions, dim=-1)
        mean_entropy = (-probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=0)

        epistemic = entropy_mean - mean_entropy
        aleatoric = mean_entropy

        epistemic = epistemic.mean().item() if epistemic.numel() > 1 else epistemic.item()
        aleatoric = aleatoric.mean().item() if aleatoric.numel() > 1 else aleatoric.item()

        return {
            "epistemic_uncertainty": max(0, epistemic),
            "aleatoric_uncertainty": max(0, aleatoric),
            "total_uncertainty": max(0, epistemic + aleatoric),
        }

    def prediction_disagreement(
        self,
        predictions: torch.Tensor,
    ) -> float:
        """Compute disagreement between predictions.

        Args:
            predictions: [n_samples, ...] predictions

        Returns:
            Disagreement score
        """
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)

        mean_pred = predictions.mean(dim=0)

        if predictions.shape[-1] > 1:
            probs = torch.softmax(predictions, dim=-1)
            mean_probs = torch.softmax(mean_pred, dim=-1)
            disagreement = (probs - mean_probs).abs().mean()
        else:
            disagreement = predictions.var(dim=0).mean()

        return disagreement.item()

    def fit_latent_distribution(self, latents: list[torch.Tensor]) -> None:
        """Fit distribution on in-distribution predictor latents.

        Args:
            latents: List of latent tensors, each [num_patches, dim]
        """
        # Aggregate per latent: mean over patches
        aggregated = [latent.mean(dim=0) for latent in latents]  # list of [dim]
        latents_tensor = torch.stack(aggregated)  # [n_samples, dim]
        self.id_mean = latents_tensor.mean(dim=0)
        self.id_cov = torch.cov(latents_tensor.T) + 1e-6 * torch.eye(
            latents_tensor.shape[1]
        )  # regularization
        self.is_ood_fitted = True

    def score_ood_latents(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Score OOD latents using Mahalanobis distance.

        Args:
            latents: List of latent tensors

        Returns:
            Scores tensor [n_samples]
        """
        if not self.is_ood_fitted:
            raise ValueError("Distribution not fitted")
        aggregated = [latent.mean(dim=0) for latent in latents]
        latents_tensor = torch.stack(aggregated)  # [n_samples, dim]
        diff = latents_tensor - self.id_mean
        inv_cov = torch.inverse(self.id_cov)
        mahal = torch.einsum("bi,ij,bj->b", diff, inv_cov, diff)
        return mahal.sqrt()

    def run_full_analysis(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        n_samples: int = 10,
    ) -> UncertaintyResult:
        """Run full uncertainty analysis.

        Args:
            model: World model adapter
            inputs: Input tensor
            n_samples: Number of samples

        Returns:
            UncertaintyResult with all metrics
        """
        result = UncertaintyResult()

        predictions = []
        for _ in range(n_samples):
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
            predictions.append(pred)

        predictions = torch.stack(predictions)

        result.mean_prediction = predictions.mean(dim=0)
        result.variance = predictions.var(dim=0)

        if predictions.shape[-1] > 1:
            result.entropy = self.entropy_from_logits(result.mean_prediction)
            result.confidence = self.confidence_from_logits(result.mean_prediction)

        decomposition = self.decompose_uncertainty(predictions)
        result.epistemic_uncertainty = decomposition.get("epistemic_uncertainty")
        result.aleatoric_uncertainty = decomposition.get("aleatoric_uncertainty")
        result.total_uncertainty = decomposition.get("total_uncertainty")

        return result
