"""Disentanglement analyzer for measuring factor separation in latent spaces.

This module provides tools to analyze how well a world model's latent space
captures independent factors of variation. Works with ANY world model type.

Metrics:
- Correlation: Measure linear correlation between latents and factors
- MIG: Mutual Information Gap
- DCI: Disentanglement, Completeness, Informativeness
- SAP: Separated Attribute Predictability
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache


@dataclass
class DisentanglementResult:
    """Result of disentanglement analysis."""

    correlation_matrix: np.ndarray | None = None
    mig_score: float | None = None
    dci_disentanglement: float | None = None
    dci_completeness: float | None = None
    dci_informativeness: float | None = None
    sap_score: float | None = None
    latent_variance: list[float] | None = None
    factor_variance: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mig_score": self.mig_score,
            "dci_disentanglement": self.dci_disentanglement,
            "dci_completeness": self.dci_completeness,
            "dci_informativeness": self.dci_informativeness,
            "sap_score": self.sap_score,
        }


class DisentanglementAnalyzer:
    """Analyzer for measuring disentanglement in latent spaces.

    Works with ANY world model - measures how well the latent space
    separates independent factors of variation.

    Can work with:
    - Ground truth factors (if available)
    - Predicted factors (from probing)
    """

    def __init__(self, n_bins: int = 10):
        """Initialize analyzer.

        Args:
            n_bins: Number of bins for discrete factor binning
        """
        self.n_bins = n_bins

    def compute_correlation(
        self,
        latents: torch.Tensor,
        factors: torch.Tensor,
    ) -> np.ndarray:
        """Compute correlation matrix between latents and factors.

        Args:
            latents: Latent activations [N, d_latent]
            factors: Ground truth factors [N, n_factors]

        Returns:
            Correlation matrix [n_factors, d_latent]
        """
        latents_np = latents.detach().cpu().numpy()
        factors_np = factors.detach().cpu().numpy()

        correlations = np.zeros((factors_np.shape[1], latents_np.shape[1]))

        for i in range(factors_np.shape[1]):
            for j in range(latents_np.shape[1]):
                correlations[i, j] = np.corrcoef(factors_np[:, i], latents_np[:, j])[0, 1]

        return correlations

    def mig(
        self,
        latents: torch.Tensor,
        factors: torch.Tensor,
    ) -> float:
        """Compute Mutual Information Gap score.

        Args:
            latents: Latent activations [N, d_latent]
            factors: Ground truth factors [N, n_factors]

        Returns:
            MIG score (higher is better)
        """
        latents_np = latents.detach().cpu().numpy()
        factors_np = factors.detach().cpu().numpy()

        n_factors = factors_np.shape[1]
        n_latents = latents_np.shape[1]

        mutual_info = np.zeros((n_factors, n_latents))

        for i in range(n_factors):
            factor_bins = np.digitize(
                factors_np[:, i],
                np.linspace(factors_np[:, i].min(), factors_np[:, i].max(), self.n_bins),
            )

            for j in range(n_latents):
                latent_vals = latents_np[:, j]

                joint_hist, _, _ = np.histogram2d(
                    factor_bins,
                    np.digitize(
                        latent_vals, np.linspace(latent_vals.min(), latent_vals.max(), self.n_bins)
                    ),
                )
                joint_prob = joint_hist / joint_hist.sum()

                p_factor = joint_prob.sum(axis=1)
                p_latent = joint_prob.sum(axis=0)

                mi = 0.0
                for fi in range(len(p_factor)):
                    for li in range(len(p_latent)):
                        if joint_prob[fi, li] > 0:
                            mi += joint_prob[fi, li] * np.log(
                                joint_prob[fi, li] / (p_factor[fi] * p_latent[li] + 1e-10) + 1e-10
                            )

                mutual_info[i, j] = mi

        sorted_mi = np.sort(mutual_info, axis=1)[:, ::-1]

        mig = 0.0
        for i in range(n_factors):
            if sorted_mi[i, 0] > 0:
                mig += (sorted_mi[i, 0] - sorted_mi[i, 1]) / np.log2(n_latents)

        return mig / n_factors

    def _entropy(self, x: np.ndarray) -> float:
        """Compute entropy of discrete variable."""
        _, counts = np.unique(x, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))

    def dci(
        self,
        latents: torch.Tensor,
        factors: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Compute DCI metrics (Disentanglement, Completeness, Informativeness).

        Args:
            latents: Latent activations [N, d_latent]
            factors: Ground truth factors [N, n_factors]

        Returns:
            Tuple of (disentanglement, completeness, informativeness)
        """
        correlations = self.compute_correlation(latents, factors)
        importances = np.abs(correlations)

        importance_sum = importances.sum(axis=1, keepdims=True)
        importance_sum[importance_sum == 0] = 1
        importance_normalized = importances / importance_sum

        n_factors = factors.shape[1]
        n_latents = latents.shape[1]

        disentanglement = 0.0
        for j in range(n_latents):
            disentanglement += 1 - self._gini(importance_normalized[:, j])
        disentanglement /= n_latents

        completeness = 0.0
        for i in range(n_factors):
            completeness += 1 - self._gini(importance_normalized[i, :])
        completeness /= n_factors

        pred = np.linalg.lstsq(
            latents.detach().cpu().numpy(), factors.detach().cpu().numpy(), rcond=None
        )[0]
        pred_factors = latents.detach().cpu().numpy() @ pred
        mse = ((pred_factors - factors.detach().cpu().numpy()) ** 2).mean()
        informativeness = 1 / (1 + mse)

        return disentanglement, completeness, informativeness

    def _gini(self, x: np.ndarray) -> float:
        """Compute Gini coefficient."""
        x = np.sort(np.abs(x))
        n = len(x)
        if n == 0:
            return 0.0
        cumx = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    def sap(
        self,
        latents: torch.Tensor,
        factors: torch.Tensor,
    ) -> float:
        """Compute Separated Attribute Predictability score.

        Args:
            latents: Latent activations [N, d_latent]
            factors: Ground truth factors [N, n_factors]

        Returns:
            SAP score (higher is better)
        """
        latents_np = latents.detach().cpu().numpy()
        factors_np = factors.detach().cpu().numpy()

        n_factors = factors_np.shape[1]
        n_latents = latents_np.shape[1]

        scores = np.zeros((n_factors, n_latents))

        for i in range(n_factors):
            for j in range(n_latents):
                from sklearn.linear_model import Ridge

                model = Ridge(alpha=1.0)
                model.fit(latents_np[:, j : j + 1], factors_np[:, i])
                pred = model.predict(latents_np[:, j : j + 1])
                scores[i, j] = 1 - ((pred - factors_np[:, i]) ** 2).mean() / (
                    factors_np[:, i].var() + 1e-10
                )

        sorted_scores = np.sort(scores, axis=1)[:, ::-1]

        sap = 0.0
        for i in range(n_factors):
            if sorted_scores[i, 0] > sorted_scores[i, 1]:
                sap += sorted_scores[i, 0] - sorted_scores[i, 1]

        return sap / n_factors

    def analyze(
        self,
        latents: torch.Tensor,
        factors: torch.Tensor,
    ) -> DisentanglementResult:
        """Run full disentanglement analysis.

        Args:
            latents: Latent activations [N, d_latent]
            factors: Ground truth factors [N, n_factors]

        Returns:
            DisentanglementResult with all metrics
        """
        result = DisentanglementResult()

        result.correlation_matrix = self.compute_correlation(latents, factors)

        try:
            result.mig_score = self.mig(latents, factors)
        except Exception:
            result.mig_score = None

        try:
            d, c, i = self.dci(latents, factors)
            result.dci_disentanglement = d
            result.dci_completeness = c
            result.dci_informativeness = i
        except Exception:
            pass

        try:
            result.sap_score = self.sap(latents, factors)
        except Exception:
            result.sap_score = None

        result.latent_variance = latents.var(dim=0).detach().cpu().numpy().tolist()
        result.factor_variance = factors.var(dim=0).detach().cpu().numpy().tolist()

        return result

    def from_cache(
        self,
        cache: "ActivationCache",
        latent_key: str = "z",
        factors: torch.Tensor | None = None,
    ) -> DisentanglementResult:
        """Analyze disentanglement from activation cache.

        Args:
            cache: ActivationCache
            latent_key: Key for latent activations
            factors: Ground truth factors (optional)

        Returns:
            DisentanglementResult
        """
        latents = cache[latent_key]

        if factors is None:
            raise ValueError("factors must be provided for disentanglement analysis")

        return self.analyze(latents, factors)
