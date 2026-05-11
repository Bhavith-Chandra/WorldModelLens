"""Latent space metrics for world model evaluation."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from world_model_lens.analysis.disentanglement import DisentanglementAnalyzer


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LatentMetricsResult:
    """Results from latent space metrics."""

    compression_ratio: float
    predictive_info: float
    temporal_coherence: float
    reconstruction_error: float
    latent_variance: float


@dataclass
class DisentanglementEvaluationResult:
    """Results from disentanglement evaluation across multiple components."""

    component_results: dict[str, dict[str, float]]
    """Dictionary mapping component names to their MIG/DCI/SAP scores."""

    summary_scores: dict[str, float]
    """Aggregated scores across all components."""


class LatentMetrics:
    """Compute various metrics on world model latent representations."""

    @staticmethod
    def compression_ratio(original_obs: torch.Tensor, latent_obs: torch.Tensor) -> float:
        """Compute compression ratio of latent representation.

        Args:
            original_obs: Original observations [T, ...].
            latent_obs: Latent representations [T, ...].

        Returns:
            Compression ratio (original_size / latent_size).
        """
        orig_size = original_obs.flatten(1).shape[1]
        lat_size = latent_obs.flatten(1).shape[1]

        if lat_size == 0:
            return 0.0

        return orig_size / lat_size

    @staticmethod
    def predictive_info(actions: torch.Tensor, latents: torch.Tensor) -> float:
        """Compute predictive information (mutual info between actions and next state).

        Args:
            actions: Action sequence [T, d_action].
            latents: Latent sequence [T, d_latent].

        Returns:
            Predictive information estimate.
        """
        if len(actions) < 2 or len(latents) < 2:
            return 0.0

        actions = actions[:-1]
        latents[1:]
        current_latents = latents[:-1]

        joint = torch.cat([actions, current_latents], dim=1)
        marginal_a = actions
        marginal_s = current_latents

        joint_cov = torch.cov(joint.T)
        marg_a_cov = torch.cov(marginal_a.T)
        marg_s_cov = torch.cov(marginal_s.T)

        if joint_cov.det() <= 0 or marg_a_cov.det() <= 0 or marg_s_cov.det() <= 0:
            return 0.0

        joint_entropy = 0.5 * torch.logdet(joint_cov)
        action_entropy = 0.5 * torch.logdet(marg_a_cov)
        state_entropy = 0.5 * torch.logdet(marg_s_cov)

        mi = action_entropy + state_entropy - joint_entropy
        return max(0.0, float(mi.item()))

    @staticmethod
    def temporal_hierarchy(latents: torch.Tensor) -> dict[str, Any]:
        """Analyze temporal hierarchy in latent sequences.

        Args:
            latents: Latent sequence [T, d_latent].

        Returns:
            Dict with hierarchy metrics.
        """
        if len(latents) < 3:
            return {"hierarchical_score": 0.0, "slow_features": [], "fast_features": []}

        latents_flat = latents.flatten(1)

        variances = latents_flat.var(dim=0)

        temporal_diffs = (latents[1:] - latents[:-1]).flatten(1)
        diff_variances = temporal_diffs.var(dim=0)

        slow_mask = variances > 2 * variances.median()
        fast_mask = diff_variances > 2 * diff_variances.median()

        slow_features = slow_mask.nonzero().squeeze().tolist() if slow_mask.any() else []
        fast_features = fast_mask.nonzero().squeeze().tolist() if fast_mask.any() else []

        hierarchical_score = len(slow_features) / max(len(variances), 1)

        return {
            "hierarchical_score": float(hierarchical_score),
            "slow_features": slow_features,
            "fast_features": fast_features,
            "n_slow": len(slow_features),
            "n_fast": len(fast_features),
        }

    @staticmethod
    def reconstruction_error(wm: Any, obs: torch.Tensor, actions: torch.Tensor) -> float:
        """Compute reconstruction error of the world model.

        Args:
            wm: HookedWorldModel.
            obs: Observations.
            actions: Actions.

        Returns:
            MSE reconstruction error.
        """
        traj, cache = wm.run_with_cache(obs, actions)

        try:
            recon = cache["reconstruction"]
            orig = cache["observation"]
            mse = functional.mse_loss(recon, orig)
            return float(mse.item())
        except Exception:
            return float("inf")

    @staticmethod
    def latent_variance(latents: torch.Tensor) -> float:
        """Compute total variance of latent representations.

        Args:
            latents: Latent tensor [T, d_latent].

        Returns:
            Total variance.
        """
        if latents.dim() == 3:
            latents = latents.flatten(1)

        variance = latents.var(dim=0).sum()
        return float(variance.item())

    @staticmethod
    def compute_all(
        latents: torch.Tensor, actions: torch.Tensor, original_obs: torch.Tensor | None = None
    ) -> LatentMetricsResult:
        """Compute all metrics at once.

        Args:
            latents: Latent sequence.
            actions: Action sequence.
            original_obs: Optional original observations.

        Returns:
            LatentMetricsResult with all computed metrics.
        """
        compression = 1.0
        if original_obs is not None:
            compression = LatentMetrics.compression_ratio(original_obs, latents)

        predictive_info = LatentMetrics.predictive_info(actions, latents)
        temporal_hierarchy = LatentMetrics.temporal_hierarchy(latents)
        variance = LatentMetrics.latent_variance(latents)

        return LatentMetricsResult(
            compression_ratio=compression,
            predictive_info=predictive_info,
            temporal_coherence=temporal_hierarchy.get("hierarchical_score", 0.0),
            reconstruction_error=0.0,
            latent_variance=variance,
        )


class CausalBenchmark:
    """Benchmark for causal analysis methods."""

    @staticmethod
    def evaluate_patching(model: Any, ground_truth_circuit: dict[str, Any]) -> dict[str, float]:
        """Evaluate patching accuracy against ground truth.

        Args:
            model: Model to evaluate.
            ground_truth_circuit: Known important components.

        Returns:
            Dict of evaluation metrics.
        """
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    @staticmethod
    def probe_attribution_accuracy(model: Any, concepts: dict[str, torch.Tensor]) -> float:
        """Measure accuracy of probe-based attribution.

        Args:
            model: Model to evaluate.
            concepts: Ground truth concept labels.

        Returns:
            Attribution accuracy.
        """
        return 0.0

    @staticmethod
    def circuit_stability(model: Any, perturbations: list[dict[str, Any]]) -> float:
        """Measure stability of discovered circuits under perturbation.

        Args:
            model: Model to evaluate.
            perturbations: List of perturbations to apply.

        Returns:
            Stability score.
        """
        return 0.0


class DisentanglementEvaluationSuite:
    """Unified evaluation suite for latent representation disentanglement.

    Computes MIG, DCI, and SAP scores across multiple model components
    to quantify how well latent spaces separate independent factors of variation.
    """

    def __init__(self, n_bins: int = 10):
        """Initialize the evaluation suite.

        Args:
            n_bins: Number of bins for discretizing factors in MIG/SAP computation.
        """
        self.analyzer = DisentanglementAnalyzer(n_bins=n_bins)

    def evaluate_components(
        self,
        cache: Any,  # ActivationCache
        factors: dict[str, torch.Tensor],
        components: list[str],
        metrics: list[str] | None = None,
    ) -> DisentanglementEvaluationResult:
        """Evaluate disentanglement across multiple components.

        Args:
            cache: ActivationCache containing model activations.
            factors: Dictionary mapping factor names to factor value tensors [T].
            components: List of component names to evaluate (e.g., ['z_posterior', 'context_encoder', 'predictor']).
            metrics: Metrics to compute ('MIG', 'DCI', 'SAP'). Defaults to all.

        Returns:
            DisentanglementEvaluationResult with per-component and summary scores.
        """
        if metrics is None:
            metrics = ["MIG", "DCI", "SAP"]

        component_results = {}

        for component in components:
            try:
                # Get latent representations for this component
                latents = cache[component]  # This should work for any component in cache

                # Ensure latents are properly shaped [T, d_latent]
                if latents.dim() == 3:
                    latents = latents.flatten(1)
                elif latents.dim() == 1:
                    latents = latents.unsqueeze(0)

                # Compute disentanglement scores
                result = self.analyzer.analyze(latents, torch.stack(list(factors.values()), dim=1))

                component_results[component] = {
                    "MIG": result.mig_score if result.mig_score is not None else 0.0,
                    "DCI_disentanglement": result.dci_disentanglement
                    if result.dci_disentanglement is not None
                    else 0.0,
                    "DCI_completeness": result.dci_completeness
                    if result.dci_completeness is not None
                    else 0.0,
                    "DCI_informativeness": result.dci_informativeness
                    if result.dci_informativeness is not None
                    else 0.0,
                    "SAP": result.sap_score if result.sap_score is not None else 0.0,
                }

            except (KeyError, ValueError, RuntimeError):
                # Component not found or computation failed
                component_results[component] = {
                    "MIG": 0.0,
                    "DCI_disentanglement": 0.0,
                    "DCI_completeness": 0.0,
                    "DCI_informativeness": 0.0,
                    "SAP": 0.0,
                }

        # Compute summary scores (average across components)
        summary_scores = {}
        if component_results:
            for metric in [
                "MIG",
                "DCI_disentanglement",
                "DCI_completeness",
                "DCI_informativeness",
                "SAP",
            ]:
                valid_scores = [
                    res[metric] for res in component_results.values() if res[metric] != 0.0
                ]
                summary_scores[metric] = (
                    sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                )

        return DisentanglementEvaluationResult(
            component_results=component_results,
            summary_scores=summary_scores,
        )
