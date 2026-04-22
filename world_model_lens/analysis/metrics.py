"""Latent space metrics for world model evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
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

    component_errors: dict[str, str] = field(default_factory=dict)
    """Errors encountered while evaluating individual components."""


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

    @staticmethod
    def _extract_latents(cache: Any, component: str) -> torch.Tensor:
        """Load and flatten latents for one component."""
        latents = cache[component]

        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected tensor activations for {component}, got {type(latents)!r}")

        if latents.dim() == 3:
            latents = latents.flatten(1)
        elif latents.dim() == 1:
            latents = latents.unsqueeze(0)
        elif latents.dim() != 2:
            raise ValueError(
                f"Expected 1D, 2D, or 3D activations for {component}, got shape {tuple(latents.shape)}"
            )

        if latents.shape[0] < 2:
            raise ValueError(f"Need at least two samples for disentanglement on {component}")

        return latents.detach().float()

    @staticmethod
    def _resolve_component_factors(
        factors: Mapping[str, Any],
        component: str,
    ) -> tuple[list[str], torch.Tensor]:
        """Resolve shared or per-component factor mappings into a stacked tensor."""
        if not factors:
            raise ValueError("factors must be a non-empty mapping")

        first_value = next(iter(factors.values()))
        if isinstance(first_value, Mapping):
            if component not in factors:
                raise KeyError(f"Missing factors for component '{component}'")
            component_factors = factors[component]
        else:
            component_factors = factors

        if not component_factors:
            raise ValueError(f"No factors provided for component '{component}'")

        factor_names = list(component_factors.keys())
        factor_tensors = []
        for factor_name in factor_names:
            values = component_factors[factor_name]
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if values.dim() != 1:
                raise ValueError(
                    f"Factor '{factor_name}' for component '{component}' must be 1D, got shape {tuple(values.shape)}"
                )
            factor_tensors.append(values.detach().float())

        stacked = torch.stack(factor_tensors, dim=1)
        if stacked.shape[0] < 2:
            raise ValueError(f"Need at least two factor samples for component '{component}'")
        return factor_names, stacked

    @staticmethod
    def _metric_fields(metrics: list[str]) -> list[str]:
        fields: list[str] = []
        for metric in metrics:
            if metric == "MIG":
                fields.append("MIG")
            elif metric == "DCI":
                fields.extend(
                    ["DCI_disentanglement", "DCI_completeness", "DCI_informativeness"]
                )
            elif metric == "SAP":
                fields.append("SAP")
        return fields

    def evaluate_components(
        self,
        cache: Any,  # ActivationCache
        factors: Mapping[str, Any],
        components: list[str],
        metrics: list[str] | None = None,
    ) -> DisentanglementEvaluationResult:
        """Evaluate disentanglement across multiple components.

        Args:
            cache: ActivationCache containing model activations.
            factors: Shared factor mapping or per-component factor mappings.
            components: List of component names to evaluate (e.g., ['z_posterior', 'context_encoder', 'predictor']).
            metrics: Metrics to compute ('MIG', 'DCI', 'SAP'). Defaults to all.

        Returns:
            DisentanglementEvaluationResult with per-component and summary scores.
        """
        if metrics is None:
            metrics = ["MIG", "DCI", "SAP"]
        else:
            metrics = [metric.upper() for metric in metrics]

        unknown_metrics = [metric for metric in metrics if metric not in {"MIG", "DCI", "SAP"}]
        if unknown_metrics:
            raise ValueError(f"Unknown disentanglement metrics requested: {unknown_metrics}")

        component_results = {}
        component_errors = {}

        for component in components:
            try:
                latents = self._extract_latents(cache, component)
                _, component_factors = self._resolve_component_factors(factors, component)
                if latents.shape[0] != component_factors.shape[0]:
                    raise ValueError(
                        f"Sample mismatch for {component}: latents have {latents.shape[0]} rows, "
                        f"factors have {component_factors.shape[0]}"
                    )

                metric_result: dict[str, float] = {}
                if "MIG" in metrics:
                    metric_result["MIG"] = float(self.analyzer.mig(latents, component_factors))
                if "DCI" in metrics:
                    disentanglement, completeness, informativeness = self.analyzer.dci(
                        latents, component_factors
                    )
                    metric_result["DCI_disentanglement"] = float(disentanglement)
                    metric_result["DCI_completeness"] = float(completeness)
                    metric_result["DCI_informativeness"] = float(informativeness)
                if "SAP" in metrics:
                    metric_result["SAP"] = float(self.analyzer.sap(latents, component_factors))

                component_results[component] = metric_result

            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                component_errors[component] = str(exc)

        if not component_results:
            raise RuntimeError(
                "Disentanglement evaluation failed for every requested component: "
                + "; ".join(f"{name}: {msg}" for name, msg in component_errors.items())
            )

        summary_scores = {}
        for metric_name in self._metric_fields(metrics):
            values = [
                result[metric_name]
                for result in component_results.values()
                if metric_name in result
            ]
            if values:
                summary_scores[metric_name] = float(sum(values) / len(values))

        return DisentanglementEvaluationResult(
            component_results=component_results,
            summary_scores=summary_scores,
            component_errors=component_errors,
        )
