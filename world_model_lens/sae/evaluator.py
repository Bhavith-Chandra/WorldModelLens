"""Evaluator for Sparse Autoencoders."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from world_model_lens.sae.trainer import SparseAutoencoder, _get_device


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SAEFeature:
    """Information about a single SAE feature."""

    index: int
    activation_rate: float
    mean_activation: float
    weight_vector: Tensor
    top_activation_samples: List[int]


@dataclass
class SAEFeatureAnalysis:
    """Complete analysis of SAE features."""

    n_features: int
    n_alive: int
    dead_fraction: float
    mean_activation_rate: float
    features: List[SAEFeature]
    reconstruction_fidelity: float
    sparsity: float
    interpretability_score: float


class SAEEvaluator:
    """Evaluator for Sparse Autoencoders."""

    def __init__(self, sae: SparseAutoencoder):
        """Initialize evaluator.

        Args:
            sae: Trained SparseAutoencoder.
        """
        self.sae = sae
        self.device = sae.device if hasattr(sae, "device") else _get_device()

    def reconstruction_fidelity(
        self,
        original: Tensor,
        reconstructed: Optional[Tensor] = None,
    ) -> float:
        """Compute reconstruction fidelity (R² score).

        Args:
            original: Original activations.
            reconstructed: Reconstructed activations (if None, computed from original).

        Returns:
            R² score (higher is better).
        """
        if reconstructed is None:
            self.sae.eval()
            with torch.no_grad():
                reconstructed = self.sae.decode(self.sae.encode(original)[0])

        original = original.to(self.device)
        reconstructed = reconstructed.to(self.device)

        mean_orig = original.mean(dim=0, keepdim=True)
        ss_res = ((original - reconstructed) ** 2).sum()
        ss_tot = ((original - mean_orig) ** 2).sum()

        if ss_tot < 1e-8:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return float(r2.item())

    def feature_coverage(
        self,
        test_activations: Tensor,
    ) -> Dict[str, Any]:
        """Compute feature coverage statistics.

        Args:
            test_activations: Test activations [N, d_input].

        Returns:
            Dictionary with coverage statistics.
        """
        self.sae.eval()
        test_activations = test_activations.to(self.device)

        with torch.no_grad():
            _, sparse_h, _ = self.sae(test_activations)

            activation_rate = (sparse_h.abs() > 1e-8).float().mean(dim=0)

            alive_mask = activation_rate > 0.01
            n_alive = alive_mask.sum().item()
            n_total = activation_rate.shape[0]
            dead_fraction = 1.0 - (n_alive / n_total)

            return {
                "n_alive": int(n_alive),
                "n_total": int(n_total),
                "dead_fraction": float(dead_fraction),
                "mean_activation_rate": float(activation_rate.mean().item()),
                "max_activation_rate": float(activation_rate.max().item()),
                "min_activation_rate": float(activation_rate.min().item()),
            }

    def interpretability_score(
        self,
        activations: Tensor,
        concepts: Optional[Dict[str, Tensor]] = None,
    ) -> float:
        """Compute interpretability score based on concept alignment.

        Args:
            activations: Latent activations.
            concepts: Optional dict of concept labels.

        Returns:
            Interpretability score (higher is better).
        """
        if concepts is None:
            return self._compute_variance_based_interpretability(activations)

        self.sae.eval()
        activations = activations.to(self.device)

        with torch.no_grad():
            sparse_h, _ = self.sae.encode(activations)

        total_alignment = 0.0
        n_concepts = 0

        for concept_name, concept_labels in concepts.items():
            if len(concept_labels) != len(sparse_h):
                continue

            concept_labels = concept_labels.to(self.device)

            for feat_idx in range(sparse_h.shape[1]):
                feat_activations = sparse_h[:, feat_idx]
                variance_explained = self._compute_variance_explained(
                    feat_activations, concept_labels
                )
                total_alignment += variance_explained
                n_concepts += 1

        if n_concepts == 0:
            return self._compute_variance_based_interpretability(activations)

        return total_alignment / n_concepts

    def _compute_variance_based_interpretability(self, activations: Tensor) -> float:
        """Compute interpretability based on activation variance.

        Args:
            activations: Latent activations.

        Returns:
            Interpretability score.
        """
        self.sae.eval()
        activations = activations.to(self.device)

        with torch.no_grad():
            sparse_h, _ = self.sae.encode(activations)

        feature_variance = sparse_h.var(dim=0)
        total_variance = feature_variance.sum()

        if total_variance < 1e-8:
            return 0.0

        normalized_variance = feature_variance / total_variance
        entropy = -(normalized_variance * torch.log(normalized_variance + 1e-8)).sum()

        max_entropy = torch.log(torch.tensor(float(sparse_h.shape[1])) + 1e-8)

        interpretability = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        return float(interpretability.item())

    def _compute_variance_explained(
        self,
        activations: Tensor,
        labels: Tensor,
    ) -> float:
        """Compute variance explained between activations and labels.

        Args:
            activations: Feature activations [N].
            labels: Concept labels [N].

        Returns:
            Variance explained (R²-like).
        """
        try:
            from sklearn.metrics import r2_score

            activations_np = activations.cpu().numpy()
            labels_np = labels.cpu().numpy()

            r2 = r2_score(labels_np, activations_np)
            return max(0.0, r2)
        except ImportError:
            return 0.0

    def analyze_features(
        self,
        test_activations: Tensor,
        n_top_samples: int = 5,
    ) -> SAEFeatureAnalysis:
        """Analyze all features in the SAE.

        Args:
            test_activations: Test activations [N, d_input].
            n_top_samples: Number of top activating samples to store.

        Returns:
            SAEFeatureAnalysis with detailed feature information.
        """
        self.sae.eval()
        test_activations = test_activations.to(self.device)

        with torch.no_grad():
            _, sparse_h, mask = self.sae(test_activations)

        features: List[SAEFeature] = []
        feature_weights = self.sae.get_feature_weights()

        activation_rate = (sparse_h.abs() > 1e-8).float().mean(dim=0)
        mean_activation = sparse_h.abs().mean(dim=0)

        n_features = sparse_h.shape[1]
        alive_mask = activation_rate > 0.01
        n_alive = alive_mask.sum().item()

        for idx in range(n_features):
            _, top_indices = sparse_h[:, idx].abs().topk(min(n_top_samples, sparse_h.shape[0]))

            feature = SAEFeature(
                index=idx,
                activation_rate=float(activation_rate[idx].item()),
                mean_activation=float(mean_activation[idx].item()),
                weight_vector=feature_weights[idx].detach().cpu(),
                top_activation_samples=top_indices.tolist(),
            )
            features.append(feature)

        reconstruction_fidelity = self.reconstruction_fidelity(test_activations)
        sparsity = (sparse_h.abs() > 1e-8).float().mean().item()
        interpretability = self.interpretability_score(test_activations)

        return SAEFeatureAnalysis(
            n_features=n_features,
            n_alive=int(n_alive),
            dead_fraction=1.0 - (n_alive / n_features) if n_features > 0 else 0.0,
            mean_activation_rate=float(activation_rate.mean().item()),
            features=features,
            reconstruction_fidelity=reconstruction_fidelity,
            sparsity=sparsity,
            interpretability_score=interpretability,
        )

    def decode_sparse(
        self,
        sparse_codes: Tensor,
    ) -> Tensor:
        """Decode sparse codes to reconstructions.

        Args:
            sparse_codes: Sparse latent codes [N, n_boj].

        Returns:
            Reconstructed activations [N, d_input].
        """
        self.sae.eval()
        with torch.no_grad():
            sparse_codes = sparse_codes.to(self.device)
            return self.sae.decode(sparse_codes)

    def get_feature_importance(self, test_activations: Tensor) -> Tensor:
        """Get feature importance scores based on reconstruction impact.

        Args:
            test_activations: Test activations.

        Returns:
            Importance scores for each feature.
        """
        self.sae.eval()
        test_activations = test_activations.to(self.device)

        with torch.no_grad():
            recon_original = self.sae.decode(self.sae.encode(test_activations)[0])

        importances = []

        for idx in range(self.sae.config.n_boj):
            sparse_h_modified = test_activations.clone()
            sparse_h_modified[:, idx] = 0

            recon_modified = self.sae.decode(sparse_h_modified)

            impact = ((recon_original - recon_modified) ** 2).sum(dim=1).mean()
            importances.append(impact)

        return torch.tensor(importances, device=self.device)

    def find_similar_features(
        self,
        similarity_threshold: float = 0.9,
    ) -> List[List[int]]:
        """Find groups of similar features based on decoder weight similarity.

        Args:
            similarity_threshold: Cosine similarity threshold for grouping.

        Returns:
            List of feature index groups.
        """
        feature_weights = self.sae.get_feature_weights()
        feature_weights = F.normalize(feature_weights, dim=1)

        similarity_matrix = torch.mm(feature_weights, feature_weights.T)

        n_features = similarity_matrix.shape[0]
        visited = set()
        groups = []

        for i in range(n_features):
            if i in visited:
                continue

            group = [i]
            for j in range(i + 1, n_features):
                if j not in visited and similarity_matrix[i, j] > similarity_threshold:
                    group.append(j)
                    visited.add(j)

            if len(group) > 1:
                groups.append(group)
                visited.update(group)

        return groups


class SAEBenchmark:
    """Benchmark for comparing SAE configurations."""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def benchmark(
        self,
        name: str,
        sae: SparseAutoencoder,
        test_activations: Tensor,
    ) -> Dict[str, float]:
        """Run benchmark on an SAE.

        Args:
            name: Benchmark name.
            sae: SAE to benchmark.
            test_activations: Test activations.

        Returns:
            Benchmark metrics.
        """
        evaluator = SAEEvaluator(sae)

        coverage = evaluator.feature_coverage(test_activations)
        analysis = evaluator.analyze_features(test_activations)

        metrics = {
            f"{name}/reconstruction_fidelity": analysis.reconstruction_fidelity,
            f"{name}/sparsity": analysis.sparsity,
            f"{name}/interpretability": analysis.interpretability_score,
            f"{name}/dead_features": analysis.dead_fraction,
            f"{name}/alive_features": analysis.n_alive,
            f"{name}/n_features": analysis.n_features,
        }

        for k, v in coverage.items():
            metrics[f"{name}/{k}"] = v

        self.results[name] = metrics

        return metrics

    def compare(
        self,
        metric: str = "reconstruction_fidelity",
    ) -> List[Tuple[str, float]]:
        """Compare benchmarked SAEs on a metric.

        Args:
            metric: Metric to compare.

        Returns:
            List of (name, value) sorted by value.
        """
        comparisons = []

        for name, metrics in self.results.items():
            if metric in metrics:
                comparisons.append((name, metrics[metric]))

        comparisons.sort(key=lambda x: x[1], reverse=True)
        return comparisons

    def summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks.

        Returns:
            Summary dictionary.
        """
        return {
            "benchmarks": list(self.results.keys()),
            "metrics": {
                name: {k.replace(f"{name}/", ""): v for k, v in metrics.items()}
                for name, metrics in self.results.items()
            },
        }


def compute_sparsity_curve(
    sae: SparseAutoencoder,
    test_activations: Tensor,
    k_values: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute metrics across different k values (sparsity levels).

    Args:
        sae: Base SAE.
        test_activations: Test activations.
        k_values: List of k values to try.

    Returns:
        Dictionary mapping k to metrics.
    """
    from copy import deepcopy

    results = {}

    for k in k_values:
        sae_copy = deepcopy(sae)
        sae_copy.topk.k = k
        sae_copy.eval()

        evaluator = SAEEvaluator(sae_copy)

        with torch.no_grad():
            _, sparse_h, _ = sae_copy(test_activations)

        sparsity = (sparse_h.abs() > 1e-8).float().mean().item()
        reconstruction = evaluator.reconstruction_fidelity(test_activations)

        results[k] = {
            "sparsity": sparsity,
            "reconstruction_fidelity": reconstruction,
        }

    return results
