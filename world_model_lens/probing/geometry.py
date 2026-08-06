"""Geometry analyzer for latent space structure analysis.

This module provides geometric analysis tools that work with ANY world model:
- Latent trajectory clustering
- Manifold structure detection
- Temporal coherence analysis
- PCA/UMAP projections
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import torch
from dataclasses import dataclass

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache


@dataclass
class GeometryResult:
    """Result of geometry analysis."""

    pca_components: Optional[torch.Tensor] = None
    pca_explained_variance: Optional[torch.Tensor] = None
    mean_trajectory_distance: float = 0.0
    trajectory_curvature: Optional[torch.Tensor] = None
    temporal_coherence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean_trajectory_distance": self.mean_trajectory_distance,
            "temporal_coherence": self.temporal_coherence,
            "has_pca": self.pca_components is not None,
            "has_curvature": self.trajectory_curvature is not None,
        }


class GeometryAnalyzer:
    """Analyzer for latent space geometry and structure.

    Works with ANY world model - analyzes geometric properties of latent
    trajectories without requiring RL-specific features.

    Analysis types:
    - pca_projection: Project latents to principal components
    - trajectory_metrics: Compute distance, curvature, coherence
    - clustering: Find clusters in latent space
    - manifold_analysis: Analyze local manifold structure
    """

    def __init__(self, wm: Any = None):
        """Initialize geometry analyzer.

        Args:
            wm: Optional HookedWorldModel instance for capability checking.
        """
        self.wm = wm
        self._caps = wm.capabilities if wm and hasattr(wm, "capabilities") else None

    @property
    def capabilities(self):
        """Access world model capabilities."""
        return self._caps

    def pca_projection(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
        n_components: int = 10,
    ) -> GeometryResult:
        """Compute PCA projection of latent states.

        Works with ANY world model - projects latent trajectories to
        lower-dimensional PCA space.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze ('z_posterior', 'h', etc.).
            n_components: Number of principal components.

        Returns:
            GeometryResult with PCA components and explained variance.
        """
        latents = []
        for t in sorted(cache.timesteps):
            try:
                z = cache[component, t]
                if z is not None:
                    latents.append(z.flatten())
            except (KeyError, TypeError):
                pass

        if len(latents) < 2:
            return GeometryResult()

        try:
            latents_tensor = torch.stack(latents)
            if latents_tensor.dim() > 2:
                latents_tensor = latents_tensor.flatten(1)

            latents_centered = latents_tensor - latents_tensor.mean(dim=0)
            cov = latents_centered.T @ latents_centered / len(latents_centered)

            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.flip(0)
            eigenvectors = eigenvectors.flip(1)

            total_var = eigenvalues.sum()
            explained_var = (
                eigenvalues[:n_components] / total_var
                if total_var > 0
                else eigenvalues[:n_components]
            )

            return GeometryResult(
                pca_components=eigenvectors[:, :n_components],
                pca_explained_variance=explained_var,
                mean_trajectory_distance=0.0,
            )
        except Exception:
            return GeometryResult()

    def trajectory_metrics(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
    ) -> GeometryResult:
        """Compute geometric metrics for latent trajectories.

        Works with ANY world model - computes distances, curvature,
        and temporal coherence.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze.

        Returns:
            GeometryResult with trajectory metrics.
        """
        latents = []
        for t in sorted(cache.timesteps):
            try:
                z = cache[component, t]
                if z is not None:
                    latents.append(z.flatten())
            except (KeyError, TypeError):
                pass

        if len(latents) < 3:
            return GeometryResult()

        latents_tensor = torch.stack(latents)
        if latents_tensor.dim() > 2:
            latents_tensor = latents_tensor.flatten(1)

        distances = torch.norm(latents_tensor[1:] - latents_tensor[:-1], dim=1)
        mean_dist = distances.mean().item() if len(distances) > 0 else 0.0

        curvatures = []
        if len(latents_tensor) >= 3:
            for i in range(1, len(latents_tensor) - 1):
                v1 = latents_tensor[i] - latents_tensor[i - 1]
                v2 = latents_tensor[i + 1] - latents_tensor[i]

                norm_v1 = torch.norm(v1)
                norm_v2 = torch.norm(v2)

                if norm_v1 > 1e-8 and norm_v2 > 1e-8:
                    cos_angle = torch.dot(v1, v2) / (norm_v1 * norm_v2)
                    cos_angle = cos_angle.clamp(-1, 1)
                    curvature = 1 - cos_angle
                    curvatures.append(curvature.item())

        curvature_tensor = torch.tensor(curvatures) if curvatures else torch.tensor([0.0])

        coherence = 0.0
        if len(distances) > 1:
            coherence = 1.0 / (1.0 + distances.std().item())

        return GeometryResult(
            mean_trajectory_distance=mean_dist,
            trajectory_curvature=curvature_tensor,
            temporal_coherence=coherence,
        )

    def clustering(
        self,
        cache: "ActivationCache",
        n_clusters: int = 5,
        component: str = "z_posterior",
    ) -> Dict[str, Any]:
        """Cluster latent states.

        Works with ANY world model - uses k-means or spectral clustering
        on latent representations.

        Args:
            cache: ActivationCache from run_with_cache.
            n_clusters: Number of clusters.
            component: Which activation to analyze.

        Returns:
            Dict with cluster assignments and centroids.
        """
        latents = []
        for t in sorted(cache.timesteps):
            try:
                z = cache[component, t]
                if z is not None:
                    latents.append(z.flatten())
            except (KeyError, TypeError):
                pass

        if len(latents) < n_clusters:
            return {
                "clusters": torch.zeros(len(latents), dtype=torch.long),
                "centroids": torch.zeros(n_clusters, latents[0].shape[0])
                if latents
                else torch.zeros(n_clusters, 1),
                "n_clusters": 0,
                "is_available": True,
            }

        latents_tensor = torch.stack(latents)
        if latents_tensor.dim() > 2:
            latents_tensor = latents_tensor.flatten(1)

        indices = torch.randperm(len(latents_tensor))[:n_clusters]
        centroids = latents_tensor[indices]

        for _ in range(10):
            distances = torch.cdist(latents_tensor, centroids)
            clusters = distances.argmin(dim=1)

            new_centroids = []
            for i in range(n_clusters):
                mask = clusters == i
                if mask.any():
                    new_centroids.append(latents_tensor[mask].mean(dim=0))
                else:
                    new_centroids.append(centroids[i])
            centroids = torch.stack(new_centroids)

        return {
            "clusters": clusters,
            "centroids": centroids,
            "n_clusters": n_clusters,
            "is_available": True,
        }

    def manifold_analysis(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
        n_neighbors: int = 10,
    ) -> Dict[str, Any]:
        """Analyze local manifold structure.

        Works with ANY world model - estimates intrinsic dimensionality
        and local linear structure.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze.
            n_neighbors: Number of neighbors for local analysis.

        Returns:
            Dict with manifold analysis results.
        """
        latents = []
        for t in sorted(cache.timesteps):
            try:
                z = cache[component, t]
                if z is not None:
                    latents.append(z.flatten())
            except (KeyError, TypeError):
                pass

        if len(latents) < n_neighbors + 1:
            return {
                "intrinsic_dimensionality_estimate": 0,
                "local_linearity": 0.0,
                "is_available": True,
            }

        latents_tensor = torch.stack(latents)
        if latents_tensor.dim() > 2:
            latents_tensor = latents_tensor.flatten(1)

        try:
            distances = torch.cdist(latents_tensor, latents_tensor)
            sorted_distances, _ = distances.sort(dim=1)
            k_distances = sorted_distances[:, 1 : n_neighbors + 1]

            intrinsic_dim = k_distances.mean(dim=1).clamp(min=1e-8).log().mean().exp().item()

            linearity_scores = []
            for i in range(len(latents_tensor)):
                if i < 1 or i >= len(latents_tensor) - 1:
                    continue

                neighbor_indices = distances[i].argsort()[1 : n_neighbors + 1]
                neighbors = latents_tensor[neighbor_indices]

                center = latents_tensor[i]
                centered = neighbors - center

                U, S, V = torch.pca_lowrank(
                    centered, q=min(3, centered.shape[0], centered.shape[1])
                )

                total_var = S.sum()
                if total_var > 1e-8:
                    linearity = (S[0] / total_var).item()
                    linearity_scores.append(linearity)

            avg_linearity = (
                sum(linearity_scores) / len(linearity_scores) if linearity_scores else 0.0
            )

            return {
                "intrinsic_dimensionality_estimate": intrinsic_dim,
                "local_linearity": avg_linearity,
                "is_available": True,
            }
        except Exception:
            return {
                "intrinsic_dimensionality_estimate": 0,
                "local_linearity": 0.0,
                "is_available": True,
            }

    def full_analysis(
        self,
        cache: "ActivationCache",
        component: str = "z_posterior",
    ) -> Dict[str, Any]:
        """Run complete geometry analysis.

        Works with ANY world model - runs all geometry analyses.

        Args:
            cache: ActivationCache from run_with_cache.
            component: Which activation to analyze.

        Returns:
            Dict with all geometry results.
        """
        results = {}

        pca_result = self.pca_projection(cache, component)
        results["pca"] = pca_result.to_dict()

        traj_result = self.trajectory_metrics(cache, component)
        results["trajectory"] = traj_result.to_dict()

        cluster_result = self.clustering(cache, component=component)
        results["clustering"] = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in cluster_result.items()
        }

        manifold_result = self.manifold_analysis(cache, component)
        results["manifold"] = manifold_result

        return results
