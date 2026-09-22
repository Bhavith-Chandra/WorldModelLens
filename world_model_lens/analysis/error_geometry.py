"""Error Geometry & Precision-Weighted Failure Analysis (Suggestions 4 & 6)

Analyzes prediction error geometry via PCA subspace breakdown and precision-weighted
Mahalanobis distance to separate isotropic magnitude errors from semantic subspace errors.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple, Optional


class ErrorGeometryAnalyzer:
    """Analyzes error vector geometry, PCA spectrum alignment, and Mahalanobis distances."""

    def __init__(self, regularizer: float = 1e-4):
        self.regularizer = regularizer
        self.target_mean = None
        self.target_cov_inv = None

    def fit_target_covariance(self, target_vectors: np.ndarray):
        """Fits empirical mean and inverse covariance matrix over target encoder representations.
        
        Args:
            target_vectors: [M, D] array of ground truth target embeddings.
        """
        M, D = target_vectors.shape
        self.target_mean = np.mean(target_vectors, axis=0)
        cov = np.cov(target_vectors, rowvar=False)  # [D, D]
        
        # Add shrinkage regularization to handle singular matrices
        reg_cov = cov + self.regularizer * np.eye(D)
        self.target_cov_inv = np.linalg.inv(reg_cov)

    def compute_mahalanobis_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Computes Mahalanobis distance between predicted and ground truth latent vectors.
        
        Args:
            pred: [D] prediction vector.
            target: [D] ground truth vector.
            
        Returns:
            Mahalanobis distance float.
        """
        diff = pred - target
        if self.target_cov_inv is None:
            # Fallback to Euclidean L2 norm
            return float(np.linalg.norm(diff))
            
        mahal_sq = np.dot(np.dot(diff, self.target_cov_inv), diff)
        return float(np.sqrt(max(0.0, mahal_sq)))

    def analyze_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_components: int = 10
    ) -> Dict[str, Any]:
        """Runs full error geometry analysis over a dataset of predictions and ground truths.
        
        Args:
            predictions: [M, D] array of predicted target features.
            targets: [M, D] array of ground truth target features.
            n_components: Number of PCA principal components to extract.
            
        Returns:
            Dict containing PCA variance spectrum, Mahalanobis vs Euclidean scores, and alignment.
        """
        M, D = predictions.shape
        error_vectors = predictions - targets  # [M, D]

        # 1. Fit PCA on Error Vectors
        pca = PCA(n_components=min(n_components, M, D))
        pca.fit(error_vectors)

        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # Top-K error subspace energy
        top_1_ratio = float(explained_var[0]) if len(explained_var) > 0 else 0.0
        top_5_ratio = float(cumulative_var[min(4, len(cumulative_var)-1)]) if len(cumulative_var) > 0 else 0.0

        # 2. Fit Target Covariance for Mahalanobis scoring
        if self.target_cov_inv is None:
            self.fit_target_covariance(targets)

        # 3. Compute per-sample metrics
        euclidean_mses = []
        mahalanobis_dists = []

        for i in range(M):
            e_mse = float(np.mean((predictions[i] - targets[i]) ** 2))
            m_dist = self.compute_mahalanobis_distance(predictions[i], targets[i])
            euclidean_mses.append(e_mse)
            mahalanobis_dists.append(m_dist)

        euclidean_mses = np.array(euclidean_mses)
        mahalanobis_dists = np.array(mahalanobis_dists)

        # Rank correlation between Euclidean MSE and Mahalanobis Distance
        from scipy import stats
        spearman_rank_corr, _ = stats.spearmanr(euclidean_mses, mahalanobis_dists)

        return {
            "num_samples": M,
            "latent_dim": D,
            "pca_explained_variance_ratio": [float(x) for x in explained_var],
            "pca_cumulative_variance_ratio": [float(x) for x in cumulative_var],
            "top_1_error_variance_share": top_1_ratio,
            "top_5_error_variance_share": top_5_ratio,
            "mean_euclidean_mse": float(np.mean(euclidean_mses)),
            "mean_mahalanobis_dist": float(np.mean(mahalanobis_dists)),
            "rank_correlation_mse_vs_mahalanobis": float(spearman_rank_corr)
        }
