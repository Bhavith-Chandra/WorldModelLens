from __future__ import annotations

"""Out-of-Distribution (OOD) detection for world model latent spaces.

This module provides multiple OOD detection methods:
- Mahalanobis distance-based detection
- Isolation Forest anomaly detection
- Gaussian Mixture Model-based detection
- Energy-based detection
- k-NN density estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class OODResult:
    """Result of OOD detection."""

    scores: torch.Tensor
    is_ood: torch.Tensor
    threshold: float
    method: str
    details: dict[str, Any]


class MahalanobisOODDetector:
    """Mahalanobis distance-based OOD detector."""

    def __init__(self, device: str = "cpu"):
        """Initialize detector.

        Args:
            device: Device for computations
        """
        self.device = device
        self.mean: torch.Tensor | None = None
        self.cov_inv: torch.Tensor | None = None
        self.fitted = False

    def fit(self, training_data: torch.Tensor) -> MahalanobisOODDetector:
        """Fit the detector on training data.

        Args:
            training_data: Training latents [N, D]

        Returns:
            Self
        """
        self.mean = training_data.mean(dim=0)
        centered = training_data - self.mean

        cov = torch.cov(centered.T)
        cov += torch.eye(cov.shape[0], device=self.device) * 1e-6

        try:
            self.cov_inv = torch.linalg.inv(cov)
        except:
            self.cov_inv = torch.linalg.pinv(cov)

        self.fitted = True
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distances.

        Args:
            data: Data to score [N, D]

        Returns:
            Distances [N]
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted")

        centered = data - self.mean
        distances = torch.sqrt(torch.sum(centered @ self.cov_inv * centered, dim=1))
        return distances

    def detect(
        self,
        data: torch.Tensor,
        threshold: float | None = None,
    ) -> OODResult:
        """Detect OOD samples.

        Args:
            data: Data to detect [N, D]
            threshold: Detection threshold (auto-computed if None)

        Returns:
            OODResult
        """
        scores = self.score(data)

        if threshold is None:
            threshold = scores.mean() + 2 * scores.std()

        is_ood = scores > threshold

        return OODResult(
            scores=scores,
            is_ood=is_ood,
            threshold=threshold,
            method="mahalanobis",
            details={"mean": self.mean, "fitted": self.fitted},
        )


class IsolationForestDetector:
    """Isolation Forest-based OOD detector."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """Initialize detector.

        Args:
            contamination: Expected proportion of OOD samples
            random_state: Random seed
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model: IsolationForest | None = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, training_data: torch.Tensor) -> IsolationForestDetector:
        """Fit the detector.

        Args:
            training_data: Training latents [N, D]

        Returns:
            Self
        """
        data_np = training_data.cpu().numpy()
        data_scaled = self.scaler.fit_transform(data_np)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.model.fit(data_scaled)
        self.fitted = True
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores (negative = more anomalous).

        Args:
            data: Data to score [N, D]

        Returns:
            Scores [N]
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted")

        data_np = data.cpu().numpy()
        data_scaled = self.scaler.transform(data_np)

        scores = self.model.score_samples(data_scaled)
        return torch.tensor(-scores, dtype=data.dtype)

    def detect(
        self,
        data: torch.Tensor,
        threshold: float | None = None,
    ) -> OODResult:
        """Detect OOD samples.

        Args:
            data: Data to detect [N, D]
            threshold: Detection threshold

        Returns:
            OODResult
        """
        scores = self.score(data)

        if threshold is None:
            threshold = scores.mean() + 2 * scores.std()

        is_ood = scores > threshold

        return OODResult(
            scores=scores,
            is_ood=is_ood,
            threshold=threshold,
            method="isolation_forest",
            details={"fitted": self.fitted},
        )


class EnergyOODDetector:
    """Energy-based OOD detection (from Liu et al., 2020)."""

    def __init__(self, temperature: float = 1.0):
        """Initialize detector.

        Args:
            temperature: Temperature for energy computation
        """
        self.temperature = temperature
        self.model: torch.nn.Module | None = None
        self.fitted = False

    def fit(self, training_data: torch.Tensor) -> EnergyOODDetector:
        """Store training statistics.

        Args:
            training_data: Training latents [N, D]

        Returns:
            Self
        """
        self.mean = training_data.mean(dim=0)
        self.std = training_data.std(dim=0) + 1e-6
        self.fitted = True
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute energy scores.

        Args:
            data: Data to score [N, D]

        Returns:
            Energy scores [N]
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted")

        normalized = (data - self.mean) / self.std
        energy = self.temperature * torch.logsumexp(normalized / self.temperature, dim=1)
        return energy

    def detect(
        self,
        data: torch.Tensor,
        threshold: float | None = None,
    ) -> OODResult:
        """Detect OOD samples.

        Args:
            data: Data to detect [N, D]
            threshold: Detection threshold

        Returns:
            OODResult
        """
        scores = self.score(data)

        if threshold is None:
            threshold = scores.mean() + 2 * scores.std()

        is_ood = scores > threshold

        return OODResult(
            scores=scores,
            is_ood=is_ood,
            threshold=threshold,
            method="energy",
            details={"temperature": self.temperature, "fitted": self.fitted},
        )


class KNNDensityDetector:
    """k-NN density-based OOD detector."""

    def __init__(self, k: int = 5):
        """Initialize detector.

        Args:
            k: Number of neighbors
        """
        self.k = k
        self.model: NearestNeighbors | None = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, training_data: torch.Tensor) -> KNNDensityDetector:
        """Fit the detector.

        Args:
            training_data: Training latents [N, D]

        Returns:
            Self
        """
        data_np = training_data.cpu().numpy()
        data_scaled = self.scaler.fit_transform(data_np)

        self.model = NearestNeighbors(n_neighbors=self.k + 1, algorithm="ball_tree")
        self.model.fit(data_scaled)
        self.training_data = data_scaled
        self.fitted = True
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        """Compute k-NN density scores (lower = more OOD).

        Args:
            data: Data to score [N, D]

        Returns:
            Density scores [N]
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted")

        data_np = data.cpu().numpy()
        data_scaled = self.scaler.transform(data_np)

        distances, _ = self.model.kneighbors(data_scaled)
        knn_distances = distances[:, 1:]

        density = 1.0 / (knn_distances.mean(axis=1) + 1e-6)
        return torch.tensor(density, dtype=data.dtype)

    def detect(
        self,
        data: torch.Tensor,
        threshold: float | None = None,
    ) -> OODResult:
        """Detect OOD samples.

        Args:
            data: Data to detect [N, D]
            threshold: Detection threshold (lower = OOD)

        Returns:
            OODResult
        """
        scores = self.score(data)

        if threshold is None:
            threshold = scores.mean() - 2 * scores.std()

        is_ood = scores < threshold

        return OODResult(
            scores=scores,
            is_ood=is_ood,
            threshold=threshold,
            method="knn_density",
            details={"k": self.k, "fitted": self.fitted},
        )


class EnsembleOODDetector:
    """Ensemble of multiple OOD detection methods."""

    def __init__(self, methods: list[str] | None = None):
        """Initialize ensemble detector.

        Args:
            methods: List of methods to ensemble
        """
        self.methods = methods or ["mahalanobis", "isolation_forest", "energy", "knn"]
        self.detectors: dict[str, Any] = {}
        self.fitted = False

    def fit(self, training_data: torch.Tensor) -> EnsembleOODDetector:
        """Fit all detectors.

        Args:
            training_data: Training latents [N, D]

        Returns:
            Self
        """
        for method in self.methods:
            if method == "mahalanobis":
                self.detectors[method] = MahalanobisOODDetector().fit(training_data)
            elif method == "isolation_forest":
                self.detectors[method] = IsolationForestDetector().fit(training_data)
            elif method == "energy":
                self.detectors[method] = EnergyOODDetector().fit(training_data)
            elif method == "knn":
                self.detectors[method] = KNNDensityDetector().fit(training_data)

        self.fitted = True
        return self

    def score(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get scores from all methods.

        Args:
            data: Data to score [N, D]

        Returns:
            Dictionary of scores per method
        """
        if not self.fitted:
            raise RuntimeError("Detectors not fitted")

        scores = {}
        for method, detector in self.detectors.items():
            scores[method] = detector.score(data)

        return scores

    def detect(
        self,
        data: torch.Tensor,
        threshold: float | None = None,
        fusion: str = "mean",
    ) -> OODResult:
        """Detect OOD samples using ensemble.

        Args:
            data: Data to detect [N, D]
            threshold: Detection threshold
            fusion: How to fuse scores ("mean", "max", "min")

        Returns:
            OODResult
        """
        if not self.fitted:
            raise RuntimeError("Detectors not fitted")

        scores_dict = self.score(data)

        scores_list = list(scores_dict.values())
        if fusion == "mean":
            scores = torch.stack(scores_list).mean(dim=0)
        elif fusion == "max":
            scores = torch.stack(scores_list).max(dim=0)[0]
        elif fusion == "min":
            scores = torch.stack(scores_list).min(dim=0)[0]
        else:
            scores = torch.stack(scores_list).mean(dim=0)

        if threshold is None:
            threshold = scores.mean() + 2 * scores.std()

        is_ood = scores > threshold

        return OODResult(
            scores=scores,
            is_ood=is_ood,
            threshold=threshold,
            method=f"ensemble_{fusion}",
            details={
                "method_scores": {k: v.tolist() for k, v in scores_dict.items()},
                "num_methods": len(self.methods),
            },
        )


def create_ood_detector(
    method: str,
    **kwargs,
) -> Any:
    """Create an OOD detector by name.

    Args:
        method: Method name
        **kwargs: Method-specific arguments

    Returns:
        OOD detector instance
    """
    if method == "mahalanobis":
        return MahalanobisOODDetector(**kwargs)
    elif method == "isolation_forest":
        return IsolationForestDetector(**kwargs)
    elif method == "energy":
        return EnergyOODDetector(**kwargs)
    elif method == "knn":
        return KNNDensityDetector(**kwargs)
    elif method == "ensemble":
        return EnsembleOODDetector(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
