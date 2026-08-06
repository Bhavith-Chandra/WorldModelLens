"""Belief drift detection for world model trajectories.

This module provides tools to detect when world model beliefs become unstable:
- Sudden belief shifts (surprise peaks)
- Gradual belief drift over time
- Regime changes in latent space
- Concept drift detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.cluster import KMeans


@dataclass
class DriftResult:
    """Result of belief drift analysis."""

    drift_scores: torch.Tensor
    drift_points: list[int]
    drift_severity: float
    regime_changes: list[int]
    method: str
    details: dict[str, Any]


@dataclass
class DriftSegment:
    """A segment of trajectory with similar belief patterns."""

    start: int
    end: int
    mean_state: torch.Tensor
    variance: float
    dominant_cluster: int


class BeliefDriftDetector:
    """Detect belief drift in world model trajectories."""

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 2.0,
        method: str = "statistical",
    ):
        """Initialize detector.

        Args:
            window_size: Size of sliding window for comparison
            threshold: Z-score threshold for drift detection
            method: Detection method ("statistical", "kl", "clustering")
        """
        self.window_size = window_size
        self.threshold = threshold
        self.method = method

    def detect(
        self,
        trajectory: Any,
        component: str = "z_posterior",
    ) -> DriftResult:
        """Detect belief drift in trajectory.

        Args:
            trajectory: WorldTrajectory to analyze
            component: Which latent component to analyze

        Returns:
            DriftResult
        """
        states = [s.state for s in trajectory.states]

        if self.method == "statistical":
            return self._detect_statistical(states)
        elif self.method == "kl":
            return self._detect_kl(states)
        elif self.method == "clustering":
            return self._detect_clustering(states)
        else:
            return self._detect_statistical(states)

    def _detect_statistical(self, states: list[torch.Tensor]) -> DriftResult:
        """Detect drift using statistical measures.

        Args:
            states: List of latent states

        Returns:
            DriftResult
        """
        states_tensor = torch.stack(states)

        drift_scores = torch.zeros(len(states))

        for i in range(self.window_size, len(states)):
            window = states_tensor[i - self.window_size : i]
            current = states_tensor[i]

            window_mean = window.mean(dim=0)
            window_std = window.std(dim=0) + 1e-6

            z_score = ((current - window_mean) / window_std).abs().mean()
            drift_scores[i] = z_score

        drift_points = (drift_scores > self.threshold).nonero().tolist()

        drift_severity = drift_scores.mean().item()

        regime_changes = self._find_regime_changes(states_tensor)

        return DriftResult(
            drift_scores=drift_scores,
            drift_points=drift_points,
            drift_severity=drift_severity,
            regime_changes=regime_changes,
            method="statistical",
            details={
                "window_size": self.window_size,
                "threshold": self.threshold,
            },
        )

    def _detect_kl(self, states: list[torch.Tensor]) -> DriftResult:
        """Detect drift using KL divergence between windows.

        Args:
            states: List of latent states

        Returns:
            DriftResult
        """
        states_tensor = torch.stack(states)

        drift_scores = torch.zeros(len(states))

        for i in range(self.window_size, len(states)):
            window1 = states_tensor[i - self.window_size : i]
            window2 = (
                states_tensor[i : i + self.window_size]
                if i + self.window_size <= len(states)
                else states_tensor[i:]
            )

            p = torch.softmax(window1, dim=-1).mean(dim=0)
            q = torch.softmax(window2, dim=-1).mean(dim=0)

            kl = torch.nn.functional.kl_div(torch.log(q + 1e-8), p, reduction="batchmean")
            drift_scores[i] = kl

        drift_points = (drift_scores > self.threshold).nonero().tolist()

        drift_severity = drift_scores.mean().item()

        regime_changes = self._find_regime_changes(states_tensor)

        return DriftResult(
            drift_scores=drift_scores,
            drift_points=drift_points,
            drift_severity=drift_severity,
            regime_changes=regime_changes,
            method="kl_divergence",
            details={
                "window_size": self.window_size,
                "threshold": self.threshold,
            },
        )

    def _detect_clustering(self, states: list[torch.Tensor]) -> DriftResult:
        """Detect drift using clustering.

        Args:
            states: List of latent states

        Returns:
            DriftResult
        """
        states_tensor = torch.stack(states)

        n_clusters = min(5, len(states) // 2)

        if n_clusters < 2:
            return DriftResult(
                drift_scores=torch.zeros(len(states)),
                drift_points=[],
                drift_severity=0.0,
                regime_changes=[],
                method="clustering",
                details={"reason": "insufficient data"},
            )

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(states_tensor.cpu().numpy())

        drift_scores = torch.zeros(len(states))

        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                drift_scores[i] = 1.0

        drift_points = (drift_scores > 0).nonero().tolist()

        drift_severity = drift_scores.sum().item() / len(states)

        regime_changes = self._find_regime_changes(states_tensor)

        return DriftResult(
            drift_scores=drift_scores,
            drift_points=drift_points,
            drift_severity=drift_severity,
            regime_changes=regime_changes,
            method="clustering",
            details={
                "n_clusters": n_clusters,
                "cluster_labels": labels.tolist(),
            },
        )

    def _find_regime_changes(self, states: torch.Tensor) -> list[int]:
        """Find regime changes using cumulative sum.

        Args:
            states: Tensor of states [T, D]

        Returns:
            List of regime change indices
        """
        if len(states) < 10:
            return []

        mean = states.mean(dim=0)
        std = states.std(dim=0) + 1e-6

        normalized = (states - mean) / std

        cusum_pos = torch.zeros(len(states))
        cusum_neg = torch.zeros(len(states))

        for i in range(1, len(states)):
            change = (normalized[i] - normalized[i - 1]).abs().mean()
            cusum_pos[i] = max(0, cusum_pos[i - 1] + change - 0.5)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - change + 0.5)

        regime_changes = ((cusum_pos > 3) | (cusum_neg > 3)).nonero().tolist()

        return regime_changes

    def analyze_segments(self, trajectory: Any, n_segments: int = 5) -> list[DriftSegment]:
        """Analyze trajectory segments.

        Args:
            trajectory: Trajectory to analyze
            n_segments: Number of segments

        Returns:
            List of DriftSegment
        """
        states = [s.state for s in trajectory.states]
        states_tensor = torch.stack(states)

        segment_size = len(states) // n_segments
        segments = []

        for i in range(n_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(states))

            segment_states = states_tensor[start:end]

            segment = DriftSegment(
                start=start,
                end=end,
                mean_state=segment_states.mean(dim=0),
                variance=segment_states.var(dim=0).mean().item(),
                dominant_cluster=0,
            )
            segments.append(segment)

        return segments


class ConceptDriftDetector:
    """Detect concept drift in latent space using probing."""

    def __init__(self, probe: Any):
        """Initialize detector.

        Args:
            probe: Trained LatentProber
        """
        self.probe = probe

    def detect(
        self,
        trajectory: Any,
        concept: str,
    ) -> DriftResult:
        """Detect concept drift.

        Args:
            trajectory: Trajectory to analyze
            concept: Concept name

        Returns:
            DriftResult
        """

        if not hasattr(trajectory, "concept_tracking") or trajectory.concept_tracking is None:
            return DriftResult(
                drift_scores=torch.zeros(len(trajectory.states)),
                drift_points=[],
                drift_severity=0.0,
                regime_changes=[],
                method="concept",
                details={"error": "no concept tracking available"},
            )

        concept_probs = trajectory.concept_tracking.get(concept)
        if concept_probs is None:
            return DriftResult(
                drift_scores=torch.zeros(len(trajectory.states)),
                drift_points=[],
                drift_severity=0.0,
                regime_changes=[],
                method="concept",
                details={"error": "concept not found"},
            )

        probs_tensor = torch.tensor(concept_probs)

        entropy = -(probs_tensor * torch.log(probs_tensor + 1e-8)).sum(dim=-1)

        drift_scores = torch.zeros(len(trajectory.states))
        for i in range(1, len(entropy)):
            drift_scores[i] = (entropy[i] - entropy[i - 1]).abs()

        drift_threshold = drift_scores.mean() + 2 * drift_scores.std()
        drift_points = (drift_scores > drift_threshold).nonero().tolist()

        return DriftResult(
            drift_scores=drift_scores,
            drift_points=drift_points,
            drift_severity=drift_scores.mean().item(),
            regime_changes=drift_points,
            method="concept",
            details={"concept": concept},
        )
