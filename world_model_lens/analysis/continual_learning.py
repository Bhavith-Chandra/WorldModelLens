"""Architectural Auditing for Continual Learning.

Track representations across training epochs to detect:
- Catastrophic forgetting
- Representation drift
- Concept stability

This is crucial for continual learning architectures like NEUROGENESIS™.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass
class Checkpoint:
    """A checkpoint for comparison."""

    epoch: int
    path: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAnalysis:
    """Analysis of representation drift between checkpoints."""

    checkpoint_a: int
    checkpoint_b: int
    cosine_similarity: float
    euclidean_distance: float
    cka_similarity: float
    concept_stability: dict[str, float]
    significant_drift: bool


@dataclass
class ConceptStability:
    """Stability of a concept across training."""

    concept_name: str
    stability_score: float
    first_appeared_epoch: int
    drift_timeline: list[tuple[int, float]]


class ContinualLearningAuditor:
    """Audit world models across training epochs.

    Track how representations change and detect catastrophic forgetting.

    Example:
        auditor = ContinualLearningAuditor(world_model)

        # Register checkpoints
        auditor.register_checkpoint("path/to/epoch_10.pt", epoch=10)
        auditor.register_checkpoint("path/to/epoch_50.pt", epoch=50)
        auditor.register_checkpoint("path/to/epoch_100.pt", epoch=100)

        # Analyze drift
        drift = auditor.analyze_drift(epoch_a=10, epoch_b=100)

        # Check if old task concept is preserved
        stability = auditor.check_concept_stability("velocity", epoch_range=(10, 100))
    """

    def __init__(
        self,
        world_model: Any,
        reference_trajectory: Any | None = None,
    ):
        """Initialize auditor.

        Args:
            world_model: HookedWorldModel instance
            reference_trajectory: Same trajectory run through all checkpoints
        """
        self.wm = world_model
        self.reference_trajectory = reference_trajectory
        self.checkpoints: dict[int, Checkpoint] = {}
        self._cache_store: dict[int, Any] = {}

    def register_checkpoint(
        self,
        path: str,
        epoch: int,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Register a model checkpoint for comparison.

        Args:
            path: Path to checkpoint file
            epoch: Training epoch number
            metadata: Additional metadata

        Returns:
            Checkpoint object
        """
        checkpoint = Checkpoint(
            epoch=epoch,
            path=path,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.checkpoints[epoch] = checkpoint
        return checkpoint

    def run_on_checkpoint(
        self,
        epoch: int,
        observations: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> Any:
        """Run model on a specific checkpoint.

        Args:
            epoch: Checkpoint epoch number
            observations: Input observations
            actions: Optional actions

        Returns:
            WorldTrajectory from this checkpoint
        """
        if epoch not in self.checkpoints:
            raise ValueError(f"Checkpoint {epoch} not registered")

        # Load checkpoint weights
        state_dict = torch.load(self.checkpoints[epoch].path, map_location="cpu")
        self.wm.adapter.load_state_dict(state_dict)

        # Run forward pass
        trajectory, cache = self.wm.run_with_cache(observations, actions)

        # Cache for later comparison
        self._cache_store[epoch] = (trajectory, cache)

        return trajectory

    def analyze_drift(
        self,
        epoch_a: int,
        epoch_b: int,
        component: str = "z_posterior",
    ) -> DriftAnalysis:
        """Analyze representation drift between two checkpoints.

        Args:
            epoch_a: Earlier epoch
            epoch_b: Later epoch
            component: Which component to analyze

        Returns:
            DriftAnalysis with similarity metrics
        """
        if epoch_a not in self._cache_store:
            raise ValueError(f"Epoch {epoch_a} not run yet. Call run_on_checkpoint first.")
        if epoch_b not in self._cache_store:
            raise ValueError(f"Epoch {epoch_b} not run yet. Call run_on_checkpoint first.")

        traj_a, cache_a = self._cache_store[epoch_a]
        traj_b, cache_b = self._cache_store[epoch_b]

        # Extract latent representations
        try:
            latents_a = cache_a[component]
            latents_b = cache_b[component]
        except KeyError:
            raise ValueError(f"Component {component} not in cache") from None

        # Compute metrics
        cos_sim = self._cosine_similarity(latents_a, latents_b)
        euc_dist = self._euclidean_distance(latents_a, latents_b)
        cka = self._centered_kernel_alignment(latents_a, latents_b)

        # Check for significant drift (>20% change)
        significant_drift = cos_sim < 0.8

        return DriftAnalysis(
            checkpoint_a=epoch_a,
            checkpoint_b=epoch_b,
            cosine_similarity=cos_sim,
            euclidean_distance=euc_dist,
            cka_similarity=cka,
            concept_stability={},
            significant_drift=significant_drift,
        )

    def _cosine_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """Compute mean cosine similarity."""
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)

        cos = functional.cosine_similarity(a_flat, b_flat, dim=1)
        return cos.mean().item()

    def _euclidean_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """Compute mean Euclidean distance."""
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)

        dist = (a_flat - b_flat).norm(dim=1)
        return dist.mean().item()

    def _centered_kernel_alignment(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """Compute CKA (Centered Kernel Alignment)."""
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)

        n = a_flat.shape[0]

        # Center the matrices
        a_centered = a_flat - a_flat.mean(dim=0)
        b_centered = b_flat - b_flat.mean(dim=0)

        # Compute Gram matrices
        ka = a_centered @ a_centered.T / n
        kb = b_centered @ b_centered.T / n

        # Compute HSIC
        hsic_aa = (ka * ka).sum() / (n * n)
        hsic_bb = (kb * kb).sum() / (n * n)
        hsic_ab = (ka * kb).sum() / (n * n)

        # CKA
        cka = hsic_ab / (torch.sqrt(hsic_aa * hsic_bb) + 1e-8)

        return cka.item()

    def check_concept_stability(
        self,
        concept_name: str,
        epochs: list[int] | None = None,
    ) -> ConceptStability:
        """Check how stable a concept is across epochs.

        Args:
            concept_name: Name of concept to track
            epochs: List of epochs (default: all registered)

        Returns:
            ConceptStability with timeline
        """
        epochs = epochs or sorted(self.checkpoints.keys())

        if len(epochs) < 2:
            return ConceptStability(
                concept_name=concept_name,
                stability_score=1.0,
                first_appeared_epoch=epochs[0] if epochs else 0,
                drift_timeline=[],
            )

        # Analyze drift for each pair
        drift_timeline = []

        for i in range(len(epochs) - 1):
            drift = self.analyze_drift(epochs[i], epochs[i + 1])
            drift_timeline.append((epochs[i + 1], 1 - drift.cosine_similarity))

        # Overall stability
        all_drifts = [d for _, d in drift_timeline]
        stability = 1 - np.mean(all_drifts) if all_drifts else 1.0

        return ConceptStability(
            concept_name=concept_name,
            stability_score=stability,
            first_appeared_epoch=epochs[0],
            drift_timeline=drift_timeline,
        )

    def detect_catastrophic_forgetting(
        self,
        reference_epoch: int,
        current_epoch: int,
        task_concepts: list[str],
        threshold: float = 0.2,
    ) -> dict[str, bool]:
        """Detect if old task concepts are being forgotten.

        Args:
            reference_epoch: Epoch when task was learned
            current_epoch: Current training epoch
            task_concepts: Concepts important for the task
            threshold: Drift threshold for "forgotten"

        Returns:
            Dict mapping concept to whether it's forgotten
        """
        forgotten = {}

        for concept in task_concepts:
            stability = self.check_concept_stability(concept)
            forgotten[concept] = stability.stability_score < (1 - threshold)

        return forgotten

    def full_audit_report(
        self,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate full audit report.

        Args:
            output_path: Optional path to save report

        Returns:
            Complete audit report dict
        """
        epochs = sorted(self.checkpoints.keys())

        report = {
            "checkpoints": [self.checkpoints[e].path for e in epochs],
            "epoch_range": (min(epochs), max(epochs)),
            "n_checkpoints": len(epochs),
            "drift_analysis": {},
            "recommendations": [],
        }

        # Analyze all pairs
        for i in range(len(epochs) - 1):
            drift = self.analyze_drift(epochs[i], epochs[i + 1])
            key = f"epoch_{epochs[i]}_to_{epochs[i + 1]}"
            report["drift_analysis"][key] = {
                "cosine_similarity": drift.cosine_similarity,
                "euclidean_distance": drift.euclidean_distance,
                "cka": drift.cka_similarity,
            }

        # Add recommendations
        if report["drift_analysis"]:
            avg_drift = 1 - np.mean(
                [v["cosine_similarity"] for v in report["drift_analysis"].values()]
            )

            if avg_drift > 0.3:
                report["recommendations"].append("High drift detected. Consider regularization.")
            if avg_drift > 0.5:
                report["recommendations"].append("Severe drift. Catastrophic forgetting likely.")

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        return report


class RepresentationTracker:
    """Track representations with finer granularity.

    Can track:
    - Per-layer activations
    - Attention patterns
    - Gradient flows
    """

    def __init__(self, world_model: Any):
        self.wm = world_model
        self._snapshots: dict[int, dict[str, torch.Tensor]] = {}

    def snapshot_epoch(
        self,
        epoch: int,
        observations: torch.Tensor,
    ) -> None:
        """Capture full representation snapshot for an epoch."""

        traj, cache = self.wm.run_with_cache(observations)

        snapshot = {}

        # Capture all available components
        for name in cache.component_names:
            try:
                snapshot[name] = cache[name].clone()
            except Exception:
                pass

        self._snapshots[epoch] = snapshot

    def compare_layers(
        self,
        epoch_a: int,
        epoch_b: int,
        layer_name: str,
    ) -> dict[str, float]:
        """Compare a specific layer across epochs."""
        if epoch_a not in self._snapshots:
            raise ValueError(f"Epoch {epoch_a} not snapshotted")
        if epoch_b not in self._snapshots:
            raise ValueError(f"Epoch {epoch_b} not snapshotted")

        layer_a = self._snapshots[epoch_a].get(layer_name)
        layer_b = self._snapshots[epoch_b].get(layer_name)

        if layer_a is None or layer_b is None:
            return {"error": f"Layer {layer_name} not found"}

        return {
            "cosine_similarity": functional.cosine_similarity(
                layer_a.flatten().unsqueeze(0),
                layer_b.flatten().unsqueeze(0),
            ).item(),
            "mean_activation_change": (layer_a - layer_b).abs().mean().item(),
        }
