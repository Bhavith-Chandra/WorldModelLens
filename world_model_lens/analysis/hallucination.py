from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

"""Hallucination detection for world model predictions.

This module provides multiple methods to detect when world models produce
hallucinated or unrealistic predictions:
- Reconstruction-based: Compare reconstructions to actual observations
- Consistency-based: Check for sudden inconsistencies
- Physics-based: Verify physical constraints
- Uncertainty-based: Use prediction confidence
"""


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    scores: torch.Tensor
    is_hallucinating: torch.Tensor
    threshold: float
    method: str
    details: dict[str, Any]


class ReconstructionHallucinationDetector:
    """Detect hallucinations using reconstruction error."""

    def __init__(
        self,
        threshold: float = 0.5,
        method: str = "mse",
    ):
        """Initialize detector.

        Args:
            threshold: Error threshold for hallucination
            method: Error metric ("mse", "mae", "ssim")
        """
        self.threshold = threshold
        self.method = method

    def detect(
        self,
        observations: torch.Tensor,
        reconstructions: torch.Tensor,
    ) -> HallucinationResult:
        """Detect hallucinations using reconstruction error.

        Args:
            observations: Original observations [T, ...]
            reconstructions: Reconstructed observations [T, ...]

        Returns:
            HallucinationResult
        """
        if self.method == "mse":
            errors = functional.mse_loss(observations, reconstructions, reduction="none").mean(
                dim=list(range(1, observations.dim()))
            )
        elif self.method == "mae":
            errors = functional.l1_loss(observations, reconstructions, reduction="none").mean(
                dim=list(range(1, observations.dim()))
            )
        else:
            errors = functional.mse_loss(observations, reconstructions, reduction="none").mean(
                dim=list(range(1, observations.dim()))
            )

        is_hallucinating = errors > self.threshold

        return HallucinationResult(
            scores=errors,
            is_hallucinating=is_hallucinating,
            threshold=self.threshold,
            method="reconstruction",
            details={
                "mean_error": errors.mean().item(),
                "max_error": errors.max().item(),
            },
        )


class ConsistencyHallucinationDetector:
    """Detect hallucinations using temporal consistency."""

    def __init__(
        self,
        window_size: int = 5,
        threshold: float = 2.0,
    ):
        """Initialize detector.

        Args:
            window_size: Window for consistency check
            threshold: Z-score threshold
        """
        self.window_size = window_size
        self.threshold = threshold

    def detect(
        self,
        states: list[torch.Tensor],
    ) -> HallucinationResult:
        """Detect inconsistencies in state sequence.

        Args:
            states: List of latent states

        Returns:
            HallucinationResult
        """
        states_tensor = torch.stack(states)

        inconsistencies = torch.zeros(len(states))

        for i in range(1, len(states)):
            diff = (states_tensor[i] - states_tensor[i - 1]).abs()

            if i > self.window_size:
                window = states_tensor[i - self.window_size : i]
                mean = window.mean(dim=0)
                std = window.std(dim=0) + 1e-6
                z_score = ((diff - mean) / std).abs().mean()
                inconsistencies[i] = z_score

        mean_inc = inconsistencies.mean()
        std_inc = inconsistencies.std() + 1e-6
        threshold = mean_inc + self.threshold * std_inc

        is_hallucinating = inconsistencies > threshold

        return HallucinationResult(
            scores=inconsistencies,
            is_hallucinating=is_hallucinating,
            threshold=threshold,
            method="consistency",
            details={
                "window_size": self.window_size,
                "mean_inconsistency": mean_inc.item(),
            },
        )


class UncertaintyHallucinationDetector:
    """Detect hallucinations using prediction uncertainty."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
    ):
        """Initialize detector.

        Args:
            uncertainty_threshold: Uncertainty threshold
        """
        self.uncertainty_threshold = uncertainty_threshold

    def detect(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> HallucinationResult:
        """Detect hallucinations based on uncertainty.

        Args:
            predictions: Predicted next states [T, D]
            uncertainties: Uncertainty estimates [T]

        Returns:
            HallucinationResult
        """
        is_hallucinating = uncertainties > self.uncertainty_threshold

        return HallucinationResult(
            scores=uncertainties,
            is_hallucinating=is_hallucinating,
            threshold=self.uncertainty_threshold,
            method="uncertainty",
            details={
                "mean_uncertainty": uncertainties.mean().item(),
                "max_uncertainty": uncertainties.max().item(),
            },
        )


class PhysicsHallucinationDetector:
    """Detect hallucinations using physical constraints."""

    def __init__(
        self,
        constraints: list[Callable] | None = None,
    ):
        """Initialize detector.

        Args:
            constraints: List of constraint functions
        """
        self.constraints = constraints or []

    def add_constraint(
        self,
        name: str,
        constraint_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Add a physics constraint.

        Args:
            name: Constraint name
            constraint_fn: Function that returns violation scores
        """
        self.constraints.append((name, constraint_fn))

    def detect(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor | None] | None = None,
    ) -> HallucinationResult:
        """Detect physics violations.

        Args:
            states: List of latent states
            actions: Optional list of actions

        Returns:
            HallucinationResult
        """
        violation_scores = torch.zeros(len(states))

        for i, state in enumerate(states):
            total_violation = 0.0

            for _name, constraint_fn in self.constraints:
                if actions and i < len(actions):
                    violation = constraint_fn(state, actions[i])
                else:
                    violation = constraint_fn(state)
                total_violation += violation.abs().mean().item()

            violation_scores[i] = total_violation

        threshold = violation_scores.mean() + 2 * violation_scores.std()
        is_hallucinating = violation_scores > threshold

        return HallucinationResult(
            scores=violation_scores,
            is_hallucinating=is_hallucinating,
            threshold=threshold,
            method="physics",
            details={
                "num_constraints": len(self.constraints),
                "constraint_names": [c[0] for c in self.constraints],
            },
        )


class EnsembleHallucinationDetector:
    """Ensemble of multiple hallucination detectors."""

    def __init__(self, detectors: list[Any] | None = None):
        """Initialize ensemble.

        Args:
            detectors: List of detector instances
        """
        self.detectors = detectors or []

    def add_detector(self, detector: Any) -> None:
        """Add a detector to the ensemble.

        Args:
            detector: Detector instance
        """
        self.detectors.append(detector)

    def detect(
        self,
        **kwargs,
    ) -> HallucinationResult:
        """Detect hallucinations using ensemble.

        Returns:
            HallucinationResult
        """
        all_scores = []
        all_is_hall = []

        for detector in self.detectors:
            result = detector.detect(**kwargs)
            all_scores.append(result.scores)
            all_is_hall.append(result.is_hallucinating)

        scores = torch.stack(all_scores).mean(dim=0)
        is_hallucinating = torch.stack(all_is_hall).any(dim=0)

        return HallucinationResult(
            scores=scores,
            is_hallucinating=is_hallucinating,
            threshold=0.5,
            method="ensemble",
            details={
                "num_detectors": len(self.detectors),
                "methods": [
                    d.method if hasattr(d, "method") else "unknown" for d in self.detectors
                ],
            },
        )


def create_hallucination_detector(
    method: str,
    **kwargs,
) -> Any:
    """Create a hallucination detector by name.

    Args:
        method: Method name
        **kwargs: Method arguments

    Returns:
        Detector instance
    """
    if method == "reconstruction":
        return ReconstructionHallucinationDetector(**kwargs)
    elif method == "consistency":
        return ConsistencyHallucinationDetector(**kwargs)
    elif method == "uncertainty":
        return UncertaintyHallucinationDetector(**kwargs)
    elif method == "physics":
        return PhysicsHallucinationDetector(**kwargs)
    elif method == "ensemble":
        return EnsembleHallucinationDetector(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
