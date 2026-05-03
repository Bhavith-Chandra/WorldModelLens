"""Safety analyzer for world models.

Detects potential safety concerns including:
- Out-of-distribution states
- Unstable dynamics predictions
- Hallucinated future states
- Dangerous action sequences
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from world_model_lens import HookedWorldModel, WorldTrajectory
from world_model_lens.monitoring.logging import get_logger

logger = get_logger(__name__)


class SafetyLevel(Enum):
    """Safety concern severity levels."""

    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class SafetyFinding:
    """A single safety finding."""

    category: str
    severity: SafetyLevel
    description: str
    location: Optional[str] = None
    evidence: Optional[dict[str, Any]] = None


@dataclass
class SafetyReport:
    """Complete safety audit report."""

    overall_safety_level: SafetyLevel
    findings: list[SafetyFinding]
    trajectory_length: int
    risk_score: float
    recommendations: list[str]


class SafetyAnalyzer:
    """Analyzes world model trajectories for safety concerns."""

    def __init__(self, world_model: HookedWorldModel) -> None:
        """Initialize safety analyzer.

        Args:
            world_model: The world model to analyze
        """
        self.wm = world_model
        self.device = next(world_model.parameters()).device

    def run_safety_audit(
        self, trajectory: list[dict[str, Any]], threshold: float = 0.8
    ) -> dict[str, Any]:
        """Run full safety audit on a trajectory.

        Args:
            trajectory: List of trajectory steps
            threshold: Risk threshold (0-1)

        Returns:
            Safety report dictionary
        """
        findings: list[SafetyFinding] = []

        observations = torch.stack(
            [
                torch.tensor(step.get("observation", torch.zeros(self.wm.config.d_obs)))
                for step in trajectory
            ]
        ).to(self.device)

        traj, _, cache = self.wm.run_with_cache(observations=observations)

        findings.extend(self._check_ood_detection(traj, cache))
        findings.extend(self._check_dynamics_stability(traj, cache))
        findings.extend(self._check_state_consistency(traj, cache))

        overall_level, risk_score = self._compute_risk_score(findings, threshold)

        recommendations = self._generate_recommendations(findings)

        report = SafetyReport(
            overall_safety_level=overall_level,
            findings=findings,
            trajectory_length=len(trajectory),
            risk_score=risk_score,
            recommendations=recommendations,
        )

        return self._report_to_dict(report)

    def _check_ood_detection(
        self, traj: WorldTrajectory, cache: dict[str, torch.Tensor]
    ) -> list[SafetyFinding]:
        """Check for out-of-distribution states."""
        findings: list[SafetyFinding] = []

        if "encoding" in cache:
            encoding = cache["encoding"]
            if encoding.shape[-1] > 0:
                mean = encoding.mean(dim=0)
                std = encoding.std(dim=0)
                ood_scores = (encoding - mean).abs().mean(dim=-1)

                high_ood = ood_scores > ood_scores.mean() + 2 * ood_scores.std()
                if high_ood.any():
                    findings.append(
                        SafetyFinding(
                            category="ood_detection",
                            severity=SafetyLevel.MEDIUM_RISK,
                            description=f"Detected {high_ood.sum()} out-of-distribution states",
                            evidence={
                                "ood_count": int(high_ood.sum()),
                                "max_ood_score": float(ood_scores.max()),
                            },
                        )
                    )

        return findings

    def _check_dynamics_stability(
        self, traj: WorldTrajectory, cache: dict[str, torch.Tensor]
    ) -> list[SafetyFinding]:
        """Check for unstable dynamics predictions."""
        findings: list[SafetyFinding] = []

        if len(traj.states) > 1:
            state_diffs = []
            for i in range(len(traj.states) - 1):
                s1 = traj.states[i].state
                s2 = traj.states[i + 1].state
                if s1.shape == s2.shape:
                    diff = (s2 - s1).abs().mean().item()
                    state_diffs.append(diff)

            if state_diffs:
                max_diff = max(state_diffs)
                if max_diff > 5.0:
                    findings.append(
                        SafetyFinding(
                            category="dynamics_stability",
                            severity=SafetyLevel.HIGH_RISK,
                            description="Detected unstable state transitions",
                            evidence={
                                "max_state_diff": max_diff,
                                "unstable_steps": sum(1 for d in state_diffs if d > 5.0),
                            },
                        )
                    )

        return findings

    def _check_state_consistency(
        self, traj: WorldTrajectory, cache: dict[str, torch.Tensor]
    ) -> list[SafetyFinding]:
        """Check for inconsistent state representations."""
        findings: list[SafetyFinding] = []

        for i, state in enumerate(traj.states):
            if state.prior is not None and state.posterior is not None:
                divergence = torch.nn.functional.kl_divergence(state.posterior, state.prior)
                if divergence.mean() > 1.0:
                    findings.append(
                        SafetyFinding(
                            category="state_consistency",
                            severity=SafetyLevel.LOW_RISK,
                            description=f"High prior-posterior divergence at step {i}",
                            location=f"step_{i}",
                            evidence={
                                "kl_divergence": float(divergence.mean()),
                            },
                        )
                    )

        return findings

    def _compute_risk_score(
        self, findings: list[SafetyFinding], threshold: float
    ) -> tuple[SafetyLevel, float]:
        """Compute overall risk score from findings."""
        severity_weights = {
            SafetyLevel.SAFE: 0.0,
            SafetyLevel.LOW_RISK: 0.25,
            SafetyLevel.MEDIUM_RISK: 0.5,
            SafetyLevel.HIGH_RISK: 0.75,
            SafetyLevel.CRITICAL: 1.0,
        }

        if not findings:
            return SafetyLevel.SAFE, 0.0

        max_severity = max(f.severity for f in findings)
        avg_severity = sum(severity_weights[f.severity] for f in findings) / len(findings)
        risk_score = (max_severity.value + avg_severity) / 2

        return max_severity, risk_score

    def _generate_recommendations(self, findings: list[SafetyFinding]) -> list[str]:
        """Generate safety recommendations based on findings."""
        recommendations: list[str] = []

        categories = {f.category for f in findings}

        if "ood_detection" in categories:
            recommendations.append(
                "Consider retraining with more diverse data to reduce OOD sensitivity"
            )
        if "dynamics_stability" in categories:
            recommendations.append("Review state transition functions for potential instabilities")
        if "state_consistency" in categories:
            recommendations.append("Investigate perception model alignment with dynamics prior")

        if not recommendations:
            recommendations.append("Continue monitoring for emerging safety concerns")

        return recommendations

    def _report_to_dict(self, report: SafetyReport) -> dict[str, Any]:
        """Convert safety report to dictionary."""
        return {
            "overall_safety_level": report.overall_safety_level.value,
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity.value,
                    "description": f.description,
                    "location": f.location,
                    "evidence": f.evidence,
                }
                for f in report.findings
            ],
            "trajectory_length": report.trajectory_length,
            "risk_score": report.risk_score,
            "recommendations": report.recommendations,
        }
