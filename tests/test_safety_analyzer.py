"""Tests for the safety analyzer."""

import torch
import pytest

from world_model_lens.safety.analyzer import (
    SafetyLevel,
    SafetyFinding,
    SafetyReport,
)


class TestSafetyLevel:
    """Tests for SafetyLevel enum."""

    def test_safety_levels_exist(self):
        """Test all safety levels are defined."""
        assert SafetyLevel.SAFE is not None
        assert SafetyLevel.LOW_RISK is not None
        assert SafetyLevel.MEDIUM_RISK is not None
        assert SafetyLevel.HIGH_RISK is not None
        assert SafetyLevel.CRITICAL is not None

    def test_safety_level_values(self):
        """Test safety level string values."""
        assert SafetyLevel.SAFE.value == "safe"
        assert SafetyLevel.HIGH_RISK.value == "high_risk"


class TestSafetyFinding:
    """Tests for SafetyFinding dataclass."""

    def test_finding_creation_minimal(self):
        """Test creating a safety finding with minimal fields."""
        finding = SafetyFinding(
            category="ood_detection",
            severity=SafetyLevel.MEDIUM_RISK,
            description="Out of distribution detected",
        )
        assert finding.category == "ood_detection"
        assert finding.severity == SafetyLevel.MEDIUM_RISK
        assert finding.location is None
        assert finding.evidence is None

    def test_finding_creation_full(self):
        """Test creating a safety finding with all fields."""
        finding = SafetyFinding(
            category="dynamics_stability",
            severity=SafetyLevel.HIGH_RISK,
            description="Unstable transition detected",
            location="step_10",
            evidence={"max_diff": 10.5, "unstable_steps": 3},
        )
        assert finding.category == "dynamics_stability"
        assert finding.severity == SafetyLevel.HIGH_RISK
        assert finding.location == "step_10"
        assert finding.evidence["max_diff"] == 10.5


class TestSafetyReport:
    """Tests for SafetyReport dataclass."""

    def test_report_creation_empty(self):
        """Test creating a safety report with no findings."""
        report = SafetyReport(
            overall_safety_level=SafetyLevel.SAFE,
            findings=[],
            trajectory_length=0,
            risk_score=0.0,
            recommendations=[],
        )
        assert report.overall_safety_level == SafetyLevel.SAFE
        assert len(report.findings) == 0
        assert report.risk_score == 0.0

    def test_report_creation_with_findings(self):
        """Test creating a safety report with findings."""
        findings = [
            SafetyFinding(
                category="ood",
                severity=SafetyLevel.LOW_RISK,
                description="Minor OOD",
            )
        ]
        report = SafetyReport(
            overall_safety_level=SafetyLevel.LOW_RISK,
            findings=findings,
            trajectory_length=10,
            risk_score=0.25,
            recommendations=["Monitor closely"],
        )
        assert report.overall_safety_level == SafetyLevel.LOW_RISK
        assert len(report.findings) == 1
        assert report.risk_score == 0.25
        assert report.recommendations == ["Monitor closely"]


class TestSafetyLevelOrdering:
    """Tests for safety level ordering."""

    def test_severity_comparison(self):
        """Test that severity levels can be compared."""
        safe = SafetyLevel.SAFE
        low = SafetyLevel.LOW_RISK
        medium = SafetyLevel.MEDIUM_RISK
        high = SafetyLevel.HIGH_RISK
        critical = SafetyLevel.CRITICAL

        assert safe != low
        assert low != medium
        assert medium != high
        assert high != critical
