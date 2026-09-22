"""Analysis modules for world model interpretability."""

from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.analysis.faithfulness import (
    FaithfulnessAnalyzer,
    AOPCResult,
    PerturbationResult,
)
from world_model_lens.analysis.attribution import (
    BaseAttribution,
    IntegratedGradientsAttribution,
    GradientXInputAttribution,
    SmoothGradAttribution,
    AttributionEvaluator,
    extract_attention_weights,
)
from world_model_lens.analysis.ablation_knockout import (
    PatchKnockoutEvaluator,
    compute_auc,
)
from world_model_lens.analysis.significance import (
    StatisticalSignificanceSuite,
    compute_bootstrap_ci,
    compute_paired_tests,
    compute_cohens_d,
    apply_multiple_comparisons_correction,
)

__all__ = [
    "BeliefAnalyzer",
    "FaithfulnessAnalyzer",
    "AOPCResult",
    "PerturbationResult",
    "BaseAttribution",
    "IntegratedGradientsAttribution",
    "GradientXInputAttribution",
    "SmoothGradAttribution",
    "AttributionEvaluator",
    "extract_attention_weights",
    "PatchKnockoutEvaluator",
    "compute_auc",
    "StatisticalSignificanceSuite",
    "compute_bootstrap_ci",
    "compute_paired_tests",
    "compute_cohens_d",
    "apply_multiple_comparisons_correction",
]
