"""SAE (Sparse Autoencoder) tools for feature discovery in world models.

This module provides:
- SAE architecture with TopK ReLU sparsity
- Training utilities with L0 regularization
- Evaluation metrics (reconstruction, sparsity, interpretability)
"""

from world_model_lens.sae.trainer import SAETrainer, SAETrainingResult, TopKReLU
from world_model_lens.sae.evaluator import SAEEvaluator, SAEFeature, SAEFeatureAnalysis
from world_model_lens.sae.sae_feature_circuits import SAEFeatureCircuitAnalyzer, FeatureCircuitGraph

__all__ = [
    "SAETrainer",
    "SAETrainingResult",
    "TopKReLU",
    "SAEEvaluator",
    "SAEFeature",
    "SAEFeatureAnalysis",
    "SAEFeatureCircuitAnalyzer",
    "FeatureCircuitGraph",
]
