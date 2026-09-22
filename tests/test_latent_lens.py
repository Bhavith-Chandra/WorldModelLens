import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any

from world_model_lens.analysis.latent_lens import LatentLensAnalyzer


def test_latent_lens_analyzer_5fold_probing():
    """Test 5-fold cross-validated linear probing with shuffled-label empirical null controls."""
    # Synthetic feature matrices across 4 layers (N=20 samples, D=16 features)
    rng = np.random.RandomState(42)
    N, D = 20, 16
    
    # Target property: y is linearly encoded strongly in layer 3, weakly in layer 0
    y = rng.randn(N)
    
    X_layers = {
        0: rng.randn(N, D),
        1: rng.randn(N, D),
        2: rng.randn(N, D),
        3: np.outer(y, np.ones(D)) + 0.1 * rng.randn(N, D) # Strong linear encoding in layer 3
    }
    
    y_properties = {
        "spatial_grid_y": y,
        "texture_laplacian": rng.randn(N)
    }

    # Dummy model for analyzer init
    class DummyAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

    class DummyWM:
        def __init__(self):
            self.adapter = DummyAdapter()

    analyzer = LatentLensAnalyzer(DummyWM())
    
    probe_results = analyzer.evaluate_5fold_probe_emergence(X_layers, y_properties, n_splits=5)
    
    assert "spatial_grid_y" in probe_results
    assert "texture_laplacian" in probe_results
    
    l3_res = probe_results["spatial_grid_y"][3]
    assert "test_r2_5fold_mean" in l3_res
    assert "null_r2_5fold_mean" in l3_res
    assert "p_fdr_task6" in l3_res
    
    # Layer 3 should have high R2 significantly exceeding null
    assert l3_res["test_r2_5fold_mean"] > l3_res["null_r2_5fold_mean"]
    assert l3_res["significant_emergence"] is True


def test_latent_lens_aggregate_dataset_trajectories():
    """Test trajectory aggregation, Spearman trend test, and pre-registered success criteria."""
    class DummyAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

    class DummyWM:
        def __init__(self):
            self.adapter = DummyAdapter()

    analyzer = LatentLensAnalyzer(DummyWM())

    # Mock sample trajectories (N=10 samples) showing increasing Cosine Similarity across 4 Predictor layers
    sample_trajectories = [
        {
            "trajectory": [
                {"layer": 0, "layer_name": "Predictor Block 0", "cosine_similarity": 0.20, "mse": 1.40, "norm_ratio_diagnostic": 0.95},
                {"layer": 1, "layer_name": "Predictor Block 1", "cosine_similarity": 0.45, "mse": 1.10, "norm_ratio_diagnostic": 0.98},
                {"layer": 2, "layer_name": "Predictor Block 2", "cosine_similarity": 0.68, "mse": 0.85, "norm_ratio_diagnostic": 1.01},
                {"layer": 3, "layer_name": "Predictor Block 3", "cosine_similarity": 0.85, "mse": 0.60, "norm_ratio_diagnostic": 1.00},
            ]
        }
        for _ in range(10)
    ]

    agg = analyzer.aggregate_dataset_trajectories(sample_trajectories)
    
    assert agg["n_samples"] == 10
    assert len(agg["layers"]) == 4
    
    eval_res = agg["pre_registered_eval"]
    assert eval_res["spearman_trend_rho"] == 1.0 # Perfect monotonic rank trend across the 4 layer means
    assert eval_res["spearman_trend_pass"] is True
    assert eval_res["net_recovery_pass"] is True
    assert eval_res["overall_success_criterion_passed"] is True
    assert "SUCCESS" in eval_res["status"]
