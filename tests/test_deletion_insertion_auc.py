import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List

from world_model_lens.analysis.ablation_knockout import (
    compute_auc,
    compute_bootstrap_ci,
    PatchKnockoutEvaluator,
)


def test_compute_auc_basic():
    """Test AUC calculation on simple linear curves."""
    k_values = [1, 3, 5, 10, 20]
    
    # Monotonically increasing curve (Deletion)
    mses_del = [0.1, 0.3, 0.5, 0.8, 1.0]
    auc_del = compute_auc(k_values, mses_del)
    assert auc_del > 0.0
    assert isinstance(auc_del, float)

    # Monotonically decreasing curve (Insertion)
    mses_ins = [1.0, 0.8, 0.5, 0.3, 0.1]
    auc_ins = compute_auc(k_values, mses_ins)
    assert auc_ins > 0.0
    
    # Monotonically increasing should have higher AUC than monotonically decreasing
    assert auc_del > auc_ins


def test_compute_bootstrap_ci():
    """Test bootstrap confidence interval computation."""
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    mean_val, low, high = compute_bootstrap_ci(data, n_bootstraps=100)
    
    assert np.isclose(mean_val, 0.55)
    assert low < mean_val < high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


class MockContextEncoder(nn.Module):
    def __init__(self, embed_dim=16, num_patches=200):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
    def patch_embed(self, x):
        # x is (B, 3, H, W)
        batch_size = x.shape[0]
        # Return deterministic dummy patch embeddings
        return torch.ones(batch_size, self.num_patches, self.embed_dim, device=x.device)

    def forward_blocks(self, ctx):
        return ctx


class MockTargetEncoder(nn.Module):
    def __init__(self, embed_dim=16, num_patches=200):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.ones(batch_size, self.num_patches, self.embed_dim, device=x.device)


class MockPredictor(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, ctx_latents, context_ids, target_ids):
        # Predict target by averaging context latents
        mean_ctx = ctx_latents.mean(dim=1, keepdim=True)
        return mean_ctx


class MockIJEPAAdapter(nn.Module):
    def __init__(self, embed_dim=16, num_patches=200):
        super().__init__()
        self.context_encoder = MockContextEncoder(embed_dim, num_patches)
        self.target_encoder = MockTargetEncoder(embed_dim, num_patches)
        self.predictor = MockPredictor(embed_dim)


def test_patch_knockout_evaluator_structure():
    """Test PatchKnockoutEvaluator initialization and parameters."""
    adapter = MockIJEPAAdapter()
    evaluator = PatchKnockoutEvaluator(
        adapter, 
        k_values=[1, 3, 5],
        patch_ablation_mode="zero",
        n_random_seeds=5
    )
    
    assert evaluator.k_values == [1, 3, 5]
    assert evaluator.patch_ablation_mode == "zero"
    assert evaluator.n_random_seeds == 5


def test_patch_knockout_aggregate_dataset_results():
    """Test aggregation logic across multiple sample results."""
    adapter = MockIJEPAAdapter()
    evaluator = PatchKnockoutEvaluator(adapter, k_values=[1, 3, 5], n_random_seeds=5)
    
    # Mock sample results
    sample_results = [
        {
            "baseline_mse": 0.05,
            "k_values": [1, 3, 5],
            "ig_deletion_mses": [0.1, 0.2, 0.4],
            "attn_deletion_mses": [0.08, 0.15, 0.3],
            "random_deletion_mses": [[0.05, 0.1, 0.2]] * 5,
            "ig_insertion_mses": [0.4, 0.2, 0.1],
            "attn_insertion_mses": [0.45, 0.25, 0.15],
            "random_insertion_mses": [[0.5, 0.3, 0.2]] * 5,
            "ig_deletion_auc": 0.3,
            "attn_deletion_auc": 0.2,
            "random_deletion_auc": [0.15] * 5,
            "ig_insertion_auc": 0.2,
            "attn_insertion_auc": 0.3,
            "random_insertion_auc": [0.35] * 5,
            "ig_random_deletion_zscore": 3.0,
            "ig_random_deletion_percentile": 1.0,
            "ig_random_insertion_zscore": -3.0,
            "ig_random_insertion_percentile": 0.0
        }
        for _ in range(10)
    ]
    
    agg = evaluator.aggregate_dataset_results(sample_results)
    
    assert agg["n_samples"] == 10
    assert agg["k_values"] == [1, 3, 5]
    assert "deletion_auc" in agg
    assert "insertion_auc" in agg
    assert "sample_outlier_summary" in agg
    
    outliers = agg["sample_outlier_summary"]
    assert outliers["frac_samples_ig_deletion_outlier_zscore"] == 1.0
    assert outliers["frac_samples_ig_deletion_outlier_percentile"] == 1.0
    assert outliers["frac_samples_ig_insertion_outlier_zscore"] == 1.0
    assert outliers["frac_samples_ig_insertion_outlier_percentile"] == 1.0
