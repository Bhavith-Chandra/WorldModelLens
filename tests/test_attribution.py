import pytest
import torch
import numpy as np

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.analysis.attribution import (
    IntegratedGradientsAttribution,
    GradientXInputAttribution,
    SmoothGradAttribution,
    AttributionEvaluator,
    extract_attention_weights,
)


@pytest.fixture
def dummy_wm():
    config = WorldModelConfig(
        backend="ijepa", d_embed=32, n_layers=2, n_heads=2, predictor_embed_dim=64,
        img_size=64, patch_size=16, predictor_heads=2
    )
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter, config)
    return wm


@pytest.fixture
def dummy_data():
    img_tensor = torch.randn(1, 3, 64, 64)
    # Total patches = (64/16)**2 = 16
    context_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    target_id = 15
    return img_tensor, context_ids, target_id


def test_integrated_gradients(dummy_wm, dummy_data):
    img_tensor, context_ids, target_id = dummy_data
    ig = IntegratedGradientsAttribution(dummy_wm.adapter, n_steps=5)
    
    attr = ig.compute(img_tensor, context_ids, target_id)
    
    assert isinstance(attr, np.ndarray)
    assert attr.shape == (len(context_ids),)
    assert not np.isnan(attr).any()


def test_gradient_x_input(dummy_wm, dummy_data):
    img_tensor, context_ids, target_id = dummy_data
    gxi = GradientXInputAttribution(dummy_wm.adapter)
    
    attr = gxi.compute(img_tensor, context_ids, target_id)
    
    assert isinstance(attr, np.ndarray)
    assert attr.shape == (len(context_ids),)
    assert not np.isnan(attr).any()


def test_smooth_grad(dummy_wm, dummy_data):
    img_tensor, context_ids, target_id = dummy_data
    sg = SmoothGradAttribution(dummy_wm.adapter, n_samples=3)
    
    attr = sg.compute(img_tensor, context_ids, target_id)
    
    assert isinstance(attr, np.ndarray)
    assert attr.shape == (len(context_ids),)
    assert not np.isnan(attr).any()


def test_attribution_evaluator(dummy_wm, dummy_data):
    img_tensor, context_ids, target_id = dummy_data
    evaluator = AttributionEvaluator(k=4)
    
    # Fake attention weights and attribution scores for testing logic
    attn_weights = np.array([0.1, 0.5, 0.2, 0.8, 0.05, 0.05, 0.15, 0.3])
    # Top 4 attn: index 3, 1, 7, 2
    attr_scores = np.array([0.2, 0.4, 0.3, 0.9, 0.1, 0.0, 0.2, 0.5])
    # Top 4 attr: index 3, 7, 1, 2
    
    # Overlap should be 4/4 = 1.0
    overlap = evaluator.compute_jaccard_overlap(attn_weights, attr_scores)
    assert np.isclose(overlap, 1.0)
    
    corr = evaluator.compute_rank_correlation(attn_weights, attr_scores)
    assert isinstance(corr, float)
    assert -1.0 <= corr <= 1.0
    
    # Test dataset evaluation
    ig = IntegratedGradientsAttribution(dummy_wm.adapter, n_steps=2)
    dataset = [(img_tensor, context_ids, target_id), (img_tensor, context_ids, target_id-1)]
    
    results = evaluator.evaluate_dataset(dummy_wm, ig, dataset)
    
    assert "mean_overlap" in results
    assert "mean_spearman" in results
    assert "failure_rate" in results
    assert "alignment_rate" in results
    assert len(results["raw_overlaps"]) == 2
