import pytest
import torch
import numpy as np

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.analysis.pvl import PhysicalVariableAnalyzer
from experiments.pvl.observables import (
    compute_patch_properties,
    LinearProbe,
    MLPProbe,
    train_probes_for_dataset,
)

@pytest.fixture
def dummy_wm():
    config = WorldModelConfig(
        backend="ijepa", d_embed=32, n_layers=2, n_heads=2, predictor_embed_dim=32,
        img_size=64, patch_size=16, predictor_heads=2, predictor_depth=2
    )
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter, config)
    return wm

@pytest.fixture
def dummy_image_tensor():
    return torch.randn(1, 3, 64, 64)

def test_compute_patch_properties(dummy_image_tensor):
    props = compute_patch_properties(dummy_image_tensor, patch_idx=0)
    expected_keys = {
        "brightness", "contrast", "complexity", "grid_y", "grid_x",
        "radial_distance", "aspect_ratio_proxy", "local_entropy",
        "color_saliency", "edge_direction"
    }
    assert expected_keys.issubset(props.keys())
    for k in expected_keys:
        assert isinstance(props[k], float)

def test_linear_probe():
    X = torch.randn(20, 32)
    y = torch.randn(20)
    probe = LinearProbe()
    probe.fit(X, y)
    preds = probe.predict(X)
    assert preds.shape == (20,)
    assert not torch.isnan(preds).any()

def test_mlp_probe():
    X = torch.randn(20, 32)
    y = torch.randn(20)
    probe = MLPProbe(hidden_dim=16)
    probe.fit(X, y, epochs=10)
    preds = probe.predict(X)
    assert preds.shape == (20,)
    assert not torch.isnan(preds).any()

def test_train_probes_for_dataset(dummy_image_tensor):
    X = torch.randn(16, 32)
    metadata = [
        {"image_idx": 0, "patch_idx": i, "grid_y": i // 4, "grid_x": i % 4}
        for i in range(16)
    ]
    image_tensors = [dummy_image_tensor]

    probes_linear = train_probes_for_dataset(X, metadata, image_tensors, probe_type="linear")
    assert len(probes_linear) == 10
    assert "radial_distance" in probes_linear
    assert "aspect_ratio_proxy" in probes_linear

    probes_mlp = train_probes_for_dataset(X, metadata, image_tensors, probe_type="mlp")
    assert len(probes_mlp) == 10
    assert isinstance(probes_mlp["radial_distance"], MLPProbe)

def test_physical_variable_analyzer(dummy_wm, dummy_image_tensor):
    analyzer = PhysicalVariableAnalyzer(dummy_wm)
    X = torch.randn(20, 32)
    
    # Test discover_subspaces
    dirs, vars_exp = analyzer.discover_subspaces(X, method="pca", n_components=3)
    assert dirs.shape == (3, 32)
    
    # Test fit_probes
    metadata = [
        {"image_idx": 0, "patch_idx": i % 16, "grid_y": (i % 16) // 4, "grid_x": (i % 16) % 4}
        for i in range(20)
    ]
    image_tensors = [dummy_image_tensor]
    probes = analyzer.fit_probes(X, metadata, image_tensors, probe_type="mlp")
    assert "radial_distance" in probes
