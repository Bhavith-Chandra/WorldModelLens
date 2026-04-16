import torch
import pytest
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.hooked_world_model import HookedWorldModel
from world_model_lens.core.forward_runner import ForwardRunner
from world_model_lens.core.activation_cache import ActivationCache

def test_ijepa_config_solidification():
    """Verify that WorldModelConfig has the new solidified parameters for I-JEPA."""
    config = WorldModelConfig(backend="ijepa")
    
    # Verify presence of explicitly added typed parameters
    assert hasattr(config, "patch_size")
    assert hasattr(config, "num_patches")
    assert hasattr(config, "embed_dim")
    assert hasattr(config, "context_mask_ratio")
    assert hasattr(config, "target_mask_scale")
    
    # Check values and types (Solidification)
    assert isinstance(config.patch_size, int)
    assert isinstance(config.num_patches, int)
    assert isinstance(config.context_mask_ratio, float)
    
    # Verify we can override them
    custom_config = WorldModelConfig(
        backend="ijepa",
        patch_size=32,
        num_patches=49,
        context_mask_ratio=0.5
    )
    assert custom_config.patch_size == 32
    assert custom_config.num_patches == 49
    assert custom_config.context_mask_ratio == 0.5

def test_ijepa_patch_axis_flow_and_keys():
    """Verify the Patch-Axis Observation Mode and standardized activation keys."""
    # 1. Setup I-JEPA Environment
    config = WorldModelConfig(
        backend="ijepa",
        world_model_family=WorldModelFamily.JEPA,
        patch_size=16,
        d_embed=192,
        predictor_depth=4 # Specifically testing for 4 layers
    )
    
    adapter = IJEPAAdapter(config)
    # We must ensure the adapter is in eval mode or works with random init
    wm = HookedWorldModel(adapter=adapter, config=config)
    runner = ForwardRunner(wm)
    
    # 2. Prepare mock spatial input [T=1, C=3, H=224, W=224]
    # ForwardRunner.run_forward(obs_seq) -> _run_forward_patch_axis(obs_batch)
    obs_seq = torch.randn(1, 3, 224, 224)
    actions = torch.zeros(1, 1)
    cache = ActivationCache()
    
    # 3. Execution
    traj = runner.run_forward(obs_seq, actions, cache, names_filter=None)
    
    # 4. Verifications: Patch-Axis Observation Mode
    # The trajectory length should match the number of target patches predicted
    # In IJEPAAdapter, the number of targets depends on masking logic.
    # We just need to verify it produced states unrolled spatially.
    assert len(traj.states) > 0, "Trajectory should contain spatial patch states"
    
    # 5. Verifications: Standardized Activation Keys (General)
    assert "encoder.out" in cache.component_names, "Should cache context encoder output"
    assert "target_encoder.out" in cache.component_names, "Should cache EMA target encoder output"
    assert "predictor.final" in cache.component_names, "Should cache final predictor output"
    
    # 6. Verifications: Specific Predictor Layer Indices
    # The user requested both general and specific layer indices verification.
    for i in range(config.predictor_depth):
        key = f"predictor.layer_{i}"
        assert key in cache.component_names, f"Should cache intermediate predictor layer: {key}"
        
        # Verify shape: [B, N_tokens, D_predictor]
        # Use simple indexing to get the tensor (stacked or at t=0)
        val = cache[key, 0]
        assert val.dim() == 3, f"Activation {key} should be [B, N, D]"
        assert val.shape[0] == 1, "Batch size should match input"

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
