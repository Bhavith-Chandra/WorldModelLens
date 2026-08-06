import torch
import pytest
import copy
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens import HookedWorldModel

def test_ijepa_adapter_architectural_init():
    config = WorldModelConfig(
        backend="ijepa",
        d_embed=128,
        predictor_embed_dim=256,
        predictor_heads=8,
        n_layers=2,
        predictor_depth=2
    )
    adapter = IJEPAAdapter(config)
    
    # Check encoders
    assert adapter.context_encoder.patch_embed.proj.out_channels == 128
    assert len(adapter.context_encoder.blocks) == 2
    
    # Check target encoder (should be deepcopy)
    assert not any(p.requires_grad for p in adapter.target_encoder.parameters())
    
    # Check predictor bottleneck
    assert adapter.predictor.predictor_embed.out_features == 256
    assert adapter.predictor.predictor_project_back.out_features == 128
    assert len(adapter.predictor.blocks) == 2
    assert adapter.predictor.blocks[0].attn.num_heads == 8

def test_ijepa_structured_masking():
    config = WorldModelConfig(backend="ijepa", d_embed=128)
    adapter = IJEPAAdapter(config)
    
    grid_size = 14
    num_patches = 196
    context_ids, target_ids_list = adapter._get_structured_masks(num_patches, grid_size)
    
    assert len(target_ids_list) == 4
    
    def _is_rectangular(ids, grid_size):
        if not ids: return False
        rows = [i // grid_size for i in ids]
        cols = [i % grid_size for i in ids]
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        expected_size = (row_max - row_min + 1) * (col_max - col_min + 1)
        if len(ids) != expected_size: return False
        expected_set = {r * grid_size + c for r in range(row_min, row_max + 1) for c in range(col_min, col_max + 1)}
        return set(ids) == expected_set

    # Verify each target block is a contiguous rectangle
    for target_ids in target_ids_list:
        assert len(target_ids) > 0
        assert _is_rectangular(target_ids, grid_size), f"Target block {target_ids} is not rectangular"
        
    assert len(context_ids) > 0
    
    # Ensure no overlap between ANY target block and the context
    all_targets = set()
    for block in target_ids_list:
        all_targets.update(block)
    
    context_set = set(context_ids)
    assert context_set.isdisjoint(all_targets), "Context and target blocks overlap"

def test_ijepa_compute_loss():
    # Use predictor_embed_dim divisible by default 6 heads
    config = WorldModelConfig(backend="ijepa", d_embed=128, predictor_embed_dim=192)
    adapter = IJEPAAdapter(config)
    
    obs = torch.randn(2, 3, 224, 224)
    loss = adapter.compute_loss(obs)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # Scalar
    assert loss.requires_grad is True # Gradients should flow through context_encoder

def test_target_encoder_receives_no_gradients():
    """Verify the most important behavioral invariant: no gradients flow to target encoder."""
    config = WorldModelConfig(backend="ijepa", d_embed=128, predictor_embed_dim=192)
    adapter = IJEPAAdapter(config)
    obs = torch.randn(2, 3, 224, 224)
    loss = adapter.compute_loss(obs)
    loss.backward()
    for name, p in adapter.target_encoder.named_parameters():
        assert p.grad is None, f"Target encoder parameter {name} received gradients"
    # Ensure context encoder DOES receive gradients
    assert adapter.context_encoder.pos_embed.grad is not None

def test_ijepa_ema_update():
    config = WorldModelConfig(backend="ijepa", d_embed=128)
    adapter = IJEPAAdapter(config)
    
    # Verify they start identical (deepcopy guarantee)
    assert torch.allclose(adapter.context_encoder.pos_embed, adapter.target_encoder.pos_embed)
    
    # Manually change context encoder
    with torch.no_grad():
        adapter.context_encoder.pos_embed.add_(1.0)
        
    old_target_pos = adapter.target_encoder.pos_embed.clone()
    
    adapter.update_target_encoder(momentum=0.9)
    # New target = 0.9 * old + 0.1 * (old + 1.0) = old + 0.1
    assert torch.allclose(adapter.target_encoder.pos_embed, old_target_pos + 0.1, atol=1e-5)

def test_ijepa_hooked_integration():
    config = WorldModelConfig(backend="ijepa", d_embed=128)
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter=adapter, config=config)
    
    obs = torch.randn(1, 3, 224, 224)
    _, _, cache = wm.run_with_cache(obs)
    
    # Verify target encoding appeared in cache (if implemented in HookedWorldModel)
    if "target_encoding" in cache.component_names:
        assert cache["target_encoding", 0].shape == (196, 128)
    
    assert "z_posterior" in cache.component_names
    latent = cache["z_posterior", 0]
    # Ranges explained:
    # Context samples 80-90% of 196 patches (157-176)
    # Then subtracts 4 target blocks of ~15-20% area (29-39 each)
    # Minimum valid context (50% retention) is around 78.
    # We assert a safe range for the masking strategy.
    assert 50 < latent.shape[0] < 185
    assert latent.shape[1] == 128
