import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.backends.vjepa_adapter import VJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily

def test_ijepa_from_checkpoint_strict():
    """Verify IJEPAAdapter.from_checkpoint with Meta ViT-H/14 weights."""
    weights_path = "vith14_in1k_ep300.pth.tar"
    assert os.path.exists(weights_path), f"{weights_path} not found"
    
    cfg = WorldModelConfig(
        backend="ijepa", patch_size=14, d_embed=1280, n_layers=32, n_heads=16,
        predictor_embed_dim=384, predictor_depth=12, predictor_heads=12,
        world_model_family=WorldModelFamily.JEPA
    )
    
    print("\n--- Testing IJEPAAdapter.from_checkpoint Strict Loading ---")
    adapter = IJEPAAdapter.from_checkpoint(weights_path, cfg)
    print("I-JEPA Strict Checkpoint Loading Test: PASSED")

def test_vjepa_from_checkpoint_strict():
    """Verify VJEPAAdapter.from_checkpoint with V-JEPA checkpoint."""
    vjepa_path = "vjepa_mini.pth"
    if not os.path.exists(vjepa_path):
        cfg_init = WorldModelConfig(backend="vjepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384, predictor_depth=4)
        adapter_init = VJEPAAdapter(cfg_init)
        torch.save(adapter_init.state_dict(), vjepa_path)
    
    cfg = WorldModelConfig(
        backend="vjepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384,
        predictor_depth=4, num_frames=16, tubelet_size=2, world_model_family=WorldModelFamily.JEPA
    )
    
    print("\n--- Testing VJEPAAdapter.from_checkpoint Strict Loading ---")
    adapter = VJEPAAdapter.from_checkpoint(vjepa_path, cfg)
    print("V-JEPA Strict Checkpoint Loading Test: PASSED")

if __name__ == "__main__":
    test_ijepa_from_checkpoint_strict()
    test_vjepa_from_checkpoint_strict()
