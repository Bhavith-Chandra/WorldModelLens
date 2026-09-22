import pytest
import torch
import numpy as np

from world_model_lens import HookedWorldModel
from world_model_lens.backends.vjepa_adapter import VJEPAAdapter, TubeletEmbed
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily


def test_tubelet_embed_dimensions():
    """Test 3D Conv3D tubelet embedding shape."""
    embed = TubeletEmbed(img_size=224, patch_size=16, num_frames=16, tubelet_size=2, in_chans=3, embed_dim=192)
    
    # 5D Video Tensor: [B=2, C=3, T=16, H=224, W=224]
    x_video = torch.randn(2, 3, 16, 224, 224)
    out = embed(x_video)
    
    # Expected spatiotemporal grid: T_grid = 16//2 = 8, H_grid = 224//16 = 14, W_grid = 224//16 = 14
    # Total tokens N = 8 * 14 * 14 = 1568
    assert out.shape == (2, 1568, 192)


def test_vjepa_adapter_forward_and_hooks():
    """Test VJEPAAdapter forward pass, context/target slicing, and post-residual hooks."""
    config = WorldModelConfig(
        backend="vjepa",
        d_embed=192,
        n_layers=6,
        n_heads=3,
        predictor_embed_dim=384,
        predictor_depth=4,
        num_frames=16,
        tubelet_size=2,
        world_model_family=WorldModelFamily.JEPA
    )
    
    adapter = VJEPAAdapter(config)
    adapter.eval()
    
    wm = HookedWorldModel(adapter, config)
    
    # 5D Video Tensor: [B=1, C=3, T=16, H=224, W=224]
    x_video = torch.randn(1, 3, 16, 224, 224)
    
    context_ids = list(range(200)) # 200 context tubelets
    target_ids = list(range(200, 250)) # 50 target tubelets
    
    adapter.last_context_ids = context_ids
    adapter.last_target_ids = target_ids
    
    captured_acts = {}
    def hook_fn(act, hook):
        captured_acts["hook_resid_post"] = act.clone()
        return act
        
    hook_handle = adapter.add_hook("predictor.blocks.0.hook_resid_post", hook_fn)
    
    try:
        ctx_latents, _ = adapter.encode(x_video)
        assert ctx_latents.shape == (1, 200, 192)
        
        target_preds = adapter.dynamics(ctx_latents)
        assert target_preds.shape == (1, 50, 192)
        
        # Check hook interception
        assert "hook_resid_post" in captured_acts
        assert captured_acts["hook_resid_post"].shape == (1, 250, 384) # [B=1, N_ctx + N_tgt, D_pred]
    finally:
        adapter.remove_hook("predictor.blocks.0.hook_resid_post")
