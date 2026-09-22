import pytest
import torch
import numpy as np

from world_model_lens import HookedWorldModel
from world_model_lens.backends.mae_adapter import MAEAdapter, MAEDecoder
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily


def test_mae_decoder_dimensions():
    """Test MAEDecoder pixel reconstruction shape."""
    decoder = MAEDecoder(
        num_patches=196, patch_size=16, in_chans=3,
        encoder_embed_dim=192, decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8
    )
    
    ctx_latents = torch.randn(2, 100, 192) # [B=2, N_ctx=100, D_enc=192]
    context_ids = list(range(100))
    target_ids = list(range(100, 150))
    
    pixels_out = decoder(ctx_latents, context_ids, target_ids)
    
    # Expected output shape: [B=2, N_tgt=50, 16*16*3 = 768 RGB pixels]
    assert pixels_out.shape == (2, 50, 768)


def test_mae_adapter_forward_and_hooks():
    """Test MAEAdapter forward pass, context encoding, pixel decoding, and post-residual hooks."""
    config = WorldModelConfig(
        backend="mae",
        d_embed=192,
        n_layers=6,
        n_heads=3,
        world_model_family=WorldModelFamily.JEPA
    )
    
    adapter = MAEAdapter(config)
    adapter.eval()
    
    wm = HookedWorldModel(adapter, config)
    
    img_tensor = torch.randn(1, 3, 224, 224)
    context_ids = list(range(150))
    target_ids = list(range(150, 196))
    
    adapter.last_context_ids = context_ids
    adapter.last_target_ids = target_ids
    
    captured_acts = {}
    def hook_fn(act, hook):
        captured_acts["hook_resid_post"] = act.clone()
        return act
        
    hook_handle = adapter.add_hook("decoder.blocks.0.hook_resid_post", hook_fn)
    
    try:
        ctx_latents, _ = adapter.encode(img_tensor)
        assert ctx_latents.shape == (1, 150, 192)
        
        pixel_preds = adapter.dynamics(ctx_latents)
        assert pixel_preds.shape == (1, 46, 768)
        
        # Check hook interception
        assert "hook_resid_post" in captured_acts
        assert captured_acts["hook_resid_post"].shape == (1, 196, 512) # [B=1, N_ctx + N_tgt, D_dec=512]
    finally:
        adapter.remove_hook(hook_handle)
