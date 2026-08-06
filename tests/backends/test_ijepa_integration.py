import torch
import os
from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter

def test_ijepa_final_integration():
    # 1. Setup config
    config = WorldModelConfig(backend="ijepa", d_embed=128)
    
    # 2. Create a mock checkpoint with Meta-style keys
    mock_sd = {
        "model": {
            "patch_embed.proj.weight": torch.randn(128, 3, 16, 16),
            "patch_embed.proj.bias": torch.randn(128),
            "pos_embed": torch.randn(1, 196, 128),
            "blocks.0.norm1.weight": torch.randn(128),
            "blocks.0.norm1.bias": torch.randn(128),
            # ... and so on
        }
    }
    checkpoint_path = "mock_vith.pth"
    torch.save(mock_sd, checkpoint_path)
    
    try:
        # 3. Test HookedWorldModel.from_checkpoint
        wm = HookedWorldModel.from_checkpoint(checkpoint_path, backend="ijepa", config=config)
        assert isinstance(wm.adapter, IJEPAAdapter)
        print("HookedWorldModel.from_checkpoint('ijepa') PASSED")
        
        # 4. Test run_with_cache including target_encoding
        obs = torch.randn(2, 3, 224, 224)
        _, _, cache = wm.run_with_cache(obs)
        
        # Verify cache points
        assert "target_encoding" in cache.component_names
        print("target_encoding hook in ActivationCache PASSED")
        
        target_repr = cache["target_encoding", 0]
        assert target_repr.shape == (196, 128) # Grid size 14x14 = 196
        
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

if __name__ == "__main__":
    test_ijepa_final_integration()
