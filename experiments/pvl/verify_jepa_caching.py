import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily
from examples.ijepa.image_utils import get_sample_image, preprocess_image

def test_cache():
    print("Loading I-JEPA mini model...")
    config = WorldModelConfig(
        backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=192,
        world_model_family=WorldModelFamily.JEPA
    )
    adapter = IJEPAAdapter(config)
    
    # Load default weights if available
    weights_path = "ijepa_mini.pth"
    if os.path.exists(weights_path):
        adapter.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
        print("Loaded weights successfully.")
    else:
        print("Weights not found. Using random weights.")
    
    adapter.eval()
    wm = HookedWorldModel(adapter, config)
    
    # Load and preprocess sample image
    # Use local Cat image
    img_path = "utils/img/Cat_November_2010-1a.jpg"
    img = get_sample_image(img_path)
    img_tensor = preprocess_image(img)
    
    print("Running forward with cache...")
    world_traj, latent_traj, cache = wm.run_with_cache(img_tensor)
    
    print("\nCached component names:")
    for name in sorted(cache.component_names):
        print(f"  - {name}: shape {cache[name, 0].shape if (name, 0) in cache else 'No t=0'}")

if __name__ == "__main__":
    test_cache()
