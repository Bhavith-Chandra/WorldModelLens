import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.hub.model_hub import ModelHub
from examples.ijepa.image_utils import get_sample_image, preprocess_image

def main():
    print("Loading ViT-H/14 checkpoint...")
    start_time = time.time()
    
    # Load using ModelHub helper which infers the config automatically
    adapter = ModelHub._load_ijepa("vith14_in1k_ep300.pth.tar", device="cpu")
    print(f"Loaded checkpoint in {time.time() - start_time:.2f} seconds.")
    print("Model Config:")
    print(f"  d_embed: {adapter.config.d_embed}")
    print(f"  n_layers: {adapter.config.n_layers}")
    print(f"  n_heads: {adapter.config.n_heads}")
    print(f"  predictor_embed_dim: {adapter.config.predictor_embed_dim}")
    print(f"  predictor_depth: {adapter.config.predictor_depth}")
    print(f"  predictor_heads: {adapter.config.predictor_heads}")
    
    wm = HookedWorldModel(adapter, adapter.config)
    
    # Load sample image
    raw_img = get_sample_image()
    img = preprocess_image(raw_img)
    
    # Run a forward pass
    print("Running forward pass...")
    # Define Target vs Context splits
    grid_size = 14
    core_ids = list(range(50, 60))
    bg_ids = list(range(0, 10))
    context_ids = list(range(20, 40))
    
    adapter.last_context_ids = context_ids
    adapter.last_target_ids = core_ids + bg_ids
    
    start_time = time.time()
    with torch.no_grad():
        h, _ = wm.adapter.encode(img)
        preds = wm.adapter.dynamics(h)
    print(f"Forward pass completed in {time.time() - start_time:.4f} seconds.")
    print(f"Prediction shape: {preds.shape}")

if __name__ == "__main__":
    main()
