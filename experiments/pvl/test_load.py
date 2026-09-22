import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))

from experiments.pvl.latent_collection import get_model_and_config

print("Testing model loading...")
weights_file = "vith14_in1k_ep300.pth.tar" if os.path.exists("vith14_in1k_ep300.pth.tar") else "ijepa_mini.pth"
print(f"Target weights file: {weights_file}")

try:
    wm, config = get_model_and_config(weights_file, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"SUCCESS! Loaded model config: d_embed={config.d_embed}, n_layers={config.n_layers}, predictor_depth={getattr(config, 'predictor_depth', 'N/A')}")
except Exception as e:
    import traceback
    print(f"FAILED to load model: {e}")
    traceback.print_exc()
