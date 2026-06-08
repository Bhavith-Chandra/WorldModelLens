"""Attention Routing Ablation (RQ5 extension)

Investigates whether the Attention mechanism is the primary engine routing
visible context patches together to infer the missing 80% of an object.
We isolate the attention matrix (forcing self-attention only) in the Context Encoder
to demonstrate that identity recovery collapses when routing is paralyzed.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import os
import sys

class AttentionRoutingAblator:
    def __init__(self, hooked_model):
        self.wm = hooked_model

    def evaluate_ablation(
        self,
        img_tensor: torch.Tensor,
        core_ids: List[int],
        bg_ids: List[int],
        context_ids: List[int],
        target_layers: List[int],
    ) -> dict:
        """Evaluate how much identity recovery degrades when Attention routing is paralyzed.
        
        Args:
            img_tensor: input image
            core_ids: target patches belonging to object core
            bg_ids: target patches belonging to background
            context_ids: visible patches (20%)
            target_layers: list of layers to ablate
        """
        self.wm.adapter.last_context_ids = context_ids
        self.wm.adapter.last_target_ids = core_ids + bg_ids
        
        # Baseline (No Ablation)
        with torch.no_grad():
            h, _ = self.wm.adapter.encode(img_tensor)
            clean_preds = self.wm.adapter.dynamics(h)
        
        # Get ground truth targets to measure true MSE
        gt_features = self.wm.adapter.target_encode(img_tensor)
        if gt_features.dim() == 2:
            gt_features = gt_features.unsqueeze(0)
            
        with torch.no_grad():
            clean_core_gt = gt_features[:, core_ids, :]
            clean_bg_gt = gt_features[:, bg_ids, :]
            
        clean_core_pred = clean_preds[:, :len(core_ids), :]
        clean_bg_pred = clean_preds[:, len(core_ids):, :]
        
        clean_core_mse = F.mse_loss(clean_core_pred, clean_core_gt).item()
        clean_bg_mse = F.mse_loss(clean_bg_pred, clean_bg_gt).item()

        hook_names = [f"context_encoder.blocks.{i}.attn.hook_pattern" for i in target_layers]
        
        def isolate_attention_hook(activation, hook):
            # activation shape: [B, num_heads, N, N]
            # Paralyze routing by forcing identity matrix (self-attention only)
            B, heads, N, _ = activation.shape
            eye = torch.eye(N, device=activation.device, dtype=activation.dtype)
            eye = eye.unsqueeze(0).unsqueeze(0).expand(B, heads, N, N)
            return eye
        
        for name in hook_names:
            self.wm.adapter.add_hook(name, isolate_attention_hook)
            
        try:
            with torch.no_grad():
                h, _ = self.wm.adapter.encode(img_tensor)
                ablated_preds = self.wm.adapter.dynamics(h)
        finally:
            for name in hook_names:
                self.wm.adapter.remove_hook(name)
            
        ablated_core_pred = ablated_preds[:, :len(core_ids), :]
        ablated_bg_pred = ablated_preds[:, len(core_ids):, :]
        
        ablated_core_mse = F.mse_loss(ablated_core_pred, clean_core_gt).item()
        ablated_bg_mse = F.mse_loss(ablated_bg_pred, clean_bg_gt).item()
        
        return {
            "clean_core_mse": clean_core_mse,
            "ablated_core_mse": ablated_core_mse,
            "core_degradation": ablated_core_mse - clean_core_mse,
            "clean_bg_mse": clean_bg_mse,
            "ablated_bg_mse": ablated_bg_mse,
            "bg_degradation": ablated_bg_mse - clean_bg_mse,
        }

def run_attention_evaluation(data_dir: str):
    from world_model_lens import HookedWorldModel
    from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
    from world_model_lens.core.config import WorldModelConfig
    from examples.ijepa.image_utils import preprocess_image
    from PIL import Image
    import glob
    
    print("Initializing Attention Routing Ablation Test on full dataset...")
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=12, n_heads=12)
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter, config)
    
    evaluator = AttentionRoutingAblator(wm)
    
    # Load dataset
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))
                
    if not image_paths:
        print(f"No images found in {data_dir}!")
        return
        
    print(f"Found {len(image_paths)} images for evaluation.")
    
    # Define Target vs Context splits
    grid_size = 14
    core_ids = []
    for r in range(4, 10):
        for c in range(4, 10):
            core_ids.append(r * grid_size + c)
            
    bg_ids = []
    for r in range(14):
        for c in range(14):
            if r < 2 or r >= 12 or c < 2 or c >= 12:
                bg_ids.append(r * grid_size + c)
                
    max_layer = len(wm.adapter.context_encoder.blocks) - 1
    stages = {
        "Early Stages (0-3)": list(range(0, min(4, max_layer + 1))),
        "Middle Stages (4-7)": list(range(4, min(8, max_layer + 1))),
        "Late Stages (8-11)": list(range(max(0, max_layer - 3), max_layer + 1)),
        "All Stages (0-11)": list(range(0, max_layer + 1)),
    }
    
    aggregated_results = {stage: {"core_deg": [], "bg_deg": []} for stage in stages.keys()}
    
    np.random.seed(42)
    
    for i, path in enumerate(image_paths):
        try:
            raw_img = Image.open(path).convert("RGB")
            img = preprocess_image(raw_img)
            
            # Sample 20% (approx 40 patches) purely from background
            context_ids = np.random.choice(bg_ids, size=40, replace=False).tolist()
            eval_bg_ids = [idx for idx in bg_ids if idx not in context_ids][:len(core_ids)]
            
            for stage_name, layers in stages.items():
                if not layers: continue
                res = evaluator.evaluate_ablation(img, core_ids, eval_bg_ids, context_ids, target_layers=layers)
                aggregated_results[stage_name]["core_deg"].append(res['core_degradation'])
                aggregated_results[stage_name]["bg_deg"].append(res['bg_degradation'])
                
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images...")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    print("\n=== FINAL AGGREGATED RESULTS (Attention Routing Ablation) ===")
    print(f"Evaluated {len(image_paths)} samples.")
    for stage_name in stages.keys():
        mean_core = np.mean(aggregated_results[stage_name]["core_deg"])
        mean_bg = np.mean(aggregated_results[stage_name]["bg_deg"])
        print(f"\n{stage_name}:")
        print(f"  Mean Core Object MSE Degradation: {'+' if mean_core > 0 else ''}{mean_core:.4f}")
        print(f"  Mean Background MSE Degradation:  {'+' if mean_bg > 0 else ''}{mean_bg:.4f}")
        
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image dataset directory")
    args = parser.parse_args()
    
    run_attention_evaluation(args.data_dir)
