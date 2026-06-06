"""Predictor Cross-Attention Ablation (RQ5 verification)

Investigates whether the Predictor's cross-attention mechanism is the primary engine
routing visible context patches to the masked target tokens to reconstruct missing identity.
We evaluate two ablations:
1. Identity: Each token in the Predictor only attends to itself.
2. Cross-Attention Blockade: Target tokens are prevented from attending to context tokens.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import os
import sys

class PredictorAttentionAblator:
    def __init__(self, hooked_model):
        self.wm = hooked_model

    def evaluate_ablation(
        self,
        img_tensor: torch.Tensor,
        core_ids: List[int],
        bg_ids: List[int],
        context_ids: List[int],
        target_layers: List[int],
        ablation_type: str,
    ) -> dict:
        """Evaluate how much identity recovery degrades under different Predictor attention ablations.
        
        Args:
            img_tensor: input image
            core_ids: target patches belonging to object core
            bg_ids: target patches belonging to background
            context_ids: visible patches (20%)
            target_layers: list of layers to ablate
            ablation_type: "identity" or "cross_block"
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

        hook_names = [f"predictor.blocks.{i}.attn.hook_pattern" for i in target_layers]
        N_context = len(context_ids)
        
        def predictor_attention_hook(activation, hook):
            # activation shape: [B, num_heads, N, N] where N = N_context + N_target
            B, heads, N, _ = activation.shape
            if ablation_type == "identity":
                eye = torch.eye(N, device=activation.device, dtype=activation.dtype)
                eye = eye.unsqueeze(0).unsqueeze(0).expand(B, heads, N, N)
                return eye
            elif ablation_type == "cross_block":
                act_clone = activation.clone()
                # Zero out target queries (rows) attending to context keys (columns)
                act_clone[:, :, N_context:, :N_context] = 0.0
                return act_clone
            return activation
        
        for name in hook_names:
            self.wm.adapter.add_hook(name, predictor_attention_hook)
            
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

def run_predictor_evaluation(data_dir: str):
    from world_model_lens import HookedWorldModel
    from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
    from world_model_lens.core.config import WorldModelConfig
    from examples.ijepa.image_utils import preprocess_image
    from PIL import Image
    
    print("Initializing Predictor Cross-Attention Ablation Test...")
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=12, n_heads=12)
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter, config)
    
    evaluator = PredictorAttentionAblator(wm)
    
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
                
    max_layer = len(wm.adapter.predictor.blocks) - 1
    stages = {
        "Early Stages (0-1)": list(range(0, min(2, max_layer + 1))),
        "Late Stages (2-3)": list(range(max(0, max_layer - 1), max_layer + 1)),
        "All Stages (0-3)": list(range(0, max_layer + 1)),
    }
    
    # Aggregated results structure
    results = {
        "identity": {stage: {"core_deg": [], "bg_deg": [], "clean_core": [], "ablated_core": []} for stage in stages.keys()},
        "cross_block": {stage: {"core_deg": [], "bg_deg": [], "clean_core": [], "ablated_core": []} for stage in stages.keys()}
    }
    
    np.random.seed(42)
    
    for i, path in enumerate(image_paths):
        try:
            raw_img = Image.open(path).convert("RGB")
            img = preprocess_image(raw_img)
            
            # Sample 20% (approx 40 patches) from background
            context_ids = np.random.choice(bg_ids, size=40, replace=False).tolist()
            eval_bg_ids = [idx for idx in bg_ids if idx not in context_ids][:len(core_ids)]
            
            for ab_type in ["identity", "cross_block"]:
                for stage_name, layers in stages.items():
                    if not layers: continue
                    res = evaluator.evaluate_ablation(
                        img, core_ids, eval_bg_ids, context_ids, target_layers=layers, ablation_type=ab_type
                    )
                    results[ab_type][stage_name]["core_deg"].append(res['core_degradation'])
                    results[ab_type][stage_name]["bg_deg"].append(res['bg_degradation'])
                    results[ab_type][stage_name]["clean_core"].append(res['clean_core_mse'])
                    results[ab_type][stage_name]["ablated_core"].append(res['ablated_core_mse'])
                    
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images...")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    for ab_type in ["identity", "cross_block"]:
        print(f"\n=== FINAL AGGREGATED RESULTS (Ablation Type: {ab_type.upper()}) ===")
        print(f"Evaluated {len(image_paths)} samples.")
        for stage_name in stages.keys():
            mean_core = np.mean(results[ab_type][stage_name]["core_deg"])
            mean_bg = np.mean(results[ab_type][stage_name]["bg_deg"])
            mean_clean = np.mean(results[ab_type][stage_name]["clean_core"])
            mean_ablated = np.mean(results[ab_type][stage_name]["ablated_core"])
            print(f"\n{stage_name}:")
            print(f"  Clean Core MSE:                    {mean_clean:.4f}")
            print(f"  Ablated Core MSE:                  {mean_ablated:.4f}")
            print(f"  Mean Core Object MSE Degradation:  {'+' if mean_core > 0 else ''}{mean_core:.4f}")
            print(f"  Mean Background MSE Degradation:   {'+' if mean_bg > 0 else ''}{mean_bg:.4f}")

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image dataset directory")
    args = parser.parse_args()
    
    run_predictor_evaluation(args.data_dir)
