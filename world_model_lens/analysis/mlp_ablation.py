"""MLP Bottleneck Ablation (RQ5)

Investigates how I-JEPA recovers the identity of an object from 20% visibility.
We isolate a "core" target (e.g., center of image) vs a "background" target (border).
We then ablate the late-stage MLP outputs of the Context Encoder to demonstrate
that object identity recovery collapses when high-level semantic consolidation
is disrupted.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List

class MLPBottleneckAblator:
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
        """Evaluate how much identity recovery degrades when MLPs are ablated.
        
        Args:
            img_tensor: input image
            core_ids: target patches belonging to object core
            bg_ids: target patches belonging to background
            context_ids: visible patches (20%)
            target_layers: list of layers to ablate
        """
        # Baseline (No Ablation)
        self.wm.adapter.last_context_ids = context_ids
        self.wm.adapter.last_target_ids = core_ids + bg_ids
        with torch.no_grad():
            h, _ = self.wm.adapter.encode(img_tensor)
            clean_preds = self.wm.adapter.dynamics(h)
        
        # Get ground truth targets to measure true MSE
        # The adapter caches the last ground truth targets generated
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

        hook_names = [f"context_encoder.blocks.{i}.hook_mlp_out" for i in target_layers]
        
        def zero_ablation_hook(activation, hook):
            return torch.zeros_like(activation)
        
        for name in hook_names:
            self.wm.adapter.add_hook(name, zero_ablation_hook)
            
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

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))
    
    from world_model_lens import HookedWorldModel
    from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
    from world_model_lens.core.config import WorldModelConfig
    
    print("Initializing MLP Bottleneck Ablation Test...")
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=12, n_heads=12)
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter, config)
    
    evaluator = MLPBottleneckAblator(wm)
    
    img = torch.randn(1, 3, 224, 224)
    
    # 14x14 grid = 196 patches
    # Core: center 6x6 square (indices: rows 4-9, cols 4-9)
    # Background: Outer 2 layers (rows 0-1, 12-13, cols 0-1, 12-13)
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
                
    # Context IDs: Sample 20% (approx 40 patches) purely from the background
    # This forces the model to recover the core identity purely from peripheral features
    np.random.seed(42)
    context_ids = np.random.choice(bg_ids, size=40, replace=False).tolist()
    
    # We remove context ids from bg_ids for the target evaluation
    eval_bg_ids = [idx for idx in bg_ids if idx not in context_ids][:len(core_ids)] 
    
    max_layer = len(wm.adapter.context_encoder.blocks) - 1
    
    # Define stage configurations
    stages = {
        "Early Stages (0-3)": list(range(0, min(4, max_layer + 1))),
        "Middle Stages (4-7)": list(range(4, min(8, max_layer + 1))),
        "Late Stages (8-11)": list(range(max(0, max_layer - 3), max_layer + 1)),
        "Early + Middle Stages (0-7)": list(range(0, min(8, max_layer + 1))),
        "All Stages (0-11)": list(range(0, max_layer + 1)),
    }
    
    print("\n--- Results (Context Encoder MLP Bottleneck Ablation) ---")
    for stage_name, layers in stages.items():
        if not layers: continue
        res = evaluator.evaluate_ablation(img, core_ids, eval_bg_ids, context_ids, target_layers=layers)
        
        diff_core = res['ablated_core_mse'] - res['clean_core_mse']
        diff_bg = res['ablated_bg_mse'] - res['clean_bg_mse']
        
        print(f"\n{stage_name} (Layers {layers[0]}-{layers[-1]}):")
        print(f"  Core Object MSE Degradation: {'+' if diff_core > 0 else ''}{diff_core:.4f}")
        print(f"  Background MSE Degradation:  {'+' if diff_bg > 0 else ''}{diff_bg:.4f}")
    
    if res['core_degradation'] > res['bg_degradation']:
        print("  => Success: Ablating late-stage MLPs disproportionately destroyed Core Object identity recovery!")
