"""MLP Bottleneck & Multi-Mode Ablation (RQ5)

Investigates how I-JEPA recovers the identity of an object from 20% visibility.
Supports Zero-Ablation, Mean-Ablation, and Resample-Ablation to test whether
negative delta MSE values are genuine representation signals or off-manifold artifacts.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional

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
        ablation_mode: str = "zero",  # "zero", "mean", or "resample"
        mean_activations: Optional[Dict[str, torch.Tensor]] = None,
        resampled_activations: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Evaluate identity recovery degradation under specified ablation mode.
        
        Args:
            img_tensor: input image tensor [1, 3, H, W]
            core_ids: target patches belonging to object core
            bg_ids: target patches belonging to background
            context_ids: visible patches (20%)
            target_layers: list of layers to ablate
            ablation_mode: "zero", "mean", or "resample"
            mean_activations: precomputed dataset mean activation per hook
            resampled_activations: activation from a different image
        """
        # Baseline (No Ablation)
        self.wm.adapter.last_context_ids = context_ids
        self.wm.adapter.last_target_ids = core_ids + bg_ids
        with torch.no_grad():
            h, _ = self.wm.adapter.encode(img_tensor)
            clean_preds = self.wm.adapter.dynamics(h)
        
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
        
        def make_ablation_hook(hook_name: str):
            def hook_fn(activation, hook):
                if ablation_mode == "zero":
                    return torch.zeros_like(activation)
                elif ablation_mode == "mean":
                    if mean_activations and hook_name in mean_activations:
                        return mean_activations[hook_name].to(activation.device)
                    else:
                        # Fallback to mean along token/batch dimensions
                        return activation.mean(dim=1, keepdim=True).expand_as(activation)
                elif ablation_mode == "resample":
                    if resampled_activations and hook_name in resampled_activations:
                        res_act = resampled_activations[hook_name].to(activation.device)
                        if res_act.shape == activation.shape:
                            return res_act
                        else:
                            return torch.roll(activation, shifts=1, dims=1)
                    else:
                        return torch.roll(activation, shifts=1, dims=1)
                else:
                    return torch.zeros_like(activation)
            return hook_fn

        for name in hook_names:
            self.wm.adapter.add_hook(name, make_ablation_hook(name))
            
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
            "ablation_mode": ablation_mode,
            "clean_core_mse": clean_core_mse,
            "ablated_core_mse": ablated_core_mse,
            "core_degradation": ablated_core_mse - clean_core_mse,
            "clean_bg_mse": clean_bg_mse,
            "ablated_bg_mse": ablated_bg_mse,
            "bg_degradation": ablated_bg_mse - clean_bg_mse,
        }
