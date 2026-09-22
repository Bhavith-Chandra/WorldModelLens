"""Attribution Patching for Scalable Transformer Interpretability (Suggestion 2)

Implements gradient-approximated activation patching (first-order Taylor expansion):
Delta_Loss \approx grad_h * (h_corrupted - h_clean)

Allows computing full causal patching sweeps across all layers/heads in 1 forward + 1 backward pass,
enabling scalable causal analysis on large models like ViT-H/14.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from world_model_lens import HookedWorldModel


class AttributionPatcher:
    """Scalable gradient-approximated activation patcher."""

    def __init__(self, hooked_model: HookedWorldModel):
        self.wm = hooked_model

    def compute_attribution_patching(
        self,
        clean_img: torch.Tensor,
        corrupted_img: torch.Tensor,
        context_ids: List[int],
        target_ids: List[int],
        layer_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Runs fast Attribution Patching across specified layer hooks.
        
        Args:
            clean_img: [1, 3, H, W] clean input image.
            corrupted_img: [1, 3, H, W] corrupted/masked/noise input image.
            context_ids: Context patch indices.
            target_ids: Target patch indices.
            layer_names: List of hook names to compute attribution patching for.
            
        Returns:
            Dict mapping hook_name -> estimated causal impact tensor.
        """
        device = next(self.wm.adapter.parameters()).device
        clean_img = clean_img.to(device)
        corrupted_img = corrupted_img.to(device)

        self.wm.adapter.last_context_ids = context_ids
        self.wm.adapter.last_target_ids = target_ids

        # 1. Corrupted Forward Pass to collect corrupted activations
        corrupted_activations = {}
        def make_corrupted_hook(name: str):
            def hook_fn(act, hook):
                corrupted_activations[name] = act.detach().clone()
                return act
            return hook_fn

        c_handles = []
        for name in layer_names:
            c_handle = self.wm.adapter.add_hook(name, make_corrupted_hook(name))
            c_handles.append((name, c_handle))

        try:
            with torch.no_grad():
                h_c, _ = self.wm.adapter.encode(corrupted_img)
                _ = self.wm.adapter.dynamics(h_c)
        finally:
            for name, _ in c_handles:
                self.wm.adapter.remove_hook(name)

        # 2. Clean Forward Pass + Backward Pass to collect gradients
        clean_activations = {}
        clean_gradients = {}

        def make_clean_hook(name: str):
            def fwd_fn(act, hook):
                clean_activations[name] = act
                # Register backward hook to catch gradients
                act.retain_grad()
                return act
            return fwd_fn

        c_handles = []
        for name in layer_names:
            c_handle = self.wm.adapter.add_hook(name, make_clean_hook(name))
            c_handles.append((name, c_handle))

        # Ground truth target
        gt_target = self.wm.adapter.target_encode(clean_img)[:, target_ids, :].detach()

        try:
            h_clean, _ = self.wm.adapter.encode(clean_img)
            clean_pred = self.wm.adapter.dynamics(h_clean)

            # Compute prediction loss
            loss = F.mse_loss(clean_pred, gt_target)

            # Zero grad and backward
            self.wm.adapter.zero_grad()
            loss.backward()

            # Collect gradients
            for name in layer_names:
                if name in clean_activations and clean_activations[name].grad is not None:
                    clean_gradients[name] = clean_activations[name].grad.detach().clone()
        finally:
            for name, _ in c_handles:
                self.wm.adapter.remove_hook(name)

        # 3. Compute Attribution Patching Taylor product: Grad * (h_corrupted - h_clean)
        patching_effects = {}
        for name in layer_names:
            if name in clean_activations and name in corrupted_activations and name in clean_gradients:
                h_clean_act = clean_activations[name].detach()
                h_corr_act = corrupted_activations[name]
                grad = clean_gradients[name]

                diff = h_corr_act - h_clean_act
                # First order Taylor approximation of loss delta
                approx_effect = (grad * diff).sum(dim=-1).squeeze(0).cpu().numpy()
                patching_effects[name] = approx_effect
            else:
                patching_effects[name] = np.zeros(len(context_ids))

        return patching_effects
