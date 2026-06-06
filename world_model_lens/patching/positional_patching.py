"""Positional Counterfactual Patching (RQ1)

This module investigates how I-JEPA routes context to target patches.
By intercepting the predictor's residual stream right before cross-attention,
we swap the positional embeddings of two target tokens to see if the predicted
visual features (pixels) swap accordingly.
"""

import torch
import torch.nn.functional as F

class PositionalPatchingEvaluator:
    def __init__(self, hooked_model):
        self.wm = hooked_model

    def evaluate_swap(self, img_tensor, context_ids, tid1, tid2):
        """
        Evaluate the counterfactual of swapping the positional embeddings of two target patches.
        
        Args:
            img_tensor: [1, 3, H, W]
            context_ids: List of context patch indices
            tid1: First target patch index
            tid2: Second target patch index
            
        Returns:
            dict containing MSE comparisons.
        """
        # Baseline (No Ablation)
        self.wm.adapter.last_context_ids = context_ids
        self.wm.adapter.last_target_ids = [tid1, tid2]
        with torch.no_grad():
            h, _ = self.wm.adapter.encode(img_tensor)
            clean_preds = self.wm.adapter.dynamics(h)
        
        # In the HookedWorldModel, the adapter handles the list of target ids.
        clean_pred_tid1 = clean_preds[:, 0, :]
        clean_pred_tid2 = clean_preds[:, 1, :]
        
        # Define the swapping hook for the predictor's entry residual stream
        def swap_target_positions_hook(activation, hook):
            # Activation shape: [B, seq_len, embed_dim]
            # In I-JEPA, the sequence is [context_tokens... , target_tokens...]
            # Since we passed exactly 2 target tokens, they are the last 2 elements.
            seq_len = activation.shape[1]
            n_context = len(context_ids)
            
            # Sanity check
            if seq_len != n_context + 2:
                raise ValueError(f"Expected seq_len {n_context + 2}, got {seq_len}")
                
            # Swap the last two tokens (which correspond to tid1 and tid2)
            patched_activation = activation.clone()
            patched_activation[:, n_context, :] = activation[:, n_context + 1, :]
            patched_activation[:, n_context + 1, :] = activation[:, n_context, :]
            
            return patched_activation

        # Run patched forward pass
        # The HookPoint name is `predictor.hook_resid_pre`
        from world_model_lens.core.hooks import HookPoint
        
        hook_name = "predictor.hook_resid_pre"
        self.wm.adapter.add_hook(hook_name, swap_target_positions_hook)
        try:
            with torch.no_grad():
                h, _ = self.wm.adapter.encode(img_tensor)
                patched_preds = self.wm.adapter.dynamics(h)
        finally:
            self.wm.adapter.remove_hook(hook_name)
        
        patched_pred_pos1 = patched_preds[:, 0, :] # This is physically position 1, but should contain visual info of tid2
        patched_pred_pos2 = patched_preds[:, 1, :] # This is physically position 2, but should contain visual info of tid1

        # Calculate MSEs
        # 1. Did position 1 generate tid2's pixels?
        cross_mse_1 = F.mse_loss(patched_pred_pos1, clean_pred_tid2).item()
        # 2. Did position 2 generate tid1's pixels?
        cross_mse_2 = F.mse_loss(patched_pred_pos2, clean_pred_tid1).item()
        
        # Control: Compare to their original intended pixels (should be high error)
        orig_mse_1 = F.mse_loss(patched_pred_pos1, clean_pred_tid1).item()
        orig_mse_2 = F.mse_loss(patched_pred_pos2, clean_pred_tid2).item()
        
        return {
            "cross_mse": (cross_mse_1 + cross_mse_2) / 2,
            "orig_mse": (orig_mse_1 + orig_mse_2) / 2,
            "success": cross_mse_1 < orig_mse_1 and cross_mse_2 < orig_mse_2
        }

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath("."))
    
    from world_model_lens import HookedWorldModel
    from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
    from world_model_lens.core.config import WorldModelConfig
    from examples.ijepa.image_utils import get_ijepa_masks
    
    print("Initializing Positional Patching Test...")
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=12, n_heads=12)
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter, config)
    
    evaluator = PositionalPatchingEvaluator(wm)
    
    img = torch.randn(1, 3, 224, 224)
    c_ids, t_ids = get_ijepa_masks(num_context=40, num_target=2)
    
    if len(t_ids) >= 2:
        res = evaluator.evaluate_swap(img, c_ids, t_ids[0], t_ids[1])
        print("Results:")
        print(f"  MSE when compared to SWAPPED identity (should be low):  {res['cross_mse']:.4f}")
        print(f"  MSE when compared to ORIGINAL identity (should be high): {res['orig_mse']:.4f}")
        print(f"  Routing Swap Successful: {res['success']}")
