import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Callable
import random

from world_model_lens import HookedWorldModel
from world_model_lens.core.hooks import HookPoint, HookContext

class PathPatchingEvaluator:
    """Evaluates causal paths using activation swapping.
    
    Path Patching swaps activations between a clean image and a paired/corrupted
    image to localize exactly where the causal signal decouples from attention.
    Instead of zero-ablating, we substitute activations from a random other image.
    """
    
    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.adapter.eval()

    @torch.no_grad()
    def evaluate_inter_batch_swaps(
        self,
        wm: HookedWorldModel,
        img_tensors: torch.Tensor,
        context_ids: List[int],
        target_id: int,
        layer_idx: int,
        sublayer: str = "attn" # "attn" or "mlp"
    ) -> Dict[str, Any]:
        """Perform path patching by swapping activations across a batch.
        
        Args:
            wm: HookedWorldModel.
            img_tensors: [B, 3, H, W] batch of images.
            context_ids: Context patch indices.
            target_id: Target patch index.
            layer_idx: Predictor layer to swap at.
            sublayer: "attn" or "mlp" to swap the specific sublayer output.
            
        Returns:
            Dictionary with original MSEs, patched MSEs, and delta.
        """
        B = img_tensors.size(0)
        if B < 2:
            raise ValueError("Batch size must be >= 2 for inter-batch swapping.")
            
        device = img_tensors.device
        
        # 1. Baseline Run (Clean)
        target_reps = self.adapter.target_encoder(img_tensors)
        target_gt = target_reps[:, [target_id], :] # [B, 1, C]
        
        # We need to set the context ids properly in adapter for the forward run
        wm.adapter.last_context_ids = context_ids
        wm.adapter.last_target_ids = [target_id]
        
        # Get baseline prediction and cache the clean activations
        hook_name = f"predictor.blocks.{layer_idx}.hook_{sublayer}_out"
        clean_out, clean_cache = wm.run_with_cache(
            img_tensors,
            names_filter=[hook_name]
        )
        
        if isinstance(clean_out, tuple):
            clean_pred = clean_out[0]
        else:
            clean_pred = clean_out
            
        clean_mse = F.mse_loss(clean_pred.squeeze(1), target_gt.squeeze(1), reduction='none').mean(dim=-1).cpu().numpy()
        
        clean_activations = clean_cache[hook_name, 0] # [B, seq_len, C]
        
        # 2. Setup permutations for inter-batch swaps
        # Each item i gets activation from item p[i]
        p = list(range(B))
        random.shuffle(p)
        # Ensure no self-swaps if possible
        for i in range(B):
            if p[i] == i:
                swap_idx = (i + 1) % B
                p[i], p[swap_idx] = p[swap_idx], p[i]
                
        corrupted_activations = clean_activations[p].detach()
        
        # 3. Patched Run
        def patching_hook(tensor: torch.Tensor, ctx: HookContext) -> torch.Tensor:
            # Substitute the activations with the corrupted ones
            return corrupted_activations
            
        patched_out = wm.run_with_hooks(
            img_tensors,
            fwd_hooks=[(hook_name, patching_hook)]
        )
        
        if isinstance(patched_out, tuple):
            patched_pred = patched_out[0]
        else:
            patched_pred = patched_out
            
        patched_mse = F.mse_loss(patched_pred.squeeze(1), target_gt.squeeze(1), reduction='none').mean(dim=-1).cpu().numpy()
        
        return {
            "permutation": p,
            "clean_mse": clean_mse,
            "patched_mse": patched_mse,
            "mse_delta": patched_mse - clean_mse # Positive means swapping caused error (expected)
        }
