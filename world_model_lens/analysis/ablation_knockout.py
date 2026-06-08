import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

from world_model_lens import HookedWorldModel
from world_model_lens.core.hooks import HookPoint, HookContext
from world_model_lens.analysis.attribution import (
    BaseAttribution, 
    extract_attention_weights
)

class PatchKnockoutEvaluator:
    """Evaluates causal impact of knocking out patches based on IG vs Attention.
    
    This converts causal sensitivity claims into true interventional causal claims
    by demonstrating that knocking out high-IG patches produces a larger MSE
    delta than knocking out high-attention patches.
    """
    
    def __init__(self, adapter: Any, k_values: List[int] = [1, 3, 5, 10]):
        """Initialize the evaluator.
        
        Args:
            adapter: Loaded World Model adapter (e.g. IJEPAAdapter).
            k_values: Number of top patches to knockout.
        """
        self.adapter = adapter
        self.adapter.eval()
        self.k_values = k_values
        
    @torch.no_grad()
    def _forward_score_with_knockout(
        self, 
        img_tensor: torch.Tensor, 
        context_ids: List[int], 
        target_id: int, 
        target_gt: torch.Tensor,
        patches_to_knockout: List[int]
    ) -> float:
        """Compute the prediction MSE score when specific patches are zeroed out.
        
        Args:
            img_tensor: [1, 3, H, W] input image.
            context_ids: Allowed context patch indices.
            target_id: Patch to predict.
            target_gt: [1, C] Ground truth embedding for the target.
            patches_to_knockout: List of context patch indices to zero out.
            
        Returns:
            MSE loss.
        """
        device = img_tensor.device
        
        # Get raw patch embeddings
        patch_emb = self.adapter.context_encoder.patch_embed(img_tensor)
        
        # Apply knockout
        for patch_idx in patches_to_knockout:
            if patch_idx in context_ids:
                patch_emb[:, patch_idx, :] = 0.0
                
        ctx_ids_t = torch.tensor(context_ids, device=device)
        
        # Add pos embed and process
        ctx_emb = patch_emb[:, ctx_ids_t, :]
        pos = self.adapter.context_encoder.pos_embed[:, ctx_ids_t, :]
        ctx_with_pos = ctx_emb + pos
        
        ctx_latents = self.adapter.context_encoder.forward_blocks(ctx_with_pos)
        pred = self.adapter.predictor(ctx_latents, context_ids, [target_id])
        
        mse = F.mse_loss(pred.squeeze(1), target_gt).item()
        return mse

    def evaluate_sample(
        self,
        wm: HookedWorldModel,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
        attr_scores: np.ndarray,
        layer_idx: int = -1
    ) -> Dict[str, Any]:
        """Evaluate patch knockout impact for a single sample.
        
        Args:
            wm: HookedWorldModel instance.
            img_tensor: Input image.
            context_ids: Context patch indices.
            target_id: Target patch index.
            attr_scores: Pre-computed attribution scores (e.g., from IG).
            layer_idx: Predictor layer to extract attention from.
            
        Returns:
            Dict containing MSE deltas for varying K.
        """
        device = next(self.adapter.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Get ground truth
        target_gt = self.adapter.target_encoder(img_tensor)[:, [target_id], :].squeeze(0).detach()
        
        # Baseline MSE (no knockout)
        baseline_mse = self._forward_score_with_knockout(
            img_tensor, context_ids, target_id, target_gt, patches_to_knockout=[]
        )
        
        # Get Attention weights (averaged across heads)
        attn_weights = extract_attention_weights(
            wm, img_tensor, context_ids, target_id, layer_idx=layer_idx, head_idx=None
        )
        
        # Map context array indices back to absolute patch indices
        # context_ids[i] corresponds to attn_weights[i] and attr_scores[i]
        
        # Get top-K indices in the context array for IG and Attention
        sorted_attr_idx = np.argsort(attr_scores)[::-1]
        sorted_attn_idx = np.argsort(attn_weights)[::-1]
        
        results = {
            "baseline_mse": baseline_mse,
            "k_values": [],
            "ig_mses": [],
            "attn_mses": [],
            "knockout_gap": [] # IG MSE - Attn MSE (positive means IG is better)
        }
        
        for k in self.k_values:
            if k > len(context_ids):
                continue
                
            top_k_attr_ctx_idx = sorted_attr_idx[:k]
            top_k_attn_ctx_idx = sorted_attn_idx[:k]
            
            top_k_attr_patches = [context_ids[i] for i in top_k_attr_ctx_idx]
            top_k_attn_patches = [context_ids[i] for i in top_k_attn_ctx_idx]
            
            ig_mse = self._forward_score_with_knockout(
                img_tensor, context_ids, target_id, target_gt, top_k_attr_patches
            )
            
            attn_mse = self._forward_score_with_knockout(
                img_tensor, context_ids, target_id, target_gt, top_k_attn_patches
            )
            
            results["k_values"].append(k)
            results["ig_mses"].append(ig_mse)
            results["attn_mses"].append(attn_mse)
            results["knockout_gap"].append(ig_mse - attn_mse)
            
        return results
