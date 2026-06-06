import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from world_model_lens.core.hooks import HookPoint, HookContext
from world_model_lens import HookedWorldModel
from world_model_lens.analysis.attribution import BaseAttribution

class SublayerAttribution:
    """Isolates causal attribution to Attention vs MLP sublayers.
    
    This helps answer whether failures in causal sensitivity are due to 
    attention routing mechanisms or the MLP mapping mechanisms within a layer.
    """
    
    def __init__(self, adapter: Any, n_steps: int = 20):
        self.adapter = adapter
        self.adapter.eval()
        self.n_steps = n_steps

    @torch.no_grad()
    def _get_target_gt(self, img_tensor: torch.Tensor, target_id: int) -> torch.Tensor:
        target_reps = self.adapter.target_encoder(img_tensor)
        return target_reps[:, [target_id], :]
        
    def compute_sublayer_ig(
        self, 
        wm: HookedWorldModel, 
        img_tensor: torch.Tensor, 
        context_ids: List[int], 
        target_id: int,
        layer_idx: int
    ) -> Dict[str, np.ndarray]:
        """Compute Integrated Gradients separately for Attention out and MLP out of a layer.
        
        Args:
            wm: HookedWorldModel instance.
            img_tensor: [1, 3, H, W]
            context_ids: Allowed context patch indices.
            target_id: Patch to predict.
            layer_idx: Predictor layer to analyze.
            
        Returns:
            Dictionary with 'attn' and 'mlp' attribution arrays.
        """
        device = img_tensor.device
        
        # 1. First run to get baseline (mean) activations for attn and mlp
        with torch.no_grad():
            _, cache = wm.run_with_cache(
                img_tensor, 
                names_filter=[
                    f"predictor.blocks.{layer_idx}.hook_attn_out",
                    f"predictor.blocks.{layer_idx}.hook_mlp_out"
                ]
            )
            
            actual_attn_out = cache[f"predictor.blocks.{layer_idx}.hook_attn_out", 0]
            actual_mlp_out = cache[f"predictor.blocks.{layer_idx}.hook_mlp_out", 0]
            
            baseline_attn = actual_attn_out.mean(dim=1, keepdim=True).expand_as(actual_attn_out)
            baseline_mlp = actual_mlp_out.mean(dim=1, keepdim=True).expand_as(actual_mlp_out)
            
        target_gt = self._get_target_gt(img_tensor, target_id)
        target_gt_flat = target_gt.squeeze(0)
        
        # 2. Setup interpolation
        alphas = torch.linspace(0, 1, self.n_steps, device=device)
        delta_attn = (actual_attn_out - baseline_attn).detach()
        delta_mlp = (actual_mlp_out - baseline_mlp).detach()
        
        accum_grad_attn = torch.zeros_like(actual_attn_out)
        accum_grad_mlp = torch.zeros_like(actual_mlp_out)
        
        for alpha in alphas:
            # We want to patch the attention and mlp outputs during the forward pass
            interp_attn = (baseline_attn + alpha * delta_attn).detach().requires_grad_(True)
            interp_mlp = (baseline_mlp + alpha * delta_mlp).detach().requires_grad_(True)
            
            def patch_attn(tensor: torch.Tensor, ctx: HookContext) -> torch.Tensor:
                return interp_attn
                
            def patch_mlp(tensor: torch.Tensor, ctx: HookContext) -> torch.Tensor:
                return interp_mlp
                
            # Need to set up the input state manually since we are using wm.run_with_hooks
            wm.adapter.last_context_ids = context_ids
            wm.adapter.last_target_ids = [target_id]
            
            # Forward pass with hooks
            out = wm.run_with_hooks(
                img_tensor,
                fwd_hooks=[
                    (f"predictor.blocks.{layer_idx}.hook_attn_out", patch_attn),
                    (f"predictor.blocks.{layer_idx}.hook_mlp_out", patch_mlp)
                ]
            )
            
            # If the model wrapper returns a tuple, get the prediction tensor
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
                
            # Score
            score = -torch.nn.functional.mse_loss(pred.squeeze(1), target_gt_flat)
            score.backward()
            
            if interp_attn.grad is not None:
                accum_grad_attn += interp_attn.grad.detach()
            if interp_mlp.grad is not None:
                accum_grad_mlp += interp_mlp.grad.detach()
                
        ig_attn = delta_attn * (accum_grad_attn / self.n_steps)
        ig_mlp = delta_mlp * (accum_grad_mlp / self.n_steps)
        
        # Only take the context patch elements
        # Shape: [1, seq_len, dim] -> [seq_len] -> [len(context_ids)]
        ig_attn_per_patch = ig_attn.abs().sum(dim=-1).squeeze(0)[:len(context_ids)].cpu().numpy()
        ig_mlp_per_patch = ig_mlp.abs().sum(dim=-1).squeeze(0)[:len(context_ids)].cpu().numpy()
        
        return {
            "attn": ig_attn_per_patch,
            "mlp": ig_mlp_per_patch
        }
