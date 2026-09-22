import torch
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from world_model_lens import HookedWorldModel
from world_model_lens.core.hooks import HookPoint, HookContext

def create_steering_hook(
    component_name: str,
    U: torch.Tensor,
    alpha: float,
    mode: str = "all"
) -> HookPoint:
    """Standard WML HookPoint. Kept for API compatibility, but we prefer run_steered_inference."""
    def hook_fn(tensor: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        return tensor + alpha * U.to(tensor.device)
    return HookPoint(name=component_name, fn=hook_fn, stage="post", timestep=0)

def run_steered_inference(
    wm: HookedWorldModel,
    img_tensor: torch.Tensor,
    component_name: str,
    U: torch.Tensor,
    alpha: float,
    mode: str = "all"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs a forward pass with native PyTorch forward hooks to guarantee propagation.
    
    This circumvents bugs in custom forward runners where hooked values are cached but
    not propagated to downstream blocks.
    
    Returns:
        target_preds: [B, N_target, d_embed] predicted latents
        target_gt: [B, N_target, d_embed] ground truth target latents
    """
    # Find the target module using regex mapping
    module = None
    if "predictor.layer_" in component_name:
        match = re.search(r'predictor\.layer_(\d+)', component_name)
        if match:
            idx = int(match.group(1))
            module = wm.adapter.predictor.blocks[idx]
    elif "encoder.blocks." in component_name:
        match = re.search(r'encoder\.blocks\.(\d+)', component_name)
        if match:
            idx = int(match.group(1))
            module = wm.adapter.context_encoder.blocks[idx]
            
    if module is None:
        # Fallback to standard unsteered pass if target module not found
        _, _, cache = wm.run_with_cache(img_tensor)
        return cache["predictor_out", 0], cache["target_encoder_out", 0]
        
    # Register the native PyTorch forward hook
    U_device = U.to(img_tensor.device)
    
    def hook_fn(mod, inp, out):
        is_batched = out.dim() == 3
        seq_len = out.shape[1] if is_batched else out.shape[0]
        dim = out.shape[2] if is_batched else out.shape[1]
        
        num_context = 0
        if hasattr(wm, "adapter") and wm.adapter is not None:
            if getattr(wm.adapter, "last_context_ids", None) is not None:
                num_context = len(wm.adapter.last_context_ids)
        if num_context == 0 or num_context >= seq_len:
            num_context = int(seq_len * 0.4)
            
        perturbation = alpha * U_device
        
        if mode == "all":
            return out + perturbation
        elif mode == "context":
            steered = out.clone()
            if is_batched:
                steered[:, :num_context, :] += perturbation
            else:
                steered[:num_context, :] += perturbation
            return steered
        elif mode == "target":
            steered = out.clone()
            if is_batched:
                steered[:, num_context:, :] += perturbation
            else:
                steered[num_context:, :] += perturbation
            return steered
        return out
        
    handle = module.register_forward_hook(hook_fn)
    try:
        # Run forward pass under steering
        _, _, cache = wm.run_with_cache(img_tensor)
        target_preds = cache["predictor_out", 0]
        target_gt = cache["target_encoder_out", 0]
    finally:
        # Clean up the hook to avoid leaking it into other passes
        handle.remove()
        
    return target_preds, target_gt
