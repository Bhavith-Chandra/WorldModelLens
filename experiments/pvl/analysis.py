import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
from world_model_lens import HookedWorldModel
from experiments.pvl.intervention import run_steered_inference
from experiments.pvl.observables import LinearProbe

def analyze_steering_causality(
    wm: HookedWorldModel,
    img_tensor: torch.Tensor,
    component_name: str,
    U: torch.Tensor,
    probes: Dict[str, LinearProbe],
    alphas: List[float] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
) -> Tuple[Dict[str, List[float]], List[float]]:
    """Sweeps alpha and measures changes in linear probe predictions and identity similarity.
    
    Returns:
        probe_responses: Dict mapping variable_name -> list of float predictions
        identity_similarities: List of float cosine similarities to original predictions
    """
    probe_responses = {k: [] for k in probes.keys()}
    identity_similarities = []
    
    # Run baseline unsteered inference to get reference latents
    baseline_pred, target_gt = run_steered_inference(wm, img_tensor, component_name, U, alpha=0.0)
    
    # Flatten targets and predictions to compute cosine similarity
    # Shapes: [B, N_target, dim] -> [B * N_target, dim]
    baseline_flat = baseline_pred.reshape(-1, baseline_pred.shape[-1])
    
    for alpha in alphas:
        pred_steered, _ = run_steered_inference(wm, img_tensor, component_name, U, alpha, mode="all")
        pred_flat = pred_steered.reshape(-1, pred_steered.shape[-1])
        
        # 1. Probe predictions
        for k, probe in probes.items():
            pred_val = float(probe.predict(pred_flat).mean().item())
            probe_responses[k].append(pred_val)
            
        # 2. Identity preservation similarity
        cos_sim = F.cosine_similarity(pred_flat, baseline_flat, dim=-1).mean().item()
        identity_similarities.append(float(cos_sim))
        
    return probe_responses, identity_similarities

def label_subspace(
    probe_responses: Dict[str, List[float]],
    identity_similarities: List[float],
    alphas: List[float]
) -> Tuple[str, float, float]:
    """Assigns a semantic label to a direction U based on its causal effect signatures.
    
    Returns:
        label: name of dominant variable
        effect_size: linear slope of response vs alpha
        identity_preservation: mean identity similarity across sweep
    """
    slopes = {}
    
    # Fit linear slope for each physical variable
    for k, vals in probe_responses.items():
        # Fit y = m * x + c using numpy
        m, _ = np.polyfit(alphas, vals, 1)
        # We normalize the slope relative to the std dev of the variable across sweep (effect scale)
        slopes[k] = float(m)
        
    # Find the dominant variable (highest absolute normalized slope)
    # We exclude position variables for styling directions unless they dominate
    dominant_var = max(slopes.keys(), key=lambda k: abs(slopes[k]))
    effect_size = slopes[dominant_var]
    mean_id_preservation = float(np.mean(identity_similarities))
    
    return dominant_var, effect_size, mean_id_preservation

def localize_layers(
    wm: HookedWorldModel,
    images: List[Tuple[str, str]],
    metadata: List[Dict[str, Any]],
    image_tensors: List[torch.Tensor],
    layers_to_check: List[str],
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """Tracks physical variables through depth by fitting probes and measuring R^2 scores.
    
    Returns:
        layer_scores: Dict mapping layer_name -> Dict of {variable: R2_score}
    """
    from experiments.pvl.latent_collection import collect_dataset_latents
    from experiments.pvl.observables import train_probes_for_dataset, compute_patch_properties
    
    layer_scores = {}
    
    print("\n--- Starting depth-wise layer localization sweep ---")
    for layer in layers_to_check:
        print(f"Collecting & probing layer: {layer}")
        # Collect activations for this layer specifically
        acts_dict, layer_meta = collect_dataset_latents(wm, images, [layer], device)
        if layer not in acts_dict:
            continue
            
        acts = acts_dict[layer]
        # Train probes on this layer
        probes = train_probes_for_dataset(acts, layer_meta, image_tensors, device)
        
        # Calculate local ground truths for this layer's metadata
        layer_targets = {
            "brightness": [],
            "contrast": [],
            "complexity": [],
            "grid_y": [],
            "grid_x": [],
            "radial_distance": [],
            "aspect_ratio_proxy": [],
            "local_entropy": [],
            "color_saliency": [],
            "edge_direction": []
        }
        for m in layer_meta:
            img_idx = m["image_idx"]
            patch_idx = m["patch_idx"]
            img_t = image_tensors[img_idx]
            patch_size = m.get("patch_size", 16)
            props = compute_patch_properties(img_t, patch_idx, patch_size=patch_size)
            for k in layer_targets.keys():
                layer_targets[k].append(props[k])
                
        layer_vars = {k: np.var(v) for k, v in layer_targets.items()}
        
        # Evaluate validation R^2 on dataset
        scores = {}
        for k, probe in probes.items():
            y_pred = probe.predict(acts.to(device)).cpu()
            y_true = torch.tensor(layer_targets[k], dtype=torch.float32)
            mse = F.mse_loss(y_pred, y_true).item()
            
            # R^2 = 1 - MSE / Var
            r2 = 1.0 - (mse / (layer_vars[k] + 1e-9))
            scores[k] = float(max(-1.0, min(1.0, r2))) # clamp
            
        layer_scores[layer] = scores
        print(f"Layer {layer} R^2: { {k: f'{v:.3f}' for k, v in scores.items()} }")
        
    return layer_scores

def test_cross_attention_consumption(
    wm: HookedWorldModel,
    img_tensor: torch.Tensor,
    U: torch.Tensor,
    predictor_layer: str,
    device: str = "cpu"
) -> Dict[str, float]:
    """Test if a discovered direction U is consumed by predictor cross-attention.
    
    We project out direction U from the predictor activations before the cross-attention block,
    and measure the changes.
    """
    U_unit = U / (torch.norm(U) + 1e-9)
    U_unit = U_unit.to(device)
    
    import re
    match = re.search(r'predictor\.layer_(\d+)', predictor_layer)
    layer_idx = int(match.group(1)) if match else 0
    if hasattr(wm.adapter, 'predictor') and hasattr(wm.adapter.predictor, 'blocks'):
        layer_idx = min(layer_idx, len(wm.adapter.predictor.blocks) - 1)
    
    target_block = wm.adapter.predictor.blocks[layer_idx]
    
    # 1. Run baseline pass to collect original attention pattern and predictions
    _, _, cache_orig = wm.run_with_cache(img_tensor)
    original_attn_pattern = target_block.attn.last_attn_weights.detach().clone() if target_block.attn.last_attn_weights is not None else None
    clean_pred = cache_orig["predictor_out", 0]
    
    # 2. Run patched pass where we project out U before the block using forward pre-hook
    def project_out_pre_hook(module, inputs):
        # inputs is a tuple, first element is the input tensor [B, seq_len, dim]
        x = inputs[0]
        proj = torch.matmul(x, U_unit) # [B, seq_len]
        projection_vector = proj.unsqueeze(-1) * U_unit.reshape(1, 1, -1) # [B, seq_len, dim]
        x_patched = x - projection_vector
        return (x_patched,) + inputs[1:]
        
    handle = target_block.register_forward_pre_hook(project_out_pre_hook)
    try:
        _, _, cache_patched = wm.run_with_cache(img_tensor)
        patched_attn_pattern = target_block.attn.last_attn_weights.detach().clone() if target_block.attn.last_attn_weights is not None else None
        patched_pred = cache_patched["predictor_out", 0]
    finally:
        handle.remove()
        
    # Measure impact
    # A. Prediction degradation (MSE increase)
    degradation = F.mse_loss(patched_pred, clean_pred).item()
    
    # B. Attention shift (KL divergence between original and patched patterns)
    attn_shift = 0.0
    if original_attn_pattern is not None and patched_attn_pattern is not None:
        p = original_attn_pattern.clamp(min=1e-8)
        q = patched_attn_pattern.clamp(min=1e-8)
        kl = (p * (p.log() - q.log())).sum(dim=-1).mean().item()
        attn_shift = kl
        
    return {
        "prediction_degradation_mse": degradation,
        "attention_pattern_kl_shift": attn_shift
    }
