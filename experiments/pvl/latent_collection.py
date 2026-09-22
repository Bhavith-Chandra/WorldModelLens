import os
import torch
from typing import List, Dict, Tuple, Any
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.hub.model_hub import ModelHub
from examples.ijepa.image_utils import get_sample_image, preprocess_image

def get_model_and_config(weights_path: str = "vith14_in1k_ep300.pth.tar", device: str = "cpu") -> Tuple[HookedWorldModel, WorldModelConfig]:
    """Loads and wraps an I-JEPA model (mini or ViT-H/14) in a HookedWorldModel wrapper."""
    if os.path.exists(weights_path):
        try:
            adapter = ModelHub._load_ijepa(weights_path, device=device)
            config = adapter.config
            print(f"Successfully loaded I-JEPA checkpoint from {weights_path} (d_embed={config.d_embed}, n_layers={config.n_layers})")
        except Exception as e:
            print(f"ModelHub loader encountered error loading {weights_path}: {e}. Falling back to default adapter loader.")
            config = WorldModelConfig(
                backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=192,
                world_model_family=WorldModelFamily.JEPA
            )
            adapter = IJEPAAdapter(config)
            adapter.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=False)
            adapter.to(device=device)
    else:
        print(f"Weights file not found at {weights_path}. Initializing default random weights.")
        config = WorldModelConfig(
            backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=192,
            world_model_family=WorldModelFamily.JEPA
        )
        adapter = IJEPAAdapter(config)
        adapter.to(device=device)

    adapter.eval()
    wm = HookedWorldModel(adapter, config)
    return wm, config

def discover_images(data_dir: str = "data/eval_dataset") -> List[Tuple[str, str]]:
    """Scan eval_dataset directory and return a list of (image_path, category) tuples."""
    image_list = []
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return []
        
    for root, dirs, files in os.walk(data_dir):
        category = os.path.basename(root)
        if category == os.path.basename(data_dir):
            continue
            
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                full_path = os.path.join(root, f)
                image_list.append((full_path, category))
                
    print(f"Discovered {len(image_list)} images across categories.")
    return image_list

def collect_dataset_latents(
    wm: HookedWorldModel,
    images: List[Tuple[str, str]],
    layers: List[str],
    device: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """Runs I-JEPA forward on all images and collects latents for selected layers.
    
    Returns:
        activations: Dict mapping layer_name -> Tensor [Total_patches, d_embed/predictor_embed_dim]
        metadata: List of dicts, one per patch/token, containing source info
    """
    activations_accum: Dict[str, List[torch.Tensor]] = {layer: [] for layer in layers}
    metadata_list = []
    
    # We enforce evaluation mode and no-grad for this collection pass
    wm.adapter.eval()
    
    first_layer = layers[0] if layers else ""
    grid_size = getattr(wm.config, 'img_size', 224) // getattr(wm.config, 'patch_size', 16)
    
    for idx, (path, category) in enumerate(images):
        if (idx + 1) % 5 == 0 or idx == 0 or idx == len(images) - 1:
            print(f"  [Phase 1] Processing image {idx + 1}/{len(images)}: {category} ({os.path.basename(path)})...")
        try:
            img = get_sample_image(path)
            img_tensor = preprocess_image(img).to(device)
            
            # Running with cache generates context_ids and target_ids automatically
            torch.manual_seed(idx)
            import numpy as np
            np.random.seed(idx)
            
            world_traj, latent_traj, cache = wm.run_with_cache(img_tensor)
            
            context_ids = list(wm.adapter.last_context_ids)
            target_ids = list(wm.adapter.last_target_ids)
            
            for layer in layers:
                if (layer, 0) not in cache:
                    continue
                val = cache[layer, 0].cpu()
                if val.dim() == 3:
                    val = val.squeeze(0) # [seq_len, dim]
                activations_accum[layer].append(val)
                
            # Determine active token sequence for metadata matching the first layer
            num_patches = grid_size * grid_size
            if "target_encoder" in first_layer:
                active_tokens = list(range(num_patches))
            elif "encoder" in first_layer and "predictor" not in first_layer:
                active_tokens = context_ids
            elif "predictor_out" in first_layer:
                active_tokens = target_ids
            else:
                active_tokens = context_ids + target_ids
                
            for pos_in_seq, patch_idx in enumerate(active_tokens):
                if "predictor_out" in first_layer:
                    is_target = True
                else:
                    is_target = pos_in_seq >= len(context_ids) if "predictor" in first_layer else False
                metadata_list.append({
                    "image_idx": idx,
                    "image_path": path,
                    "category": category,
                    "is_target": is_target,
                    "patch_idx": patch_idx,
                    "seq_pos": pos_in_seq,
                    "grid_y": patch_idx // grid_size,
                    "grid_x": patch_idx % grid_size
                })
                
        except Exception as e:
            print(f"Error collecting latents for image {path}: {e}")
            
    stacked_activations = {}
    for layer, tensors in activations_accum.items():
        if tensors:
            stacked_activations[layer] = torch.cat(tensors, dim=0)
            print(f"Collected {layer} activations shape: {stacked_activations[layer].shape}")
            
    return stacked_activations, metadata_list
