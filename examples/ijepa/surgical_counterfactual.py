import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import logging

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookPoint
from image_utils import preprocess_image, get_sample_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SurgicalCounterfactual")

class SurgicalCounterfactual:
    def __init__(self, cat_path, dog_path, target_patch=118):
        self.target_patch = target_patch
        self.config = WorldModelConfig(backend="ijepa")
        # Ensure correct dimensions
        self.config.n_layers = 6
        self.config.n_heads = 3
        self.config.embed_dim = 192
        self.config.predictor_embed_dim = 192
        
        self.adapter = IJEPAAdapter(self.config)
        checkpoint_path = "ijepa_mini.pth"
        if os.path.exists(checkpoint_path):
            self.adapter.from_checkpoint(checkpoint_path)
            
        self.wm = HookedWorldModel(self.adapter, self.config)
        self.wm.adapter.eval()
        
        # Load images
        self.img_cat = preprocess_image(Image.open(cat_path).convert("RGB"))
        self.img_dog = preprocess_image(Image.open(dog_path).convert("RGB"))
        
        # Consistent mask (top half)
        self.context_ids = list(range(196 // 2))
        self.adapter.last_context_ids = self.context_ids
        self.adapter.last_target_ids = [self.target_patch]
        
        self.head_dim = self.config.embed_dim // self.config.n_heads

    def discover_circuit_heads(self):
        """Finds top-3 and bottom-3 heads based on importance delta."""
        logger.info("Discovering causal circuit heads...")
        
        # 1. Baseline
        self.wm.clear_hooks()
        _, cache_ref = self.wm.run_with_cache(self.img_dog)
        baseline_pred = cache_ref.get("z_prior", 0)
        
        # 2. Optimized Sweep (Last 8 layers for CPU efficiency)
        importance = []
        start_layer = max(0, self.config.n_layers - 8)
        logger.info(f"Sweeping layers {start_layer} to {self.config.n_layers-1}...")
        for l in range(start_layer, self.config.n_layers):
            for h in range(self.config.n_heads):
                def ablation_hook(x, ctx, head_idx=h):
                    x_new = x.clone()
                    x_new[:, :, head_idx*self.head_dim : (head_idx+1)*self.head_dim] = 0
                    return x_new
                
                self.wm.clear_hooks()
                self.wm.add_hook(HookPoint(name=f"context_encoder.blocks.{l}.attn.hook_z", fn=ablation_hook))
                _, cache_abl = self.wm.run_with_cache(self.img_dog)
                mse = F.mse_loss(cache_abl.get("z_prior", 0), baseline_pred).item()
                importance.append(((l, h), mse))
                # Add progress heartbeat
                if (h + 1) % 4 == 0 or h == self.config.n_heads - 1:
                    logger.info(f"  > Layer {l}, Head {h} complete ({h+1}/{self.config.n_heads})")
        
        # Sort by importance (MSE shift)
        importance.sort(key=lambda x: x[1], reverse=True)
        top_3 = [x[0] for x in importance[:3]]
        bottom_3 = [x[0] for x in importance[-3:]]
        
        logger.info(f"Top-3 Causal Heads: {top_3}")
        logger.info(f"Bottom-3 (Control) Heads: {bottom_3}")
        return top_3, bottom_3

    def capture_cat_values(self, target_heads):
        """Captures value vectors for specific heads from the cat image."""
        captured_values = {}
        
        def capture_hook(v, ctx):
            # v is [B, heads, N, head_dim]
            captured_values[ctx.component] = v.detach().clone()
            return v
            
        self.wm.clear_hooks()
        for layer_idx, _ in target_heads:
            hook_name = f"context_encoder.blocks.{layer_idx}.attn.hook_value"
            if hook_name not in [h.name for h in self.wm._hooks._hooks.get(hook_name, [])]:
                self.wm.add_hook(HookPoint(name=hook_name, fn=capture_hook))
            
        self.wm.run_with_cache(self.img_cat)
        return captured_values

    def run_surgical_intervention(self, target_heads, cat_values):
        """Intervenes on specific heads by injecting cat value vectors into dog pass."""
        
        def surgical_swap_hook(v, ctx):
            v_new = v.clone()
            # Find which heads in this layer are targeted
            layer_idx = int(ctx.component.split('.')[2])
            for l, h in target_heads:
                if l == layer_idx:
                    # Replace specific head h with cat's vector
                    v_new[:, h, :, :] = cat_values[ctx.component][:, h, :, :]
            return v_new
            
        self.wm.clear_hooks()
        # Add hooks to all unique layers involved
        unique_layers = set([l for l, h in target_heads])
        for l in unique_layers:
            self.wm.add_hook(HookPoint(name=f"context_encoder.blocks.{l}.attn.hook_value", fn=surgical_swap_hook))
            
        _, cache_intervened = self.wm.run_with_cache(self.img_dog)
        return cache_intervened.get("z_prior", 0)

    def run_experiment(self):
        top_3, bottom_3 = self.discover_circuit_heads()
        
        # Baselines
        self.wm.clear_hooks()
        _, cache_dog = self.wm.run_with_cache(self.img_dog)
        dog_base = cache_dog.get("z_prior", 0)
        
        self.wm.clear_hooks()
        _, cache_cat = self.wm.run_with_cache(self.img_cat)
        cat_base = cache_cat.get("z_prior", 0)
        
        # Capture Cat Values for all 6 heads
        all_targets = list(set(top_3 + bottom_3))
        cat_values = self.capture_cat_values(all_targets)
        
        # 1. Causal Intervention
        logger.info("Executing Surgical Swap on TOP-3 Heads...")
        pred_causal = self.run_surgical_intervention(top_3, cat_values)
        
        # 2. Control Intervention
        logger.info("Executing Surgical Swap on BOTTOM-3 Heads...")
        pred_control = self.run_surgical_intervention(bottom_3, cat_values)
        
        # Calculate Distances
        def get_dist(p1, p2):
            return torch.norm(p1 - p2).item()
            
        results = {
            "causal": {
                "to_dog": get_dist(pred_causal, dog_base),
                "to_cat": get_dist(pred_causal, cat_base)
            },
            "control": {
                "to_dog": get_dist(pred_control, dog_base),
                "to_cat": get_dist(pred_control, cat_base)
            }
        }
        return results

    def visualize(self, results):
        labels = ['To Dog (Ref)', 'To Cat (Target)']
        causal_dists = [results['causal']['to_dog'], results['causal']['to_cat']]
        control_dists = [results['control']['to_dog'], results['control']['to_cat']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, causal_dists, width, label='Causal Heads (Top-3)', color='#4CAF50')
        rects2 = ax.bar(x + width/2, control_dists, width, label='Control Heads (Bottom-3)', color='#FF5722')
        
        ax.set_ylabel('L2 Latent Distance')
        ax.set_title('Surgical Semantic Crossover: Causal vs Control Circuits')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("surgical_crossover.png")
        logger.info("Experiment complete. Result saved to surgical_crossover.png")

if __name__ == "__main__":
    cat_p = "C:\\Users\\suruc\\.gemini\\antigravity\\brain\\c829103c-d6d2-4f4a-b6e2-d671288c5eef\\cat_sample_1776423263206.png"
    dog_p = "C:\\Users\\suruc\\.gemini\\antigravity\\brain\\c829103c-d6d2-4f4a-b6e2-d671288c5eef\\dog_sample_1776423281892.png"
    
    # Use the Huge backbone we just verified
    checkpoint = "vith14_in1k_ep300.pth.tar"
    if not os.path.exists(checkpoint):
        checkpoint = "ijepa_mini.pth" # Fallback
        
    analyzer = SurgicalCounterfactual(cat_p, dog_p)
    # Re-initialize with Huge if available
    if os.path.exists(checkpoint):
        analyzer.adapter = IJEPAAdapter.from_checkpoint(checkpoint)
        analyzer.wm = HookedWorldModel(analyzer.adapter, analyzer.adapter.config)
        analyzer.config = analyzer.adapter.config # Sync config
        
    results = analyzer.run_experiment()
    analyzer.visualize(results)
