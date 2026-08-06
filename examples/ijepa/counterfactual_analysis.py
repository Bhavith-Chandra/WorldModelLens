#examples/ijepa/counterfactual_analysis.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookContext, HookPoint
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import os

class IJEPACounterfactualAnalyzer:
    def __init__(self):
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384)
        self.adapter = IJEPAAdapter(config)
        checkpoint_path = "ijepa_mini.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            self.adapter.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
            
        self.wm = HookedWorldModel(self.adapter, config)
        self.wm.adapter.eval()
        
        self.img_a_raw = get_sample_image()
        self.img_a = preprocess_image(self.img_a_raw)
        self.img_b_raw = self.img_a_raw.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        self.img_b = preprocess_image(self.img_b_raw)
        
        self.hook_fire_count = 0

    def _run_verified_pred(self, img_tensor, expect_hook=False):
        self.hook_fire_count = 0
        with torch.no_grad():
            if hasattr(self.wm.adapter, "hooks"):
                registry = self.wm.hook_registry
                self.wm.adapter.hooks = registry
                self.wm.adapter.context_encoder.hooks = registry
                self.wm.adapter.context_encoder.current_timestep = 0

                self.wm.adapter.target_encoder.hooks = registry
                self.wm.adapter.target_encoder.current_timestep = 0

                self.wm.adapter.predictor.hooks = registry
                self.wm.adapter.predictor.current_timestep = 0
            #self.wm.adapter.current_timestep = 0
            
            traj, cache = self.wm.run_with_cache(img_tensor)
            pred = cache.get("z_prior", 0)
            assert pred is not None, "Error: 'z_prior' absent from cache."
            
            if expect_hook:
                assert self.hook_fire_count > 0, "Causal Failure: Hook did not fire during forward pass!"
                
            return pred.squeeze(0), cache

    def run_spatial_sensitivity(self, grid_size=14, step=2):
        print("Running Verified Spatial Sensitivity Analysis...")
        sensitivity_map = np.zeros((grid_size, grid_size))
        orig_ctx = self.adapter.last_context_ids
        orig_tgt = self.adapter.last_target_ids
        
        try:
            context_ids = [i for i in range(grid_size**2 // 2, grid_size**2)]
            block_w, block_h = 2, 2
            
            for r in range(0, grid_size // 2, step):
                for c in range(0, grid_size - block_w, step):
                    target_ids = []
                    for i in range(block_h):
                        for j in range(block_w):
                            target_ids.append((r+i)*grid_size + (c+j))
                    
                    self.adapter.last_context_ids = context_ids
                    self.adapter.last_target_ids = target_ids
                    
                    pred, cache = self._run_verified_pred(self.img_a)
                    target_reps = cache.get("target_encoding", timestep=0)
                    assert target_reps is not None, "target_encoding missing from cache"
                    target_gt = target_reps[0, target_ids, :] if target_reps.dim() == 3 else target_reps[target_ids, :]
                    mse = F.mse_loss(pred, target_gt).item()
                    
                    for i in range(block_h):
                        for j in range(block_w):
                            sensitivity_map[r+i, c+j] = mse
        finally:
            self.adapter.last_context_ids = orig_ctx
            self.adapter.last_target_ids = orig_tgt
        return sensitivity_map

    def run_semantic_crossover(self, num_test_patches=4):
        print(f"Running Verified Semantic Crossover...")
        test_patches = [50, 52, 64, 66]
        context_ids = list(range(196 // 2))
        
        # 1. Capture B Latents
        activations_b = {}
        def capture_hook(x, ctx):
            self.hook_fire_count += 1
            activations_b['val'] = x.detach()
            return x
            
        self.wm.clear_hooks()
        self.wm.add_hook(HookPoint(name="context_encoder.norm", fn=capture_hook))
        
        self.adapter.last_context_ids = context_ids
        self.adapter.last_target_ids = test_patches
        
        _, cache_b = self._run_verified_pred(self.img_b, expect_hook=True)
        encoded_b = activations_b['val']
        target_reps_b = cache_b.get("target_encoding", timestep=0)
        assert target_reps_b is not None, "target_encoding missing from cache" 
        gt_b_latent = target_reps_b[0, test_patches, :] if target_reps_b.dim() == 3 else target_reps_b[test_patches, :]
        
        self.wm.clear_hooks()
        _, cache_a = self._run_verified_pred(self.img_a, expect_hook=False)
        target_reps_a = cache_a.get("target_encoding", timestep=0)
        assert target_reps_a is not None, "target_encoding missing from cache"
        gt_a_latent = target_reps_a[0, test_patches, :] if target_reps_a.dim() == 3 else target_reps_a[test_patches, :]

        # 2. Intervention
        def swap_context_hook(x, ctx):
            self.hook_fire_count += 1
            return encoded_b
            
        self.wm.clear_hooks()
        self.wm.add_hook(HookPoint(name="context_encoder.norm", fn=swap_context_hook))
        
        pred_cross, _ = self._run_verified_pred(self.img_a, expect_hook=True)
            
        dist_to_a = torch.norm(pred_cross - gt_a_latent, dim=-1).mean().item()
        dist_to_b = torch.norm(pred_cross - gt_b_latent, dim=-1).mean().item()
            
        return {
            "dist_a": dist_to_a, "dist_b": dist_to_b,
            "alignment": "Semantic (B)" if dist_to_b < dist_to_a else "Structural (A)"
        }

    def visualize(self, sensitivity_map, crossover_results):
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Spatial Sensitivity (Heatmap)
        plt.subplot(2, 3, 1)
        plt.imshow(sensitivity_map, cmap='hot')
        plt.colorbar(label='MSE')
        plt.title("Spatial Mask Sensitivity")
        
        # 2. Image A (Source of Mask)
        plt.subplot(2, 3, 2)
        plt.imshow(self.img_a_raw.resize((224, 224)))
        plt.title("Image A (Mask Reference)")
        plt.axis('off')
        
        # 3. Image B (Source of Context)
        plt.subplot(2, 3, 3)
        plt.imshow(self.img_b_raw.resize((224, 224)))
        plt.title("Image B (Semantic Content)")
        plt.axis('off')
        
        # 4. Semantic Crossover (Bar Chart)
        plt.subplot(2, 3, 4)
        methods = ['Dist to A', 'Dist to B']
        dists = [crossover_results['dist_a'], crossover_results['dist_b']]
        plt.bar(methods, dists, color=['#2196F3', '#4CAF50'])
        plt.ylabel("Ex-post L2 Latent Distance")
        plt.title(f"Semantic Crossover Result")
        
        # 5. Explanatory Text
        plt.subplot(2, 3, 5)
        plt.text(0.1, 0.3, f"Intervention Result: {crossover_results['alignment']}\n\n"
                 f"Caveat: Correlation exists (A/B are flips).\n"
                 f"consistent with causal routing under current interventions\n\n"
                 f"Winner {crossover_results['alignment']} suggests whether\n"
                 f"predictor follows masked locations (A)\n"
                 f"or injected semantic concepts (B).", fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='#4CAF50'))
        plt.axis('off')

        plt.suptitle("IJEPACounterfactualAnalyzer | Verified Visual Foundations", fontsize=18, y=0.98)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        
        if os.environ.get("SAVE_PLOT"):
            plt.savefig("counterfactual_analysis_verified.png")
            print("Verified counterfactual analysis saved to counterfactual_analysis_verified.png")
        else:
            plt.show()

if __name__ == "__main__":
    analyzer = IJEPACounterfactualAnalyzer()
    sens_map = analyzer.run_spatial_sensitivity()
    cross_res = analyzer.run_semantic_crossover()
    analyzer.visualize(sens_map, cross_res)
