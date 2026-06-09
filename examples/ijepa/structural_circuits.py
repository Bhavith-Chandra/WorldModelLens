import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookContext, HookPoint
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import os

class IJEPAStructuralTracer:
    def __init__(self, target_id=114, control_target_id=42):
        """Args:
            target_id: Primary patch to explain.
            control_target_id: A different patch used as a negative control.
                               A truly causal circuit should show HIGH importance
                               for target_id but LOW importance for control_target_id.
        """
        self.target_id = target_id
        self.control_target_id = control_target_id
        self.raw_img = get_sample_image()
        self.img_tensor = preprocess_image(self.raw_img)
        
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384, predictor_depth=4, predictor_heads=6)
        self.adapter = IJEPAAdapter(config)
        
        checkpoint_path = os.path.join(os.path.dirname(__file__), "ijepa_mini.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            self.adapter.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
            
        self.wm = HookedWorldModel(self.adapter, config)
        self.wm.adapter.eval()
        
        # Determine context patches
        self.context_ids, _ = get_ijepa_masks(num_context=80)
        if target_id in self.context_ids: self.context_ids.remove(target_id)
        
        # Sync masks to adapter once
        self.adapter.last_context_ids = self.context_ids
        self.adapter.last_target_ids = [self.target_id]
        
        self.hook_fire_count = 0
        
        # --- Pre-compute mean activations for on-manifold ablation ---
        # We perturb the input with Gaussian noise to get a distribution of activations,
        # then take the mean. This represents E[activation] and stays on-manifold.
        print("Pre-computing mean activations for on-manifold ablation...")
        self._mean_acts = self._compute_mean_activations(n_samples=8)
        print(f"  Cached {len(self._mean_acts)} hook point means.")

    def _compute_mean_activations(self, n_samples=8):
        """Runs n_samples noisy forward passes and records the mean activation at
        each predictor block output. This serves as the on-manifold ablation baseline.
        """
        means = {}  # hook_name -> running mean tensor
        counts = {}

        def make_capture_hook(name):
            def hook(x, ctx):
                if name not in means:
                    means[name] = x.detach().clone()
                    counts[name] = 1
                else:
                    means[name] += x.detach()
                    counts[name] += 1
                return x
            return hook

        depth = len(self.adapter.predictor.blocks)
        num_heads = self.adapter.predictor.blocks[0].attn.num_heads

        # Register capture hooks on all target points
        for i in range(depth):
            self.wm.add_hook(HookPoint(name=f"predictor.block_{i}", fn=make_capture_hook(f"predictor.block_{i}")))
            self.wm.add_hook(HookPoint(name=f"predictor.block_{i}.attn.heads", fn=make_capture_hook(f"predictor.block_{i}.attn.heads")))

        with torch.no_grad():
            for _ in range(n_samples):
                # Add small Gaussian noise to stay near the data manifold
                noisy = self.img_tensor + 0.05 * torch.randn_like(self.img_tensor)
                self.wm.run_with_cache(noisy)

        self.wm.clear_hooks()

        # Finalize means
        return {k: v / counts[k] for k, v in means.items()}

    def _get_mean_act(self, hook_name):
        """Returns the pre-computed mean activation for a given hook point."""
        return self._mean_acts.get(hook_name, None)

    def run_eval(self, expect_hook=False, for_target_id=None):
        """Runs the model for a given target patch and returns MSE.
        
        Args:
            expect_hook: If True, asserts at least one hook fired (guards against silent failures).
            for_target_id: Override the target ID (used for control target comparison).
        """
        target_id = for_target_id if for_target_id is not None else self.target_id
        self.adapter.last_target_ids = [target_id]
        self.hook_fire_count = 0
        
        with torch.no_grad():
            if hasattr(self.wm.adapter, "hooks"):
                self.wm.adapter.hooks = self.wm.hook_registry
            self.wm.adapter.current_timestep = 0
            
            traj, cache = self.wm.run_with_cache(self.img_tensor)
            
            pred = cache.get("z_prior", 0)
            assert pred is not None, "Error: 'z_prior' (prediction) missing from cache."
            
            if expect_hook:
                assert self.hook_fire_count > 0, (
                    "CRITICAL ERROR: No hooks fired during intervention! "
                    "The forward pass is bypassing the HookRegistry. Results are invalid."
                )
            
            target_reps = cache.get("target_encoding", 0)
            if target_reps is not None:
                target_gt = target_reps[0, [target_id], :] if target_reps.dim() == 3 else target_reps[[target_id], :]
            else:
                target_gt = self.adapter.target_encode(self.img_tensor)[:, [target_id], :]
            
            mse = F.mse_loss(pred.squeeze(0), target_gt.squeeze(0)).item()
        
        # Restore primary target
        self.adapter.last_target_ids = [self.target_id]
        return mse, target_gt

    def run_layer_ablation(self):
        """Layer-by-layer causal tracing with mean ablation and control comparison.
        
        For each predictor block:
          - Replace the block's residual output with its mean activation (on-manifold)
          - Measure MSE increase for PRIMARY target (causal signal)
          - Measure MSE increase for CONTROL target (should be low if causal, not generic)
          - Specificity = primary_importance - control_importance
        """
        print(f"Phase 3.1: Layer-by-layer causal tracing for Target {self.target_id}...")
        print(f"           Control target: {self.control_target_id}")

        self.wm.clear_hooks()
        clean_mse, _ = self.run_eval(expect_hook=False)
        control_clean_mse, _ = self.run_eval(expect_hook=False, for_target_id=self.control_target_id)
        
        predictor_depth = len(self.adapter.predictor.blocks)
        results = []

        def make_mean_ablation_hook(hook_name):
            """Returns a hook that replaces activations with their mean.
            This is an on-manifold intervention — the model sees a plausible
            representation, not an all-zeros OOD vector.
            """
            mean_act = self._get_mean_act(hook_name)
            def ablate_hook(x, ctx):
                self.hook_fire_count += 1
                if mean_act is not None and mean_act.shape == x.shape:
                    return mean_act.to(x.device)
                # Fallback: mean over the batch and token dims (still on-manifold)
                return x.mean(dim=(0, 1), keepdim=True).expand_as(x)
            return ablate_hook

        for layer_idx in range(predictor_depth):
            hook_name = f"predictor.block_{layer_idx}"

            # -- Primary target ablation --
            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=hook_name, fn=make_mean_ablation_hook(hook_name)))
            primary_mse, _ = self.run_eval(expect_hook=True)
            primary_importance = (primary_mse - clean_mse) / (clean_mse + 1e-8)

            # -- Control target ablation (same intervention, different target) --
            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=hook_name, fn=make_mean_ablation_hook(hook_name)))
            control_mse, _ = self.run_eval(expect_hook=True, for_target_id=self.control_target_id)
            control_importance = (control_mse - control_clean_mse) / (control_clean_mse + 1e-8)

            # Specificity: how much MORE important is this layer for our target vs the control?
            specificity = primary_importance - control_importance

            results.append({
                "primary": primary_importance,
                "control": control_importance,
                "specificity": specificity,
            })
            print(
                f"  Layer {layer_idx} | "
                f"Primary: {primary_importance:+.4f}  "
                f"Control: {control_importance:+.4f}  "
                f"Specificity: {specificity:+.4f}  "
                f"[Fires: {self.hook_fire_count}]"
            )

        return results, clean_mse

    def run_head_zoom(self, layer_idx):
        """Per-head mean ablation with specificity comparison.
        
        Uses the same on-manifold mean ablation strategy as layer ablation,
        targeting individual attention heads. Heads with HIGH specificity
        (large primary - control difference) are the true causal heads.
        """
        print(f"Phase 3.2: Zooming into Layer {layer_idx} (Greedy Head Ablation, on-manifold)...")

        self.wm.clear_hooks()
        clean_mse, _ = self.run_eval(expect_hook=False)
        control_clean_mse, _ = self.run_eval(expect_hook=False, for_target_id=self.control_target_id)
        
        num_heads = self.adapter.predictor.blocks[layer_idx].attn.num_heads
        head_results = []
        hook_name = f"predictor.block_{layer_idx}.attn.heads"
        mean_heads = self._get_mean_act(hook_name)

        def make_head_hook(h):
            def ablate_head_hook(out_heads, ctx):
                assert out_heads.dim() == 4, (
                    f"Structural Error: Expected [B, heads, N, head_dim], got {out_heads.shape}."
                )
                self.hook_fire_count += 1
                out = out_heads.clone()
                if mean_heads is not None and mean_heads.shape == out_heads.shape:
                    # On-manifold: replace with mean head activation
                    out[:, h, :, :] = mean_heads[:, h, :, :].to(out.device)
                else:
                    # Fallback: mean over tokens for this head
                    out[:, h, :, :] = out_heads[:, h, :, :].mean(dim=1, keepdim=True).expand_as(out_heads[:, h, :, :])
                return out
            return ablate_head_hook

        for head_idx in range(num_heads):
            # Primary
            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=hook_name, fn=make_head_hook(head_idx)))
            primary_mse, _ = self.run_eval(expect_hook=True)
            primary_importance = (primary_mse - clean_mse) / (clean_mse + 1e-8)

            # Control
            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=hook_name, fn=make_head_hook(head_idx)))
            control_mse, _ = self.run_eval(expect_hook=True, for_target_id=self.control_target_id)
            control_importance = (control_mse - control_clean_mse) / (control_clean_mse + 1e-8)

            specificity = primary_importance - control_importance
            head_results.append({
                "primary": primary_importance,
                "control": control_importance,
                "specificity": specificity,
            })
            print(
                f"    Head {head_idx} | "
                f"Primary: {primary_importance:+.4f}  "
                f"Control: {control_importance:+.4f}  "
                f"Specificity: {specificity:+.4f}  "
                f"[Fires: {self.hook_fire_count}]"
            )

        return head_results

    def visualize(self, layer_results, head_results_map):
        """Plots primary, control, and specificity bars side by side.
        
        The KEY causal signal is the SPECIFICITY bar — a layer/head with high
        specificity demonstrably affects THIS target more than a random other target,
        which is the minimum bar for a causal claim.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        colors = {"primary": "#2196F3", "control": "#FF9800", "specificity": "#4CAF50"}

        # -- Layer-level plots --
        ax_l_primary = axes[0, 0]
        ax_l_spec = axes[0, 1]
        n_layers = len(layer_results)
        layers = range(n_layers)

        primary_vals = [r["primary"] for r in layer_results]
        control_vals = [r["control"] for r in layer_results]
        spec_vals = [r["specificity"] for r in layer_results]

        width = 0.35
        ax_l_primary.bar([l - width/2 for l in layers], primary_vals, width, label=f"Target {self.target_id}", color=colors["primary"], alpha=0.85)
        ax_l_primary.bar([l + width/2 for l in layers], control_vals, width, label=f"Control {self.control_target_id}", color=colors["control"], alpha=0.85)
        ax_l_primary.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax_l_primary.set_xlabel("Predictor Block")
        ax_l_primary.set_ylabel("Causal Importance (ΔMSErel)")
        ax_l_primary.set_title("Layer Importance: Primary vs Control\n(same intervention, different target)")
        ax_l_primary.legend()
        ax_l_primary.grid(True, axis='y', alpha=0.3)

        ax_l_spec.bar(layers, spec_vals, color=colors["specificity"], alpha=0.85)
        ax_l_spec.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax_l_spec.set_xlabel("Predictor Block")
        ax_l_spec.set_ylabel("Specificity (Primary − Control)")
        ax_l_spec.set_title("Layer Specificity\n(positive = causally specific to target)")
        ax_l_spec.grid(True, axis='y', alpha=0.3)

        # -- Head-level plots (only first entry in map) --
        ax_h_primary = axes[1, 0]
        ax_h_spec = axes[1, 1]
        for i, (l_idx, scores) in enumerate(head_results_map.items()):
            n_heads = len(scores)
            heads = range(n_heads)
            p_vals = [s["primary"] for s in scores]
            c_vals = [s["control"] for s in scores]
            s_vals = [s["specificity"] for s in scores]

            ax_h_primary.bar([h - width/2 for h in heads], p_vals, width, label=f"Target {self.target_id}", color=colors["primary"], alpha=0.85)
            ax_h_primary.bar([h + width/2 for h in heads], c_vals, width, label=f"Control {self.control_target_id}", color=colors["control"], alpha=0.85)
            ax_h_primary.axhline(0, color='white', linewidth=0.8, linestyle='--')
            ax_h_primary.set_xlabel("Head Index")
            ax_h_primary.set_ylabel("Causal Importance (ΔMSErel)")
            ax_h_primary.set_title(f"Head Importance (Layer {l_idx}): Primary vs Control")
            ax_h_primary.legend()
            ax_h_primary.grid(True, axis='y', alpha=0.3)

            ax_h_spec.bar(heads, s_vals, color=colors["specificity"], alpha=0.85)
            ax_h_spec.axhline(0, color='white', linewidth=0.8, linestyle='--')
            ax_h_spec.set_xlabel("Head Index")
            ax_h_spec.set_ylabel("Specificity (Primary − Control)")
            ax_h_spec.set_title(f"Head Specificity (Layer {l_idx})\n(bars above 0 = causally specific heads)")
            ax_h_spec.grid(True, axis='y', alpha=0.3)

        plt.suptitle(
            f"I-JEPA Causal Circuit Discovery (On-Manifold Ablation)\n"
            f"Target: {self.target_id} | Control: {self.control_target_id}",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        
        if os.environ.get("SAVE_PLOT"):
            plt.savefig("structural_circuits_causal.png", dpi=150)
            print("Causal circuit plot saved to structural_circuits_causal.png")
        else:
            plt.show()

if __name__ == "__main__":
    # Target=114 is the primary patch to explain
    # Control=42 is a spatially different patch used as a negative control
    # A causally specific layer/head will show high importance for 114
    # but NOT for 42 (or the reverse, if 42 is equally affected, the circuit is generic)
    tracer = IJEPAStructuralTracer(target_id=114, control_target_id=42)
    layer_results, _ = tracer.run_layer_ablation()
    critical_layer = np.argmax([r["specificity"] for r in layer_results])
    print(f"\nMost causally specific layer: {critical_layer} "
          f"(specificity={layer_results[critical_layer]['specificity']:+.4f})")
    head_results = tracer.run_head_zoom(critical_layer)
    tracer.visualize(layer_results, {critical_layer: head_results})
