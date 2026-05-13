#examples/ijepa/causal_evaluation.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookPoint
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import numpy as np

class IJEPAFaithfulnessEvaluator:
    def __init__(self, model: IJEPAAdapter, img_tensor: torch.Tensor):
        # Wrap in HookedWorldModel for deep activation access
        self.wm = HookedWorldModel(model, model.config)
        self.model = model # Reference to adapter
        self.img_tensor = img_tensor
        self.patch_size = 16
        self.num_patches = 196
        
        # Pre-compute ground truth latents and attention using Hooks
        with torch.no_grad():
            # Run with cache to capture all activations
            traj, cache = self.wm.run_with_cache(img_tensor)
            
            # Extract target encoding (final output of target encoder)
            self.full_latents = cache["target_encoding", 0] # [B, N_full, C]
            
            # For attention matrix, we look at the last block of the context encoder
            last_block_name = f"context_encoder.block_{model.config.n_layers - 1}"
            if (last_block_name, 0) in cache:
                self.attn_matrix = model.context_encoder.blocks[-1].attn.last_attn_weights[0].mean(0).cpu().numpy()
            else:
                self.attn_matrix = np.zeros((196, 196))
                
            # Pre-compute patch embeddings before positional encoding
            self.patch_embeddings = self.model.context_encoder.patch_embed(self.img_tensor)
            # Mean patch embedding — used as the on-manifold replacement in mean ablation
            self.mean_patch_emb = self.patch_embeddings.mean(dim=1, keepdim=True) # [1, 1, C]
            self.zero_patch_emb = torch.zeros_like(self.mean_patch_emb)

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Computes MSE, L2, and Cosine Similarity."""
        mse = F.mse_loss(pred, target).item()
        l2 = torch.norm(pred - target).item()
        cos = F.cosine_similarity(pred.flatten(), target.flatten(), dim=0).item()
        return {"mse": mse, "l2": l2, "cos": cos}

    def run_intervention(
        self, 
        target_id: int, 
        active_context_ids: List[int], 
        ablated_ids: List[int], 
        baseline="mean"
    ) -> Dict[str, float]:
        """Runs the model with specific context patches active.
        
        Crucially, ablated patches are replaced with the MEAN patch embedding
        rather than being dropped. This holds total context length constant,
        isolating the effect of *information content* vs. *context size*.
        This is the key correction over the naive patch-drop approach.
        """
        # Set masks on the adapter
        self.model.last_context_ids = active_context_ids + ablated_ids  # Keep full context length
        self.model.last_target_ids = [target_id]
        
        with torch.no_grad():
            context_latents, _ = self.model.encode(self.img_tensor)
            
            # If ablated_ids is non-empty, replace their latent slots with mean embedding
            # This keeps the sequence length fixed (no information about *how many* got removed)
            if ablated_ids and baseline == "mean":
                all_ids = active_context_ids + ablated_ids
                ablated_positions = [all_ids.index(i) for i in ablated_ids if i in all_ids]
                if ablated_positions and context_latents.shape[1] > max(ablated_positions):
                    mean_lat = context_latents.mean(dim=1, keepdim=True)
                    for pos in ablated_positions:
                        context_latents[:, pos, :] = mean_lat.squeeze(1)
            
            pred = self.model.dynamics(context_latents)
            
            target_gt = self.full_latents[[target_id], :] # [1, C]
            pred = pred.squeeze(0) # [1, C]
            
        return self.compute_metrics(pred, target_gt)

    def evaluate_faithfulness(self, target_id: int, context_ids: List[int], n_steps=20, n_random=5):
        """Runs ablation and reconstruction tests for a target."""
        # Get attributions from predictor's perspective (using predictor attn)
        self.model.last_context_ids = context_ids
        self.model.last_target_ids = [target_id]
        
        with torch.no_grad():
            ctx_latents, _ = self.model.encode(self.img_tensor)
            _ = self.model.dynamics(ctx_latents)
            attn = self.model.predictor.get_last_self_attention()[0].mean(0)
            # Index of target in the combined [context + target] seq is the last one
            target_to_context_attn = attn[-1, :len(context_ids)].cpu().numpy()

        sorted_indices = np.argsort(target_to_context_attn)[::-1]
        top_context_ids = [context_ids[i] for i in sorted_indices]
        random_orders = [np.random.permutation(context_ids) for _ in range(n_random)]
        bottom_context_ids = top_context_ids[::-1]

        results = {
            "ablation": {"top": [], "random": [], "bottom": []},
            "reconstruction": {"top": [], "random": []}
        }

        # Ablation Test
        for k in range(0, min(n_steps, len(context_ids)), max(1, n_steps // 20)):
            # Top-K removal
            ablated = top_context_ids[:k]
            active = [pid for pid in context_ids if pid not in ablated]
            results["ablation"]["top"].append(self.run_intervention(target_id, active, ablated))
            
            # Bottom-K removal
            ablated_b = bottom_context_ids[:k]
            active_b = [pid for pid in context_ids if pid not in ablated_b]
            results["ablation"]["bottom"].append(self.run_intervention(target_id, active_b, ablated_b))
            
            # Random removal
            rand_metrics = []
            for order in random_orders:
                ablated_r = order[:k].tolist()
                active_r = [pid for pid in context_ids if pid not in ablated_r]
                rand_metrics.append(self.run_intervention(target_id, active_r, ablated_r))
            
            avg_rand = {m: np.mean([r[m] for r in rand_metrics]) for m in ["mse", "l2", "cos"]}
            results["ablation"]["random"].append(avg_rand)

        # Reconstruction Test
        for k in range(1, min(n_steps, len(context_ids)), max(1, n_steps // 20)):
            # Top-K addition
            active = top_context_ids[:k]
            results["reconstruction"]["top"].append(self.run_intervention(target_id, active, []))
            
            # Random addition
            rand_metrics = []
            for order in random_orders:
                active_r = order[:k].tolist()
                rand_metrics.append(self.run_intervention(target_id, active_r, []))
            
            avg_rand = {m: np.mean([r[m] for r in rand_metrics]) for m in ["mse", "l2", "cos"]}
            results["reconstruction"]["random"].append(avg_rand)

        return results

def calculate_auc(values):
    return np.trapezoid(values) / len(values) if values else 0

def compute_monotonicity(values):
    if len(values) < 2: return 0
    diffs = np.diff(values)
    increasing = np.sum(diffs > 0) / len(diffs)
    decreasing = np.sum(diffs < 0) / len(diffs)
    return max(increasing, decreasing)

def find_predictive_circuit(metric_values, threshold_ratio=0.9):
    if not metric_values: return 0
    best_val = min(metric_values)
    target_val = best_val + (1 - threshold_ratio) * (metric_values[0] - best_val)
    for i, v in enumerate(metric_values):
        if v <= target_val:
            return i + 1
    return len(metric_values)

def compute_faithfulness_gap(results_list, metric="mse"):
    """Computes the area-under-curve gap between top-K and random-K ablation curves.
    
    This is the standard faithfulness metric used in mechanistic interpretability:
    - A LARGE gap means: removing the model's own highest-attribution patches
      degrades performance MUCH faster than removing random patches.
    - This is evidence that the attribution ranking is causally meaningful.
    - Gap ≈ 0 means the attribution ranking is no better than random (not causal).
    
    Returns:
        gap: AUC(top) - AUC(random), higher = more faithful attribution
        monotonicity_top: fraction of steps where top-K removal increases error
        circuit_size: smallest k achieving 90% of total potential degradation
    """
    top_curves = []
    rand_curves = []
    for res in results_list:
        top_curves.append([x[metric] for x in res["ablation"]["top"]])
        rand_curves.append([x[metric] for x in res["ablation"]["random"]])
    
    top_avg = np.mean(top_curves, axis=0)
    rand_avg = np.mean(rand_curves, axis=0)
    
    auc_top = calculate_auc(top_avg)
    auc_rand = calculate_auc(rand_avg)
    gap = auc_top - auc_rand  # positive = top-K degrades faster = faithful
    monotonicity = compute_monotonicity(top_avg)
    circuit_size = find_predictive_circuit(list(top_avg))
    
    print("\n" + "="*50)
    print("FAITHFULNESS SUMMARY")
    print("="*50)
    print(f"  AUC gap (top vs random): {gap:+.6f}")
    print(f"  {'FAITHFUL' if gap > 0 else 'NOT FAITHFUL'}: top-K attribution {'degrades' if gap > 0 else 'does NOT degrade'} performance faster than random")
    print(f"  Monotonicity (top curve): {monotonicity:.2%} of steps increase error")
    print(f"  Minimal circuit size: {circuit_size} patches (90% degradation threshold)")
    print("="*50)
    
    return {"gap": gap, "monotonicity": monotonicity, "circuit_size": circuit_size,
            "auc_top": auc_top, "auc_rand": auc_rand}

def plot_faithfulness(results_list, metric="mse"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    types = ["top", "random", "bottom"]
    
    # Ablation
    for t_type in types:
        all_vals = []
        for res in results_list:
            all_vals.append([x[metric] for x in res["ablation"][t_type]])
        avg_vals = np.mean(all_vals, axis=0)
        color = "red" if t_type == "top" else "blue" if t_type == "bottom" else "gray"
        ax1.plot(avg_vals, label=f"{t_type.capitalize()}-K", color=color, lw=2)

    ax1.set_title(f"Ablation Suite ({metric.upper()})")
    ax1.set_xlabel("Patches Intervened")
    ax1.set_ylabel(metric.upper())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reconstruction
    for t_type in ["top", "random"]:
        all_vals = []
        for res in results_list:
            all_vals.append([x[metric] for x in res["reconstruction"][t_type]])
        avg_vals = np.mean(all_vals, axis=0)
        color = "green" if t_type == "top" else "gray"
        ax2.plot(avg_vals, label=f"{t_type.capitalize()}-K", color=color, lw=2)

    ax2.set_title(f"Reconstruction Suite ({metric.upper()})")
    ax2.set_xlabel("Patches Provided")
    ax2.set_ylabel(metric.upper())
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if os.environ.get("SAVE_PLOT"):
        plt.savefig("causal_evaluation.png")
        print("Plot saved to causal_evaluation.png")
    else:
        plt.show()

if __name__ == "__main__":
    import os
    
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384)
    model = IJEPAAdapter(config)
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), "ijepa_mini.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)
    
    evaluator = IJEPAFaithfulnessEvaluator(model, img_tensor)
    
    context_ids, target_ids = get_ijepa_masks(num_context=80, num_target=3)
    
    print(f"Starting Multi-Target Causal Validation...")
    all_results = []
    for tid in target_ids:
        print(f"  Validating Target {tid}...")
        res = evaluator.evaluate_faithfulness(tid, context_ids, n_steps=30)
        all_results.append(res)
    
    # Compute and print the faithfulness gap — the key causal metric
    faith_stats = compute_faithfulness_gap(all_results, metric="mse")
    
    plot_faithfulness(all_results, metric="mse")
