import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from typing import List, Tuple, Dict, Any, Optional

from world_model_lens import HookedWorldModel
from world_model_lens.core.hooks import HookPoint, HookContext
from world_model_lens.analysis.attribution import (
    BaseAttribution, 
    extract_attention_weights
)


def compute_auc(k_values: List[int], mses: List[float]) -> float:
    """Computes Normalized Area Under the Curve (AUC) for deletion/insertion curves.
    
    Note on Polarity Framing:
    Unlike standard probability-based RISE insertion/deletion metrics (where deletion AUC lower is better
    and insertion AUC higher is better), here the metric is prediction error (MSE).
    - Higher deletion AUC indicates better attribution (deleting important context spikes MSE fast).
    - Lower insertion AUC indicates better attribution (restoring important context drops MSE fast).
    """
    if len(k_values) < 2:
        return 0.0
    x = np.array(k_values, dtype=float)
    y = np.array(mses, dtype=float)
    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(y, x)
    else:
        auc = np.trapz(y, x)
    norm_auc = auc / (x[-1] - x[0]) if (x[-1] - x[0]) > 0 else 0.0
    return float(norm_auc)


def compute_bootstrap_ci(data: np.ndarray, n_bootstraps: int = 1000, ci_level: float = 0.95) -> Tuple[float, float, float]:
    """Computes mean and bootstrap confidence interval (low, high) across the sample dimension."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(data))
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return mean_val, low, high


class PatchKnockoutEvaluator:
    """Evaluates causal impact of knocking out patches based on IG vs Attention vs Random.
    
    Converts attribution claims into interventional causal claims by verifying
    that deleting high-IG patches degrades prediction MSE significantly more (higher Deletion AUC)
    and restoring high-IG patches recovers prediction MSE significantly faster (lower Insertion AUC)
    compared to Attention or Random patch selection.
    """
    
    def __init__(
        self, 
        adapter: Any, 
        k_values: List[int] = [1, 3, 5, 10, 20],
        patch_ablation_mode: str = "zero",
        n_random_seeds: int = 20,
        dataset_mean_patch: Optional[torch.Tensor] = None
    ):
        """Initialize the evaluator.
        
        Args:
            adapter: Loaded World Model adapter (e.g. IJEPAAdapter).
            k_values: Number of top patches to evaluate.
            patch_ablation_mode: Input patch replacement mode ("zero" or "dataset_mean_patch").
            n_random_seeds: Number of random permutations per sample (M=20 default).
            dataset_mean_patch: Optional precomputed dataset mean patch embedding.
        """
        self.adapter = adapter
        self.adapter.eval()
        self.k_values = k_values
        self.patch_ablation_mode = patch_ablation_mode
        self.n_random_seeds = n_random_seeds
        self.dataset_mean_patch = dataset_mean_patch
        
    @torch.no_grad()
    def _forward_score_with_knockout(
        self, 
        img_tensor: torch.Tensor, 
        context_ids: List[int], 
        target_id: int, 
        target_gt: torch.Tensor,
        patches_to_knockout: List[int]
    ) -> float:
        """Compute prediction MSE when specific input context patches are deleted/ablated."""
        device = img_tensor.device
        
        patch_emb = self.adapter.context_encoder.patch_embed(img_tensor).clone()
        
        for patch_idx in patches_to_knockout:
            if patch_idx in context_ids:
                if self.patch_ablation_mode == "dataset_mean_patch" and self.dataset_mean_patch is not None:
                    patch_emb[:, patch_idx, :] = self.dataset_mean_patch.to(device)
                else:
                    patch_emb[:, patch_idx, :] = 0.0
                
        ctx_ids_t = torch.tensor(context_ids, device=device)
        
        ctx_emb = patch_emb[:, ctx_ids_t, :]
        pos = self.adapter.context_encoder.pos_embed[:, ctx_ids_t, :]
        ctx_with_pos = ctx_emb + pos
        
        ctx_latents = self.adapter.context_encoder.forward_blocks(ctx_with_pos)
        if hasattr(self.adapter, "predictor"):
            pred = self.adapter.predictor(ctx_latents, context_ids, [target_id])
        elif hasattr(self.adapter, "decoder"):
            pred = self.adapter.decoder(ctx_latents, context_ids, [target_id])
        else:
            pred = self.adapter.dynamics(ctx_latents)
        
        mse = F.mse_loss(pred.squeeze(1), target_gt).item()
        return mse

    @torch.no_grad()
    def _forward_score_with_insertion(
        self,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
        target_gt: torch.Tensor,
        patches_to_restore: List[int]
    ) -> float:
        """Compute prediction MSE when starting from fully ablated context and restoring top-K patches."""
        device = img_tensor.device
        
        patch_emb = self.adapter.context_encoder.patch_embed(img_tensor).clone()
        
        # Fully ablate all context patches first
        for patch_idx in context_ids:
            if self.patch_ablation_mode == "dataset_mean_patch" and self.dataset_mean_patch is not None:
                patch_emb[:, patch_idx, :] = self.dataset_mean_patch.to(device)
            else:
                patch_emb[:, patch_idx, :] = 0.0
                
        # Restore only top-K selected patches
        original_patch_emb = self.adapter.context_encoder.patch_embed(img_tensor)
        for patch_idx in patches_to_restore:
            if patch_idx in context_ids:
                patch_emb[:, patch_idx, :] = original_patch_emb[:, patch_idx, :]
                
        ctx_ids_t = torch.tensor(context_ids, device=device)
        
        ctx_emb = patch_emb[:, ctx_ids_t, :]
        pos = self.adapter.context_encoder.pos_embed[:, ctx_ids_t, :]
        ctx_with_pos = ctx_emb + pos
        
        ctx_latents = self.adapter.context_encoder.forward_blocks(ctx_with_pos)
        if hasattr(self.adapter, "predictor"):
            pred = self.adapter.predictor(ctx_latents, context_ids, [target_id])
        elif hasattr(self.adapter, "decoder"):
            pred = self.adapter.decoder(ctx_latents, context_ids, [target_id])
        else:
            pred = self.adapter.dynamics(ctx_latents)
        
        mse = F.mse_loss(pred.squeeze(1), target_gt).item()
        return mse

    def evaluate_sample(
        self,
        wm: HookedWorldModel,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
        attr_scores: np.ndarray,
        layer_idx: int = -1,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Evaluate patch deletion and insertion curves for a single sample across IG, Attention, and Random baseline."""
        device = next(self.adapter.parameters()).device
        img_tensor = img_tensor.to(device)
        
        if hasattr(self.adapter, "target_encode"):
            target_gt = self.adapter.target_encode(img_tensor)[:, [target_id], :].squeeze(0).detach()
        else:
            target_gt = self.adapter.target_encoder(img_tensor)[:, [target_id], :].squeeze(0).detach()
        
        baseline_mse = self._forward_score_with_knockout(
            img_tensor, context_ids, target_id, target_gt, patches_to_knockout=[]
        )
        
        attn_weights = extract_attention_weights(
            wm, img_tensor, context_ids, target_id, layer_idx=layer_idx, head_idx=None
        )
        
        sorted_attr_idx = np.argsort(attr_scores)[::-1]
        sorted_attn_idx = np.argsort(attn_weights)[::-1]
        
        # M=n_random_seeds permutations for stochastic random baseline null distribution
        rng = np.random.RandomState(seed)
        random_perms = [rng.permutation(len(context_ids)) for _ in range(self.n_random_seeds)]
        
        results = {
            "baseline_mse": baseline_mse,
            "k_values": self.k_values,
            "ig_deletion_mses": [],
            "attn_deletion_mses": [],
            "random_deletion_mses": [], # Shape: [n_random_seeds, len(k_values)]
            "ig_insertion_mses": [],
            "attn_insertion_mses": [],
            "random_insertion_mses": [] # Shape: [n_random_seeds, len(k_values)]
        }
        
        # Initialize random baseline arrays
        random_del_grid = np.zeros((self.n_random_seeds, len(self.k_values)))
        random_ins_grid = np.zeros((self.n_random_seeds, len(self.k_values)))
        
        for k_idx, k in enumerate(self.k_values):
            top_k_attr_patches = [context_ids[i] for i in sorted_attr_idx[:k]]
            top_k_attn_patches = [context_ids[i] for i in sorted_attn_idx[:k]]
            
            # 1. Integrated Gradients step (deterministic)
            ig_del = self._forward_score_with_knockout(img_tensor, context_ids, target_id, target_gt, top_k_attr_patches)
            ig_ins = self._forward_score_with_insertion(img_tensor, context_ids, target_id, target_gt, top_k_attr_patches)
            results["ig_deletion_mses"].append(ig_del)
            results["ig_insertion_mses"].append(ig_ins)
            
            # 2. Attention step (deterministic)
            attn_del = self._forward_score_with_knockout(img_tensor, context_ids, target_id, target_gt, top_k_attn_patches)
            attn_ins = self._forward_score_with_insertion(img_tensor, context_ids, target_id, target_gt, top_k_attn_patches)
            results["attn_deletion_mses"].append(attn_del)
            results["attn_insertion_mses"].append(attn_ins)
            
            # 3. Random baseline steps (M seeds)
            for m_seed in range(self.n_random_seeds):
                rand_perm = random_perms[m_seed]
                top_k_rand_patches = [context_ids[i] for i in rand_perm[:k]]
                
                rd_del = self._forward_score_with_knockout(img_tensor, context_ids, target_id, target_gt, top_k_rand_patches)
                rd_ins = self._forward_score_with_insertion(img_tensor, context_ids, target_id, target_gt, top_k_rand_patches)
                
                random_del_grid[m_seed, k_idx] = rd_del
                random_ins_grid[m_seed, k_idx] = rd_ins

        results["random_deletion_mses"] = random_del_grid.tolist()
        results["random_insertion_mses"] = random_ins_grid.tolist()

        # AUC calculations
        results["ig_deletion_auc"] = compute_auc(self.k_values, results["ig_deletion_mses"])
        results["attn_deletion_auc"] = compute_auc(self.k_values, results["attn_deletion_mses"])
        
        results["ig_insertion_auc"] = compute_auc(self.k_values, results["ig_insertion_mses"])
        results["attn_insertion_auc"] = compute_auc(self.k_values, results["attn_insertion_mses"])

        # Per-seed random AUCs
        rand_del_aucs = [compute_auc(self.k_values, random_del_grid[m, :]) for m in range(self.n_random_seeds)]
        rand_ins_aucs = [compute_auc(self.k_values, random_ins_grid[m, :]) for m in range(self.n_random_seeds)]
        
        results["random_deletion_auc"] = rand_del_aucs
        results["random_insertion_auc"] = rand_ins_aucs

        # Sample-level diagnostic metrics (Z-score and empirical percentile rank)
        mean_rand_del_auc = float(np.mean(rand_del_aucs))
        std_rand_del_auc = float(np.std(rand_del_aucs)) if len(rand_del_aucs) > 1 else 1e-5
        if std_rand_del_auc == 0:
            std_rand_del_auc = 1e-5

        mean_rand_ins_auc = float(np.mean(rand_ins_aucs))
        std_rand_ins_auc = float(np.std(rand_ins_aucs)) if len(rand_ins_aucs) > 1 else 1e-5
        if std_rand_ins_auc == 0:
            std_rand_ins_auc = 1e-5

        results["ig_random_deletion_zscore"] = float((results["ig_deletion_auc"] - mean_rand_del_auc) / std_rand_del_auc)
        results["ig_random_deletion_percentile"] = float(np.mean(np.array(rand_del_aucs) < results["ig_deletion_auc"]))

        results["ig_random_insertion_zscore"] = float((results["ig_insertion_auc"] - mean_rand_ins_auc) / std_rand_ins_auc)
        results["ig_random_insertion_percentile"] = float(np.mean(np.array(rand_ins_aucs) < results["ig_insertion_auc"]))

        return results

    def aggregate_dataset_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates sample-level deletion and insertion metrics with bootstrap CIs, paired t-tests, and outlier fractions."""
        if not sample_results:
            return {}
            
        k_values = sample_results[0]["k_values"]
        n_samples = len(sample_results)
        
        # Primary arrays across n_samples (deterministic metrics)
        ig_del_auc = np.array([r["ig_deletion_auc"] for r in sample_results])
        attn_del_auc = np.array([r["attn_deletion_auc"] for r in sample_results])
        
        ig_ins_auc = np.array([r["ig_insertion_auc"] for r in sample_results])
        attn_ins_auc = np.array([r["attn_insertion_auc"] for r in sample_results])

        # Random baseline seed-means per sample for image-level paired testing
        rand_del_auc_seed_means = np.array([np.mean(r["random_deletion_auc"]) for r in sample_results])
        rand_ins_auc_seed_means = np.array([np.mean(r["random_insertion_auc"]) for r in sample_results])

        # Paired t-tests across n_samples
        _, p_del_ig_vs_attn = stats.ttest_rel(ig_del_auc, attn_del_auc)
        _, p_del_ig_vs_rand = stats.ttest_rel(ig_del_auc, rand_del_auc_seed_means)
        
        _, p_ins_ig_vs_attn = stats.ttest_rel(ig_ins_auc, attn_ins_auc)
        _, p_ins_ig_vs_rand = stats.ttest_rel(ig_ins_auc, rand_ins_auc_seed_means)

        # Wilcoxon signed-rank tests across n_samples
        try:
            _, p_wilc_del_ig_attn = stats.wilcoxon(ig_del_auc, attn_del_auc)
            _, p_wilc_del_ig_rand = stats.wilcoxon(ig_del_auc, rand_del_auc_seed_means)
            _, p_wilc_ins_ig_attn = stats.wilcoxon(ig_ins_auc, attn_ins_auc)
            _, p_wilc_ins_ig_rand = stats.wilcoxon(ig_ins_auc, rand_ins_auc_seed_means)
        except Exception:
            p_wilc_del_ig_attn, p_wilc_del_ig_rand = 1.0, 1.0
            p_wilc_ins_ig_attn, p_wilc_ins_ig_rand = 1.0, 1.0

        # Outlier fractions
        del_zscores = np.array([r["ig_random_deletion_zscore"] for r in sample_results])
        del_percentiles = np.array([r["ig_random_deletion_percentile"] for r in sample_results])
        
        ins_zscores = np.array([r["ig_random_insertion_zscore"] for r in sample_results])
        ins_percentiles = np.array([r["ig_random_insertion_percentile"] for r in sample_results])

        frac_del_z_outlier = float(np.mean(del_zscores >= 2.0))
        frac_del_p_outlier = float(np.mean(del_percentiles >= 0.95))
        
        frac_ins_z_outlier = float(np.mean(ins_zscores <= -2.0))
        frac_ins_p_outlier = float(np.mean(ins_percentiles <= 0.05))

        agg = {
            "n_samples": n_samples,
            "k_values": k_values,
            "baseline_mse_mean": float(np.mean([r["baseline_mse"] for r in sample_results])),
            "deletion_auc": {
                "ig_auc": {"mean": float(np.mean(ig_del_auc)), "ci_95": list(compute_bootstrap_ci(ig_del_auc)[1:])},
                "attn_auc": {"mean": float(np.mean(attn_del_auc)), "ci_95": list(compute_bootstrap_ci(attn_del_auc)[1:])},
                "random_auc": {"mean": float(np.mean(rand_del_auc_seed_means)), "ci_95": list(compute_bootstrap_ci(rand_del_auc_seed_means)[1:])},
                "p_val_ttest_ig_vs_attn": float(p_del_ig_vs_attn),
                "p_val_ttest_ig_vs_rand": float(p_del_ig_vs_rand),
                "p_val_wilcoxon_ig_vs_attn": float(p_wilc_del_ig_attn),
                "p_val_wilcoxon_ig_vs_rand": float(p_wilc_del_ig_rand)
            },
            "insertion_auc": {
                "ig_auc": {"mean": float(np.mean(ig_ins_auc)), "ci_95": list(compute_bootstrap_ci(ig_ins_auc)[1:])},
                "attn_auc": {"mean": float(np.mean(attn_ins_auc)), "ci_95": list(compute_bootstrap_ci(attn_ins_auc)[1:])},
                "random_auc": {"mean": float(np.mean(rand_ins_auc_seed_means)), "ci_95": list(compute_bootstrap_ci(rand_ins_auc_seed_means)[1:])},
                "p_val_ttest_ig_vs_attn": float(p_ins_ig_vs_attn),
                "p_val_ttest_ig_vs_rand": float(p_ins_ig_vs_rand),
                "p_val_wilcoxon_ig_vs_attn": float(p_wilc_ins_ig_attn),
                "p_val_wilcoxon_ig_vs_rand": float(p_wilc_ins_ig_rand)
            },
            "sample_outlier_summary": {
                "frac_samples_ig_deletion_outlier_zscore": frac_del_z_outlier,
                "frac_samples_ig_deletion_outlier_percentile": frac_del_p_outlier,
                "frac_samples_ig_insertion_outlier_zscore": frac_ins_z_outlier,
                "frac_samples_ig_insertion_outlier_percentile": frac_ins_p_outlier
            }
        }
        
        return agg
