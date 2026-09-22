"""Latent Lens Trajectory Analysis & Property Emergence Engine (Task 6)

Projects intermediate Predictor layer activations into ground-truth target-encoder space
to observe coarse-to-fine identity emergence layer by layer, and evaluates out-of-sample
5-fold cross-validated linear probing with shuffled-label null controls across physical properties.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

from world_model_lens import HookedWorldModel
from world_model_lens.analysis.significance import compute_bootstrap_ci, apply_multiple_comparisons_correction


class LatentLensAnalyzer:
    """Projects intermediate Predictor layer activations to ground-truth target space to observe identity emergence."""

    def __init__(self, hooked_model: HookedWorldModel):
        self.wm = hooked_model

    @torch.no_grad()
    def analyze_sample_trajectory(
        self,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_ids: List[int]
    ) -> Dict[str, Any]:
        """Extracts layer-by-layer target reconstruction trajectory against ground-truth target embeddings.
        
        Args:
            img_tensor: [1, 3, H, W] input image.
            context_ids: List of visible context patch indices.
            target_ids: List of target patch indices to predict.
            
        Returns:
            Dict containing layer-wise Ground-Truth Cosine Similarity, Target MSE, and Diagnostic Norm Ratios.
        """
        device = next(self.wm.adapter.parameters()).device
        img_tensor = img_tensor.to(device)

        # 1. Extract Ground Truth target representations from unmasked target encoder
        target_gt = self.wm.adapter.target_encoder(img_tensor)[:, target_ids, :]  # [1, N_tgt, D]
        if target_gt.dim() == 2:
            target_gt = target_gt.unsqueeze(0)

        # 2. Set context & target IDs on adapter
        self.wm.adapter.last_context_ids = context_ids
        self.wm.adapter.last_target_ids = target_ids

        predictor_blocks = self.wm.adapter.predictor.blocks
        n_pred_layers = len(predictor_blocks)

        captured_activations = {}

        def make_layer_hook(layer_idx: int):
            def hook_fn(act, hook):
                captured_activations[layer_idx] = act.clone()
                return act
            return hook_fn

        # Add post-residual hooks to each Predictor block
        hook_handles = []
        for l_idx in range(n_pred_layers):
            hook_name = f"predictor.blocks.{l_idx}.hook_resid_post"
            try:
                h_handle = self.wm.adapter.add_hook(hook_name, make_layer_hook(l_idx))
                hook_handles.append((hook_name, h_handle))
            except Exception:
                pass

        try:
            patch_emb = self.wm.adapter.context_encoder.patch_embed(img_tensor)
            ctx_ids_t = torch.tensor(context_ids, device=device)
            ctx_emb = patch_emb[:, ctx_ids_t, :]
            pos = self.wm.adapter.context_encoder.pos_embed[:, ctx_ids_t, :]
            ctx_latents = self.wm.adapter.context_encoder.forward_blocks(ctx_emb + pos)
            self.wm.adapter.predictor.hooks = self.wm.adapter.hooks
            final_pred = self.wm.adapter.predictor(ctx_latents, context_ids, target_ids)
        finally:
            for h_name, _ in hook_handles:
                self.wm.adapter.remove_hook(h_name)

        # Project residual activations through predictor norm + projection head
        norm = self.wm.adapter.predictor.norm
        proj = self.wm.adapter.predictor.predictor_project_back

        trajectory_metrics = []
        layer_activations_dict = {}

        for l_idx in range(n_pred_layers):
            if l_idx in captured_activations:
                layer_act = captured_activations[l_idx]
                n_ctx = len(context_ids)
                target_tokens_act = layer_act[:, n_ctx:, :]
                
                layer_activations_dict[l_idx] = target_tokens_act.squeeze(0).cpu().numpy()

                # Project to target space
                proj_latents = proj(norm(target_tokens_act))  # [1, N_tgt, D]

                # Compute Ground-Truth comparison metrics
                layer_mse = F.mse_loss(proj_latents, target_gt).item()
                cos_sim = F.cosine_similarity(proj_latents, target_gt, dim=-1).mean().item()

                proj_norm = torch.norm(proj_latents, dim=-1).mean().item()
                gt_norm = torch.norm(target_gt, dim=-1).mean().item()
                norm_ratio = float(proj_norm / (gt_norm + 1e-8))

                trajectory_metrics.append({
                    "layer": l_idx,
                    "layer_name": f"Predictor Block {l_idx}",
                    "mse": float(layer_mse),
                    "cosine_similarity": float(cos_sim),
                    "proj_norm": float(proj_norm),
                    "gt_norm": float(gt_norm),
                    "norm_ratio_diagnostic": norm_ratio
                })

        # Add final predictor output metrics
        final_mse = F.mse_loss(final_pred, target_gt).item()
        final_cos_sim = F.cosine_similarity(final_pred, target_gt, dim=-1).mean().item()
        final_norm = torch.norm(final_pred, dim=-1).mean().item()
        gt_norm_val = torch.norm(target_gt, dim=-1).mean().item()

        trajectory_metrics.append({
            "layer": n_pred_layers,
            "layer_name": "Predictor Output (Final)",
            "mse": float(final_mse),
            "cosine_similarity": float(final_cos_sim),
            "proj_norm": float(final_norm),
            "gt_norm": float(gt_norm_val),
            "norm_ratio_diagnostic": float(final_norm / (gt_norm_val + 1e-8))
        })

        return {
            "n_context": len(context_ids),
            "n_targets": len(target_ids),
            "trajectory": trajectory_metrics,
            "layer_activations": layer_activations_dict
        }

    def evaluate_5fold_probe_emergence(
        self,
        X_layers: Dict[int, np.ndarray],  # layer_idx -> [N_samples, D] feature matrix
        y_properties: Dict[str, np.ndarray], # property_name -> [N_samples] continuous values
        n_splits: int = 5,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Evaluates 5-fold cross-validated linear probing with shuffled-label empirical null controls per layer block.
        
        Properties categorized into 3 functional groups:
        1. Spatial Coordinates: Grid Y, Grid X
        2. Photometric / Information Density: Color Saliency, Entropy
        3. Structural Texture: Laplacian Variance, Edge Orientation Angle
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        layer_property_results = {}
        raw_p_values = []
        test_keys = []

        for prop_name, y_true in y_properties.items():
            layer_property_results[prop_name] = {}
            
            # Prepare shuffled label null baseline
            rng = np.random.RandomState(seed)
            y_shuffled = rng.permutation(y_true)

            for l_idx, X_feat in X_layers.items():
                test_r2_folds = []
                null_r2_folds = []

                for train_idx, test_idx in kf.split(X_feat):
                    X_tr, X_te = X_feat[train_idx], X_feat[test_idx]
                    y_tr, y_te = y_true[train_idx], y_true[test_idx]
                    y_shuf_tr, y_shuf_te = y_shuffled[train_idx], y_shuffled[test_idx]

                    # True label probe
                    probe = Ridge(alpha=1.0)
                    probe.fit(X_tr, y_tr)
                    y_pred = probe.predict(X_te)
                    
                    # Compute out-of-sample R2
                    ss_res = np.sum((y_te - y_pred) ** 2)
                    ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
                    r2_test = 1.0 - (ss_res / (ss_tot + 1e-8))
                    test_r2_folds.append(r2_test)

                    # Null label probe
                    null_probe = Ridge(alpha=1.0)
                    null_probe.fit(X_tr, y_shuf_tr)
                    y_null_pred = null_probe.predict(X_te)
                    ss_res_null = np.sum((y_shuf_te - y_null_pred) ** 2)
                    ss_tot_null = np.sum((y_shuf_te - np.mean(y_shuf_te)) ** 2)
                    r2_null = 1.0 - (ss_res_null / (ss_tot_null + 1e-8))
                    null_r2_folds.append(r2_null)

                mean_r2 = float(np.mean(test_r2_folds))
                mean_r2_null = float(np.mean(null_r2_folds))
                
                # Paired test against null control
                t_stat, p_val = stats.ttest_rel(test_r2_folds, null_r2_folds)
                p_val_clean = float(p_val) if not np.isnan(p_val) else 1.0

                raw_p_values.append(p_val_clean)
                test_keys.append((prop_name, l_idx))

                layer_property_results[prop_name][l_idx] = {
                    "test_r2_5fold_mean": mean_r2,
                    "null_r2_5fold_mean": mean_r2_null,
                    "r2_above_null": float(mean_r2 - mean_r2_null),
                    "p_val_raw": p_val_clean
                }

        # Apply isolated Task 6 FDR correction across all 24 tests (6 properties x 4 layers)
        fdr_results = apply_multiple_comparisons_correction(raw_p_values, alpha=0.05)
        
        for idx, (prop_name, l_idx) in enumerate(test_keys):
            layer_property_results[prop_name][l_idx]["p_fdr_task6"] = fdr_results["p_fdr"][idx]
            layer_property_results[prop_name][l_idx]["p_bonferroni"] = fdr_results["p_bonferroni"][idx]
            layer_property_results[prop_name][l_idx]["significant_emergence"] = (
                fdr_results["p_fdr"][idx] < 0.05 and layer_property_results[prop_name][l_idx]["r2_above_null"] > 0
            )

        return layer_property_results

    def aggregate_dataset_trajectories(self, sample_trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates layer-wise trajectories across dataset samples with 95% bootstrap CIs and trend tests."""
        if not sample_trajectories:
            return {}

        n_samples = len(sample_trajectories)
        n_layers = len(sample_trajectories[0]["trajectory"])

        layer_cos_sims = {l: [] for l in range(n_layers)}
        layer_mses = {l: [] for l in range(n_layers)}
        layer_norm_ratios = {l: [] for l in range(n_layers)}

        for sample in sample_trajectories:
            for step in sample["trajectory"]:
                l = step["layer"]
                layer_cos_sims[l].append(step["cosine_similarity"])
                layer_mses[l].append(step["mse"])
                layer_norm_ratios[l].append(step["norm_ratio_diagnostic"])

        layer_aggregated = []
        for l in range(n_layers):
            cos_arr = np.array(layer_cos_sims[l])
            mse_arr = np.array(layer_mses[l])
            norm_arr = np.array(layer_norm_ratios[l])

            cos_mean, cos_low, cos_high = compute_bootstrap_ci(cos_arr)
            mse_mean, mse_low, mse_high = compute_bootstrap_ci(mse_arr)

            layer_aggregated.append({
                "layer": l,
                "layer_name": sample_trajectories[0]["trajectory"][l]["layer_name"],
                "cosine_similarity": {"mean": cos_mean, "ci_95": [cos_low, cos_high]},
                "mse": {"mean": mse_mean, "ci_95": [mse_low, mse_high]},
                "norm_ratio_diagnostic_mean": float(np.mean(norm_arr))
            })

        # Pre-registered Trajectory Upward Trend Criteria (L_pred=4 layers)
        pred_cos_means = [step["cosine_similarity"]["mean"] for step in layer_aggregated[:4]] # Predictor blocks 0 to 3
        layer_indices = list(range(len(pred_cos_means)))

        # 1. Spearman Trajectory Rank Correlation
        spearman_rho, spearman_p = stats.spearmanr(layer_indices, pred_cos_means)
        
        # 2. Net Recovery Paired t-test (Layer 3 vs Layer 0)
        l0_cos_arr = np.array(layer_cos_sims[0])
        l3_cos_arr = np.array(layer_cos_sims[3]) if 3 in layer_cos_sims else l0_cos_arr
        
        t_stat, p_net_recovery = stats.ttest_rel(l3_cos_arr, l0_cos_arr, alternative="greater")
        p_net_clean = float(p_net_recovery) if not np.isnan(p_net_recovery) else 1.0

        spearman_pass = bool(spearman_rho >= 0.80)
        net_recovery_pass = bool(p_net_clean < 0.001 and np.mean(l3_cos_arr) > np.mean(l0_cos_arr))

        success_criterion_passed = spearman_pass and net_recovery_pass

        return {
            "n_samples": n_samples,
            "layers": layer_aggregated,
            "pre_registered_eval": {
                "spearman_trend_rho": float(spearman_rho),
                "spearman_trend_p": float(spearman_p),
                "spearman_trend_pass": spearman_pass,
                "net_recovery_p_val": p_net_clean,
                "net_recovery_pass": net_recovery_pass,
                "overall_success_criterion_passed": success_criterion_passed,
                "status": "SUCCESS: Monotonic coarse-to-fine identity emergence confirmed" if success_criterion_passed else "AMBIGUOUS: Trajectory non-monotonic, triggering Per-Head QK/OV Path Patching fallback"
            }
        }
