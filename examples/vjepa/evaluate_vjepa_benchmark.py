"""V-JEPA Cross-Modal Verification Benchmark (Tasks 1, 2 & 6 for Video World Models)

Evaluates whether the Distributed Context Representation Phenomenon, Cross-Attention Unfaithfulness,
and 5-Fold Physical Property Probing hold across 3D spatiotemporal video clips (T x H x W tubelets).
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.stats as stats
import torch
import gc

# Ensure local library takes precedence
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.vjepa_adapter import VJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.analysis.significance import (
    compute_bootstrap_ci, compute_paired_tests, compute_cohens_d, apply_multiple_comparisons_correction
)
from world_model_lens.analysis.attribution import IntegratedGradientsAttribution, extract_attention_weights
from world_model_lens.analysis.ablation_knockout import PatchKnockoutEvaluator
from world_model_lens.analysis.latent_lens import LatentLensAnalyzer


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def generate_synthetic_video_clip(num_frames=16, height=224, width=224, seed=42) -> torch.Tensor:
    """Generates a synthetic 5D video clip [1, 3, T, H, W] with randomized temporal motion and spatial geometry."""
    rng = np.random.RandomState(seed)
    T, H, W = num_frames, height, width
    
    # Base randomized spatial background grid
    freq_y = rng.uniform(0.5, 3.0)
    freq_x = rng.uniform(0.5, 3.0)
    grid_y, grid_x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    bg_base = np.sin(freq_y * np.pi * grid_y) * np.cos(freq_x * np.pi * grid_x)
    
    # Randomized motion trajectory
    start_y, start_x = rng.uniform(-0.7, 0.7, size=2)
    end_y, end_x = rng.uniform(-0.7, 0.7, size=2)
    obj_radius = rng.uniform(0.08, 0.25)
    
    frames = []
    for t in range(T):
        alpha = t / max(T - 1, 1)
        center_y = (1 - alpha) * start_y + alpha * end_y
        center_x = (1 - alpha) * start_x + alpha * end_x
        dist = np.sqrt((grid_y - center_y)**2 + (grid_x - center_x)**2)
        object_mask = np.exp(-dist**2 / (obj_radius**2))
        
        frame = np.zeros((3, H, W), dtype=np.float32)
        frame[0] = 0.5 * (bg_base + 1.0)
        frame[1] = 0.5 * (grid_x + 1.0)
        frame[2] = object_mask
        
        # Add high-frequency motion noise
        frame += 0.05 * rng.randn(3, H, W).astype(np.float32)
        frames.append(frame)
        
    video_tensor = torch.from_numpy(np.stack(frames, axis=1)).unsqueeze(0) # [1, 3, T, H, W]
    return video_tensor


def main():
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA Cross-Modal Benchmark.")
    parser.add_argument("--weights", type=str, default="vjepa_mini.pth", help="Path to weights file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_samples", type=int, default=100, help="Total video clips to evaluate.")
    parser.add_argument("--output_json", type=str, default="vjepa_benchmark_results.json", help="Path to save JSON results.")
    parser.add_argument("--output_fig", type=str, default="fig5_vjepa_benchmark.png", help="Path to save Figure 5 plot.")
    parser.add_argument("--save_plots", action="store_true", help="Save plots without GUI popup.")
    args = parser.parse_args()

    if args.save_plots:
        os.environ["SAVE_PLOT"] = "1"

    cleanup_memory()
    print("=" * 70)
    print("V-JEPA CROSS-MODAL VERIFICATION BENCHMARK (100% EMPIRICAL PYTORCH)")
    print("=" * 70)

    # 1. Instantiate V-JEPA Adapter & World Model
    config = WorldModelConfig(
        backend="vjepa",
        d_embed=192,
        n_layers=6,
        n_heads=3,
        predictor_embed_dim=384,
        predictor_depth=4,
        num_frames=16,
        tubelet_size=2,
        world_model_family=WorldModelFamily.JEPA
    )

    if os.path.exists(args.weights):
        print(f"[Model] Loading V-JEPA weights via VJEPAAdapter.from_checkpoint from {args.weights}")
        adapter = VJEPAAdapter.from_checkpoint(args.weights, config)
    else:
        print(f"[Warning] Weights file not found at {args.weights}. Initializing VJEPAAdapter from config.")
        adapter = VJEPAAdapter(config)

    adapter.to(device=args.device)
    adapter.eval()
    wm = HookedWorldModel(adapter, config)

    # Total tubelets: T_grid = 16//2 = 8, H_grid = 14, W_grid = 14 -> 1568 tubelets
    n_total_tubelets = 8 * 14 * 14
    n_ctx_tubelets = 800
    n_tgt_tubelets = 100

    ig_evaluator = IntegratedGradientsAttribution(adapter, n_steps=20)
    knockout_evaluator = PatchKnockoutEvaluator(adapter, k_values=[1, 3, 5, 10, 20], n_random_seeds=20)
    analyzer = LatentLensAnalyzer(wm)

    print(f"[V-JEPA] Model initialized on {args.device} (1,568 spatiotemporal tubelets per video clip).")
    print(f"[Evaluation] Running 100% PyTorch empirical sweeps across {args.n_samples} video clips...")

    # 2. Run Spatiotemporal Benchmark Sweeps
    ig_del_aucs, attn_del_aucs, rand_del_aucs = [], [], []
    ig_ins_aucs, attn_ins_aucs, rand_ins_aucs = [], [], []
    spearman_rhos = []
    jaccard_overlaps = []

    layer_features_list = {l: [] for l in range(4)}
    y_properties = {
        "spatial_grid_y": [],
        "spatial_grid_x": [],
        "temporal_frame_t": []
    }

    for i in range(args.n_samples):
        cleanup_memory()
        video_t = generate_synthetic_video_clip(num_frames=16, seed=42 + i).to(args.device)
        
        np.random.seed(42 + i)
        perm = np.random.permutation(n_total_tubelets)
        ctx_ids = list(perm[:n_ctx_tubelets])
        tgt_ids = list(perm[n_ctx_tubelets:n_ctx_tubelets + n_tgt_tubelets])
        target_id = tgt_ids[0]

        adapter.last_context_ids = ctx_ids
        adapter.last_target_ids = tgt_ids

        # 100% Empirical Integrated Gradients & Attention scores
        ig_scores = ig_evaluator.compute(video_t, ctx_ids, target_id, batch_size=1)
        attn_scores = extract_attention_weights(wm, video_t, ctx_ids, target_id, layer_idx=-1)

        # Calculate Spearman correlation & Jaccard overlap
        rho, _ = stats.spearmanr(ig_scores, attn_scores)
        spearman_rhos.append(float(rho))

        top20_ig = set(np.argsort(ig_scores)[-20:])
        top20_attn = set(np.argsort(attn_scores)[-20:])
        jaccard = len(top20_ig & top20_attn) / len(top20_ig | top20_attn)
        jaccard_overlaps.append(float(jaccard))

        # Real PyTorch Deletion & Insertion AUCs via PatchKnockoutEvaluator
        sample_res = knockout_evaluator.evaluate_sample(wm, video_t, ctx_ids, target_id, ig_scores, seed=42 + i)
        ig_del_aucs.append(float(sample_res["ig_deletion_auc"]))
        attn_del_aucs.append(float(sample_res["attn_deletion_auc"]))
        rand_del_aucs.append(float(np.mean(sample_res["random_deletion_auc"])))

        ig_ins_aucs.append(float(sample_res["ig_insertion_auc"]))
        attn_ins_aucs.append(float(sample_res["attn_insertion_auc"]))
        rand_ins_aucs.append(float(np.mean(sample_res["random_insertion_auc"])))

        # Extract actual intermediate Predictor layer representations for 5-fold cross-validated probing
        traj_res = analyzer.analyze_sample_trajectory(video_t, ctx_ids, tgt_ids)
        for l_idx in range(4):
            feat_vec = traj_res["layer_activations"][l_idx].mean(axis=0)
            layer_features_list[l_idx].append(feat_vec)

        # Ground truth spatial & temporal coordinates for target tubelets
        tgt_sample = tgt_ids[0]
        tube_t = tgt_sample // (14 * 14)
        rem = tgt_sample % (14 * 14)
        tube_y = rem // 14
        tube_x = rem % 14

        y_properties["spatial_grid_y"].append(float(tube_y))
        y_properties["spatial_grid_x"].append(float(tube_x))
        y_properties["temporal_frame_t"].append(float(tube_t))

        cleanup_memory()
        if (i + 1) % 10 == 0 or (i + 1) == args.n_samples:
            print(f"  [Progress] Processed {i + 1}/{args.n_samples} video clips...")

    # 3. Compute Statistical Package for V-JEPA
    ig_del_arr, attn_del_arr, rand_del_arr = np.array(ig_del_aucs), np.array(attn_del_aucs), np.array(rand_del_aucs)
    ig_ins_arr, attn_ins_arr, rand_ins_arr = np.array(ig_ins_aucs), np.array(attn_ins_aucs), np.array(rand_ins_aucs)

    del_ig_m, del_ig_low, del_ig_high = compute_bootstrap_ci(ig_del_arr)
    del_attn_m, del_attn_low, del_attn_high = compute_bootstrap_ci(attn_del_arr)
    del_rand_m, del_rand_low, del_rand_high = compute_bootstrap_ci(rand_del_arr)

    ins_ig_m, ins_ig_low, ins_ig_high = compute_bootstrap_ci(ig_ins_arr)
    ins_attn_m, ins_attn_low, ins_attn_high = compute_bootstrap_ci(attn_ins_arr)
    ins_rand_m, ins_rand_low, ins_rand_high = compute_bootstrap_ci(rand_ins_arr)

    del_tests = compute_paired_tests(ig_del_arr, rand_del_arr)
    ins_tests = compute_paired_tests(ig_ins_arr, rand_ins_arr)
    d_del = compute_cohens_d(ig_del_arr, rand_del_arr)
    d_ins = compute_cohens_d(ig_ins_arr, rand_ins_arr)

    # 4. Run 5-Fold Cross-Validated Probing for Spatial & Temporal Observables
    X_layers_np = {l: np.array(vecs) for l, vecs in layer_features_list.items()}
    y_properties_np = {k: np.array(v) for k, v in y_properties.items()}

    probe_emergence_results = analyzer.evaluate_5fold_probe_emergence(
        X_layers_np, y_properties_np, n_splits=min(5, args.n_samples)
    )

    # Compute Bootstrap CI & Two-Proportion Test vs I-JEPA (N=448, 182/448 = 40.625% Inversion Rate)
    inversion_flags = (np.array(spearman_rhos) < 0).astype(float)
    inv_rate_m, inv_rate_low, inv_rate_high = compute_bootstrap_ci(inversion_flags)

    n_ijepa = 448
    k_ijepa = 182
    p_ijepa = k_ijepa / n_ijepa

    n_vjepa = len(inversion_flags)
    k_vjepa = int(np.sum(inversion_flags))
    p_vjepa = k_vjepa / n_vjepa

    p_pooled = (k_vjepa + k_ijepa) / (n_vjepa + n_ijepa)
    se_pooled = np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n_vjepa + 1.0 / n_ijepa))
    z_stat = (p_vjepa - p_ijepa) / (se_pooled + 1e-12)
    p_val_two_prop = float(2 * (1.0 - stats.norm.cdf(abs(z_stat))))

    output_contract = {
        "metadata": {
            "model": "V-JEPA (Video Joint-Embedding Predictive Architecture)",
            "n_samples": args.n_samples,
            "n_tubelets": n_total_tubelets
        },
        "deletion_auc": {
            "ig": {"mean": del_ig_m, "ci_95": [del_ig_low, del_ig_high]},
            "attn": {"mean": del_attn_m, "ci_95": [del_attn_low, del_attn_high]},
            "rand": {"mean": del_rand_m, "ci_95": [del_rand_low, del_rand_high]},
            "cohens_d": d_del,
            "p_val": del_tests["p_val_ttest"],
            "p_fdr": del_tests.get("p_fdr", del_tests["p_val_ttest"])
        },
        "insertion_auc": {
            "ig": {"mean": ins_ig_m, "ci_95": [ins_ig_low, ins_ig_high]},
            "attn": {"mean": ins_attn_m, "ci_95": [ins_attn_low, ins_attn_high]},
            "rand": {"mean": ins_rand_m, "ci_95": [ins_rand_low, ins_rand_high]},
            "cohens_d": d_ins,
            "p_val": ins_tests["p_val_ttest"],
            "p_fdr": ins_tests.get("p_fdr", ins_tests["p_val_ttest"])
        },
        "attention_gradient_alignment": {
            "spearman_rho_mean": float(np.mean(spearman_rhos)),
            "spearman_rho_ci_95": list(compute_bootstrap_ci(np.array(spearman_rhos))[1:]),
            "jaccard_overlap_mean": float(np.mean(jaccard_overlaps)),
            "inversion_rate": inv_rate_m,
            "inversion_rate_ci_95": [inv_rate_low, inv_rate_high],
            "ijepa_comparison": {
                "ijepa_inversion_rate": p_ijepa,
                "ijepa_n_samples": n_ijepa,
                "z_statistic": float(z_stat),
                "p_val_two_proportion": p_val_two_prop
            }
        },
        "5fold_spatiotemporal_probing": probe_emergence_results
    }

    with open(args.output_json, "w") as f:
        json.dump(output_contract, f, indent=2)
    print(f"\n[Results] V-JEPA Benchmark metrics exported to {args.output_json}")

    # Print Summary Table
    print("\n" + "=" * 70)
    print("V-JEPA BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(f"Deletion AUC  : IG = {del_ig_m:.4f} | Attn = {del_attn_m:.4f} | Rand = {del_rand_m:.4f} (Cohen's d = {d_del:+.4f}, p = {del_tests['p_val_ttest']:.4e})")
    print(f"Insertion AUC : IG = {ins_ig_m:.4f} | Attn = {ins_attn_m:.4f} | Rand = {ins_rand_m:.4f} (Cohen's d = {d_ins:+.4f}, p = {ins_tests['p_val_ttest']:.4e})")
    print(f"Attn-IG Rho   : Mean rho = {np.mean(spearman_rhos):+.4f} | Inversion Rate = {np.mean(np.array(spearman_rhos) < 0)*100:.1f}%")

    print("\n5-Fold Spatiotemporal Probing (R^2 > R^2_null, p_FDR < 0.05):")
    for prop_name, l_map in probe_emergence_results.items():
        print(f"  Property: {prop_name}")
        for l_idx, l_data in l_map.items():
            sig_str = "EMERGENT *" if l_data["significant_emergence"] else "null"
            print(f"    Layer {l_idx}: R^2 = {l_data['test_r2_5fold_mean']:.4f} (Null: {l_data['null_r2_5fold_mean']:.4f}, p_FDR = {l_data['p_fdr_task6']:.4e}) -> {sig_str}")
    print("=" * 70)

    # 5. Render Figure 5 Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Figure 5A: Deletion & Insertion AUCs
        ax1 = axes[0]
        methods = ['IG', 'Attention', 'Random (M=20)']
        del_vals = [del_ig_m, del_attn_m, del_rand_m]
        ins_vals = [ins_ig_m, ins_attn_m, ins_rand_m]
        x = np.arange(len(methods))
        width = 0.35

        ax1.bar(x - width/2, del_vals, width, label='Deletion AUC (Higher Better)', color='#1f77b4')
        ax1.bar(x + width/2, ins_vals, width, label='Insertion AUC (Lower Better)', color='#ff7f0e')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=9)
        ax1.set_ylabel("AUC Score (MSE)", fontsize=10)
        ax1.set_title("Figure 5A: V-JEPA Spatiotemporal Tubelet Knockout AUCs", fontsize=10, fontweight='bold')
        ymin = min(del_vals + ins_vals) * 0.95
        ymax = max(del_vals + ins_vals) * 1.05
        ax1.set_ylim(ymin, ymax)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=9)

        # Figure 5B: 5-Fold Spatiotemporal Probing
        ax2 = axes[1]
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        for idx, (prop_name, l_map) in enumerate(probe_emergence_results.items()):
            p_layers = sorted(list(l_map.keys()))
            p_r2s = [l_map[l]["test_r2_5fold_mean"] for l in p_layers]
            ax2.plot(p_layers, p_r2s, 'o--', color=colors[idx % len(colors)], linewidth=2.0, label=prop_name)

        ax2.set_xticks(list(range(4)))
        ax2.set_xticklabels([f"Block {l}" for l in range(4)], fontsize=9)
        ax2.set_title("Figure 5B: 5-Fold Spatiotemporal Property Emergence Map", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Out-of-Sample Test R^2", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=8)

        # Figure 5C: Attn-IG Alignment Distribution
        ax3 = axes[2]
        ax3.hist(spearman_rhos, bins=10, color='#9467bd', edgecolor='black', alpha=0.7)
        ax3.axvline(0.0, color='red', linestyle='--', linewidth=2.0, label='Inversion Boundary (rho < 0)')
        ax3.set_title("Figure 5C: V-JEPA Attention-IG Spearman Rho Distribution", fontsize=10, fontweight='bold')
        ax3.set_xlabel("Spearman Rank Correlation (rho)", fontsize=10)
        ax3.set_ylabel("Video Clip Frequency", fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(fontsize=9)

        fig.suptitle(
            "V-JEPA Cross-Modal Benchmark: Distributed Context & Spatiotemporal Property Emergence\n"
            "Verifies cross-modal generalization across 3D spatiotemporal video clips (T x H x W tubelets)",
            fontsize=10, fontstyle='italic', y=1.03
        )

        plt.tight_layout()
        plt.savefig(args.output_fig, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved Figure 5 to {args.output_fig}")

        if not os.environ.get("SAVE_PLOT"):
            plt.show()

    except ImportError:
        print("[Warning] matplotlib not installed, skipping plot generation.")


if __name__ == "__main__":
    main()
