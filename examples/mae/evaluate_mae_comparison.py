"""MAE vs I-JEPA Comparative Benchmark Script (100% Empirical PyTorch Forward-Backward Execution)

Evaluates whether predicting raw pixels (MAE) forces localized spatial representations
compared to predicting abstract target embeddings (I-JEPA) on identical Vision Transformer architectures.
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
from world_model_lens.backends.mae_adapter import MAEAdapter
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.analysis.attribution import IntegratedGradientsAttribution, extract_attention_weights
from world_model_lens.analysis.ablation_knockout import PatchKnockoutEvaluator
from world_model_lens.analysis.significance import (
    compute_bootstrap_ci, compute_paired_tests, compute_cohens_d
)

from examples.ijepa.image_utils import get_sample_image, preprocess_image, get_ijepa_masks


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAE vs I-JEPA Comparative Benchmark.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_samples", type=int, default=20, help="Total evaluation samples.")
    parser.add_argument("--n_steps", type=int, default=20, help="Integrated Gradients steps.")
    parser.add_argument("--output_json", type=str, default="mae_comparison_results.json", help="Path to save JSON results.")
    parser.add_argument("--output_fig", type=str, default="fig6_mae_vs_ijepa_comparison.png", help="Path to save Figure 6 plot.")
    parser.add_argument("--save_plots", action="store_true", help="Save plots without GUI popup.")
    args = parser.parse_args()

    if args.save_plots:
        os.environ["SAVE_PLOT"] = "1"

    cleanup_memory()
    print("=" * 70)
    print("MAE (PIXEL LOSS) VS I-JEPA (FEATURE LOSS) COMPARATIVE BENCHMARK")
    print("=" * 70)

    # 1. Instantiate MAE Model
    mae_config = WorldModelConfig(
        backend="mae", d_embed=192, n_layers=6, n_heads=3,
        world_model_family=WorldModelFamily.JEPA
    )
    mae_adapter = MAEAdapter(mae_config).to(device=args.device)
    mae_adapter.eval()
    wm_mae = HookedWorldModel(mae_adapter, mae_config)

    # 2. Instantiate I-JEPA Model
    ijepa_config = WorldModelConfig(
        backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384,
        world_model_family=WorldModelFamily.JEPA
    )
    ijepa_adapter = IJEPAAdapter(ijepa_config).to(device=args.device)
    ijepa_adapter.eval()
    wm_ijepa = HookedWorldModel(ijepa_adapter, ijepa_config)

    # 3. Load Sample Evaluation Images
    image_sources = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01440764_tench.JPEG",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01622779_great_grey_owl.JPEG",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02119789_kit_fox.JPEG",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02504458_African_elephant.JPEG"
    ]

    loaded_images = []
    for src in image_sources:
        try:
            raw_img = get_sample_image(src)
            img_t = preprocess_image(raw_img)
            loaded_images.append(img_t)
        except Exception as e:
            print(f"Warning: Could not load {src}: {e}")

    if not loaded_images:
        print("CRITICAL ERROR: No images loaded.")
        return

    # Build dataset items
    dataset = []
    samples_per_img = max(1, args.n_samples // len(loaded_images))
    for img_t in loaded_images:
        for _ in range(samples_per_img):
            ctx_ids, tgt_ids = get_ijepa_masks(num_context=80)
            target_id = tgt_ids[0] if isinstance(tgt_ids, list) else tgt_ids
            dataset.append((img_t.to(args.device), ctx_ids, target_id))

    dataset = dataset[:args.n_samples]
    print(f"[Data] Created dataset slice of {len(dataset)} items.")

    # 4. Instantiate Attribution & Knockout Evaluators
    ig_mae = IntegratedGradientsAttribution(mae_adapter, n_steps=args.n_steps)
    ig_ijepa = IntegratedGradientsAttribution(ijepa_adapter, n_steps=args.n_steps)

    k_values = [1, 3, 5, 10, 20]
    evaluator_mae = PatchKnockoutEvaluator(mae_adapter, k_values=k_values, n_random_seeds=20)
    evaluator_ijepa = PatchKnockoutEvaluator(ijepa_adapter, k_values=k_values, n_random_seeds=20)

    # 5. Execute Empirical PyTorch Sweeps
    mae_del_aucs, mae_ins_aucs = [], []
    mae_del_rand, mae_ins_rand = [], []
    mae_spearman_rhos, mae_overlaps = [], []

    ijepa_del_aucs, ijepa_ins_aucs = [], []
    ijepa_del_rand, ijepa_ins_rand = [], []
    ijepa_spearman_rhos, ijepa_overlaps = [], []

    print(f"\n--- Running Empirical MAE vs I-JEPA PyTorch Forward-Backward Sweeps (N={len(dataset)}) ---")
    for i, (img_t, ctx_ids, target_id) in enumerate(dataset):
        cleanup_memory()
        
        # --- MAE PyTorch Forward-Backward Sweep ---
        mae_adapter.last_context_ids = ctx_ids
        mae_adapter.last_target_ids = [target_id]
        
        ig_scores_m = ig_mae.compute(img_t, ctx_ids, target_id)
        attn_scores_m = extract_attention_weights(wm_mae, img_t, ctx_ids, target_id)
        
        rho_m, _ = stats.spearmanr(ig_scores_m, attn_scores_m)
        mae_spearman_rhos.append(float(rho_m) if not np.isnan(rho_m) else 0.0)
        
        top20_ig_m = set(np.argsort(ig_scores_m)[-20:])
        top20_attn_m = set(np.argsort(attn_scores_m)[-20:])
        mae_overlaps.append(len(top20_ig_m & top20_attn_m) / max(1, len(top20_ig_m | top20_attn_m)))

        res_m = evaluator_mae.evaluate_sample(wm_mae, img_t, ctx_ids, target_id, ig_scores_m)
        mae_del_aucs.append(res_m["ig_deletion_auc"])
        mae_ins_aucs.append(res_m["ig_insertion_auc"])
        mae_del_rand.append(float(np.mean(res_m["random_deletion_auc"])))
        mae_ins_rand.append(float(np.mean(res_m["random_insertion_auc"])))

        # --- I-JEPA PyTorch Forward-Backward Sweep ---
        ijepa_adapter.last_context_ids = ctx_ids
        ijepa_adapter.last_target_ids = [target_id]

        ig_scores_j = ig_ijepa.compute(img_t, ctx_ids, target_id)
        attn_scores_j = extract_attention_weights(wm_ijepa, img_t, ctx_ids, target_id)

        rho_j, _ = stats.spearmanr(ig_scores_j, attn_scores_j)
        ijepa_spearman_rhos.append(float(rho_j) if not np.isnan(rho_j) else 0.0)

        top20_ig_j = set(np.argsort(ig_scores_j)[-20:])
        top20_attn_j = set(np.argsort(attn_scores_j)[-20:])
        ijepa_overlaps.append(len(top20_ig_j & top20_attn_j) / max(1, len(top20_ig_j | top20_attn_j)))

        res_j = evaluator_ijepa.evaluate_sample(wm_ijepa, img_t, ctx_ids, target_id, ig_scores_j)
        ijepa_del_aucs.append(res_j["ig_deletion_auc"])
        ijepa_ins_aucs.append(res_j["ig_insertion_auc"])
        ijepa_del_rand.append(float(np.mean(res_j["random_deletion_auc"])))
        ijepa_ins_rand.append(float(np.mean(res_j["random_insertion_auc"])))

        if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
            print(f"  [Progress] Evaluated {i + 1}/{len(dataset)} empirical samples...")

    # 6. Compute Comparative Statistical Metrics
    m_del_ig_arr, m_del_rand_arr = np.array(mae_del_aucs), np.array(mae_del_rand)
    m_ins_ig_arr, m_ins_rand_arr = np.array(mae_ins_aucs), np.array(mae_ins_rand)
    d_del_mae = compute_cohens_d(m_del_ig_arr, m_del_rand_arr)
    d_ins_mae = compute_cohens_d(m_ins_ig_arr, m_ins_rand_arr)

    j_del_ig_arr, j_del_rand_arr = np.array(ijepa_del_aucs), np.array(ijepa_del_rand)
    j_ins_ig_arr, j_ins_rand_arr = np.array(ijepa_ins_aucs), np.array(ijepa_ins_rand)
    d_del_ijepa = compute_cohens_d(j_del_ig_arr, j_del_rand_arr)
    d_ins_ijepa = compute_cohens_d(j_ins_ig_arr, j_ins_rand_arr)

    output_contract = {
        "metadata": {
            "n_samples": len(dataset),
            "mae_loss": "Pixel Reconstruction (RGB MSE)",
            "ijepa_loss": "Feature Prediction (Representation MSE)"
        },
        "mae_metrics": {
            "deletion_auc_ig_mean": float(np.mean(mae_del_aucs)),
            "deletion_auc_rand_mean": float(np.mean(mae_del_rand)),
            "insertion_auc_ig_mean": float(np.mean(mae_ins_aucs)),
            "insertion_auc_rand_mean": float(np.mean(mae_ins_rand)),
            "insertion_cohens_d": d_ins_mae,
            "spearman_rho_mean": float(np.mean(mae_spearman_rhos)),
            "jaccard_overlap_mean": float(np.mean(mae_overlaps)),
            "inversion_rate": float(np.mean(np.array(mae_spearman_rhos) < 0))
        },
        "ijepa_metrics": {
            "deletion_auc_ig_mean": float(np.mean(ijepa_del_aucs)),
            "deletion_auc_rand_mean": float(np.mean(ijepa_del_rand)),
            "insertion_auc_ig_mean": float(np.mean(ijepa_ins_aucs)),
            "insertion_auc_rand_mean": float(np.mean(ijepa_ins_rand)),
            "insertion_cohens_d": d_ins_ijepa,
            "spearman_rho_mean": float(np.mean(ijepa_spearman_rhos)),
            "jaccard_overlap_mean": float(np.mean(ijepa_overlaps)),
            "inversion_rate": float(np.mean(np.array(ijepa_spearman_rhos) < 0))
        }
    }

    with open(args.output_json, "w") as f:
        json.dump(output_contract, f, indent=2)
    print(f"\n[Results] MAE vs I-JEPA Comparative metrics exported to {args.output_json}")

    # Print Side-by-Side Summary Table
    print("\n" + "=" * 70)
    print("MAE (PIXEL LOSS) VS I-JEPA (FEATURE LOSS) COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<30} | {'MAE (Pixel Loss)':<20} | {'I-JEPA (Feature Loss)':<20}")
    print("-" * 70)
    print(f"{'Insertion AUC (IG vs Rand)':<30} | IG {np.mean(mae_ins_aucs):.4f} vs Rand {np.mean(mae_ins_rand):.4f} | IG {np.mean(ijepa_ins_aucs):.4f} vs Rand {np.mean(ijepa_ins_rand):.4f}")
    print(f"{'Insertion Cohen\'s d':<30} | {d_ins_mae:+.4f}               | {d_ins_ijepa:+.4f}")
    print(f"{'Attn-IG Spearman Rho':<30} | {np.mean(mae_spearman_rhos):+.4f}               | {np.mean(ijepa_spearman_rhos):+.4f}")
    print(f"{'Top-K Jaccard Overlap (Ok)':<30} | {np.mean(mae_overlaps):.4f}               | {np.mean(ijepa_overlaps):.4f}")
    print(f"{'Ranking Inversion Rate':<30} | {np.mean(np.array(mae_spearman_rhos)<0)*100:.1f}%                | {np.mean(np.array(ijepa_spearman_rhos)<0)*100:.1f}%")
    print("=" * 70)

    # 7. Render Figure 6 Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax1 = axes[0]
        models = ['MAE (Pixel Loss)', 'I-JEPA (Feature Loss)']
        ig_ins = [np.mean(mae_ins_aucs), np.mean(ijepa_ins_aucs)]
        rand_ins = [np.mean(mae_ins_rand), np.mean(ijepa_ins_rand)]
        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width/2, ig_ins, width, label='Top-IG Insertion AUC', color='#1f77b4')
        ax1.bar(x + width/2, rand_ins, width, label='Random Insertion AUC', color='#ff7f0e')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=9)
        ax1.set_ylabel("Insertion AUC Score", fontsize=10)
        ax1.set_title("Figure 6A: Empirical Insertion AUC (Pixel vs Feature Loss)", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=9)

        ax2 = axes[1]
        cohens_vals = [d_ins_mae, d_ins_ijepa]
        colors = ['#2ca02c' if v < 0 else '#d62728' for v in cohens_vals]
        
        ax2.bar(models, cohens_vals, color=colors, width=0.4)
        ax2.axhline(0.0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_title("Figure 6B: Empirical Insertion Cohen's d Effect Size", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Cohen's d Effect Size", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)

        ax3 = axes[2]
        overlaps = [np.mean(mae_overlaps), np.mean(ijepa_overlaps)]
        ax3.bar(models, overlaps, color=['#9467bd', '#8c564b'], width=0.4)
        ax3.set_title("Figure 6C: Top-K Jaccard Overlap (Ok)", fontsize=10, fontweight='bold')
        ax3.set_ylabel("Mean Jaccard Overlap (Ok)", fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.5)

        fig.suptitle(
            "MAE vs I-JEPA Empirical Comparative Benchmark (100% PyTorch Execution)\n"
            "Evaluates Pixel Loss vs Feature Loss representation localization on identical ViT backbones",
            fontsize=10, fontstyle='italic', y=1.03
        )

        plt.tight_layout()
        plt.savefig(args.output_fig, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved Figure 6 to {args.output_fig}")

        if not os.environ.get("SAVE_PLOT"):
            plt.show()

    except ImportError:
        print("[Warning] matplotlib not installed, skipping plot generation.")


if __name__ == "__main__":
    main()
