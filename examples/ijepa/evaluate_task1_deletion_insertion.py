"""Task 1 Evaluation Entry Point: Random Patch Knockout & Deletion/Insertion AUC Curves

Evaluates Integrated Gradients (IG) vs Cross-Attention Weights vs Random Patch Baseline (M=20 seeds)
across patch counts K in {1, 3, 5, 10, 20} with 95% bootstrap confidence intervals and paired statistical tests.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import gc

# Ensure local library takes precedence
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.analysis.attribution import IntegratedGradientsAttribution
from world_model_lens.analysis.ablation_knockout import PatchKnockoutEvaluator, compute_bootstrap_ci

from image_utils import get_sample_image, preprocess_image, get_ijepa_masks


def cleanup_memory():
    """Clear CUDA cache and collect garbage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 1: Deletion and Insertion AUC curves for I-JEPA.")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="ijepa_mini.pth",
        help="Path to weights file, or 'meta' to use official Meta ViT-H weights."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_samples", type=int, default=10, help="Total samples to evaluate.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for evaluation.")
    parser.add_argument("--n_steps", type=int, default=50, help="Integrated Gradients steps.")
    parser.add_argument("--n_random_seeds", type=int, default=20, help="Random seeds for null distribution (M=20 default).")
    parser.add_argument("--patch_ablation_mode", type=str, default="zero", choices=["zero", "dataset_mean_patch"], help="Patch ablation mode.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to local image dataset.")
    parser.add_argument("--output_json", type=str, default="task1_deletion_insertion_results.json", help="Path to output JSON results.")
    parser.add_argument("--output_fig", type=str, default="fig3_deletion_insertion_auc.png", help="Path to save Figure 3 plot.")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk without GUI popup.")
    args = parser.parse_args()

    if args.save_plots:
        os.environ["SAVE_PLOT"] = "1"

    cleanup_memory()
    print("=" * 70)
    print("TASK 1: Random Patch Knockout & Deletion/Insertion AUC Evaluation")
    print("=" * 70)

    # 1. Setup Model Architecture and Weights
    if args.weights == "meta" or "vith" in args.weights.lower():
        print("[Model] Loading Meta ViT-H/14 architecture...")
        config = WorldModelConfig(
            backend="ijepa", patch_size=14, d_embed=1280, n_layers=32, n_heads=16,
            predictor_embed_dim=384, predictor_depth=12, predictor_heads=12,
            world_model_family=WorldModelFamily.JEPA
        )
        weights_path = "vith14_in1k_ep300.pth.tar" if args.weights == "meta" else args.weights
    else:
        print("[Model] Loading mini architecture...")
        config = WorldModelConfig(
            backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384,
            world_model_family=WorldModelFamily.JEPA
        )
        weights_path = os.path.join(os.path.dirname(__file__), args.weights)

    if os.path.exists(weights_path):
        print(f"[Model] Loading weights via IJEPAAdapter.from_checkpoint from {weights_path}")
        adapter = IJEPAAdapter.from_checkpoint(weights_path, config)
    else:
        print(f"[Warning] Weights file not found at {weights_path}. Using random initialization.")
        adapter = IJEPAAdapter(config)

    adapter.to(device=args.device)
    adapter.eval()
    wm = HookedWorldModel(adapter, config)

    # 2. Build Validation Dataset
    print(f"\n[Data] Preparing dataset samples (target N={args.n_samples})...")
    np.random.seed(42)

    image_sources = []
    category_map = {}

    if args.data_dir and os.path.isdir(args.data_dir):
        for root, dirs, files in os.walk(args.data_dir):
            cat = os.path.basename(root)
            if cat == os.path.basename(args.data_dir):
                cat = "unspecified"
            valid_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if valid_files:
                if cat not in category_map:
                    category_map[cat] = []
                category_map[cat].extend(valid_files)

        if category_map:
            n_cats = len(category_map)
            per_cat = max(1, args.n_samples // n_cats)
            for cat, paths in category_map.items():
                image_sources.extend(paths[:per_cat])
            print(f"[Data] Found {n_cats} categories, sampled {len(image_sources)} images.")

    if not image_sources:
        print("[Data] Using default sample image URLs...")
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
            cat_name = "default"
            for c, p_list in category_map.items():
                if src in p_list:
                    cat_name = c
                    break
            loaded_images.append((src, img_t, cat_name))
        except Exception as e:
            print(f"Warning: Could not load {src}: {e}")

    if not loaded_images:
        print("CRITICAL ERROR: No images accessible.")
        return

    dataset = []
    samples_per_image = max(1, args.n_samples // len(loaded_images))

    for src, img_t, category in loaded_images:
        target_ids = list(range(10, 190, max(1, 190 // samples_per_image)))[:samples_per_image]
        for tid in target_ids:
            context_ids, _ = get_ijepa_masks(num_context=80)
            if tid in context_ids:
                context_ids.remove(tid)
            dataset.append((img_t, context_ids, tid, category))

    dataset = dataset[args.start_idx : args.start_idx + args.n_samples]
    print(f"[Data] Created dataset slice of {len(dataset)} evaluation items.")

    # Compute optional dataset mean patch embedding if requested
    dataset_mean_patch = None
    if args.patch_ablation_mode == "dataset_mean_patch":
        print("[Ablation] Computing dataset mean patch embedding...")
        all_patches = []
        with torch.no_grad():
            for item in dataset[:10]:
                img = item[0].to(args.device)
                emb = adapter.context_encoder.patch_embed(img)
                all_patches.append(emb.squeeze(0))
        dataset_mean_patch = torch.stack(all_patches, dim=0).mean(dim=[0, 1])

    # 3. Pre-compute Integrated Gradients Attributions
    ig_method = IntegratedGradientsAttribution(adapter, n_steps=args.n_steps)
    cache_path = "ig_cache.pth"
    cached_attributions = []

    if os.path.exists(cache_path):
        print(f"[Cache] Loading precomputed Integrated Gradients from {cache_path}...")
        try:
            cached_attributions = torch.load(cache_path, weights_only=False)
            if len(cached_attributions) < len(dataset):
                print(f"[Cache] Cached items ({len(cached_attributions)}) < dataset items ({len(dataset)}). Recomputing missing...")
                cached_attributions = []
            else:
                cached_attributions = cached_attributions[:len(dataset)]
        except Exception:
            cached_attributions = []

    if not cached_attributions:
        print(f"\n--- Computing Integrated Gradients ({args.n_steps} steps) ---")
        cached_attributions = []
        for i, item in enumerate(dataset):
            img_t, context_ids, target_id = item[0], item[1], item[2]
            if i % 5 == 0:
                print(f"  [Progress] IG computation {i}/{len(dataset)}...")
                cleanup_memory()
            attr = ig_method.compute(img_t, context_ids, target_id, batch_size=2)
            cached_attributions.append(attr)
        torch.save(cached_attributions, cache_path)
        print(f"[Cache] Saved IG results to {cache_path}")

    # 4. Execute Task 1 Deletion & Insertion Knockout Evaluation
    k_values = [1, 3, 5, 10, 20]
    evaluator = PatchKnockoutEvaluator(
        adapter=adapter,
        k_values=k_values,
        patch_ablation_mode=args.patch_ablation_mode,
        n_random_seeds=args.n_random_seeds,
        dataset_mean_patch=dataset_mean_patch
    )

    print(f"\n--- Running Task 1 Evaluation (K={k_values}, M={args.n_random_seeds} random seeds) ---")
    sample_results = []
    layer_idx = len(adapter.predictor.blocks) - 1 # Evaluate at output layer of predictor

    for i, item in enumerate(dataset):
        img_t, context_ids, target_id, cat = item[0], item[1], item[2], item[3]
        attr_scores = cached_attributions[i]
        
        res = evaluator.evaluate_sample(
            wm, img_t, context_ids, target_id, attr_scores, layer_idx=layer_idx, seed=42 + i
        )
        res["sample_idx"] = i
        res["category"] = cat
        sample_results.append(res)

        if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
            print(f"  [Progress] Evaluated {i + 1}/{len(dataset)} samples...")

    # 5. Aggregate Dataset Statistics
    agg_results = evaluator.aggregate_dataset_results(sample_results)

    # Combine into full structured JSON contract
    output_contract = {
        "metadata": {
            "weights": args.weights,
            "n_samples": len(dataset),
            "n_steps": args.n_steps,
            "n_random_seeds": args.n_random_seeds,
            "patch_ablation_mode": args.patch_ablation_mode,
            "k_values": k_values
        },
        "tensor_shapes": {
            "ig_deletion_mses": ["n_samples", "K"],
            "attn_deletion_mses": ["n_samples", "K"],
            "random_deletion_mses": ["n_samples", "n_random_seeds", "K"],
            "ig_insertion_mses": ["n_samples", "K"],
            "attn_insertion_mses": ["n_samples", "K"],
            "random_insertion_mses": ["n_samples", "n_random_seeds", "K"],
            "ig_deletion_auc": ["n_samples"],
            "attn_deletion_auc": ["n_samples"],
            "random_deletion_auc": ["n_samples", "n_random_seeds"],
            "ig_insertion_auc": ["n_samples"],
            "attn_insertion_auc": ["n_samples"],
            "random_insertion_auc": ["n_samples", "n_random_seeds"]
        },
        "summary": agg_results,
        "samples": sample_results
    }

    with open(args.output_json, "w") as f:
        json.dump(output_contract, f, indent=2)
    print(f"\n[Results] Full quantitative metrics exported to {args.output_json}")

    # Print summary benchmark table to console
    print("\n" + "=" * 70)
    print("TASK 1 BENCHMARK SUMMARY (Normalized AUC)")
    print("=" * 70)
    del_auc = agg_results["deletion_auc"]
    ins_auc = agg_results["insertion_auc"]
    outliers = agg_results["sample_outlier_summary"]

    print(f"Deletion AUC (Higher is Better):")
    print(f"  Integrated Gradients (IG) : {del_auc['ig_auc']['mean']:.4f} (95% CI: [{del_auc['ig_auc']['ci_95'][0]:.4f}, {del_auc['ig_auc']['ci_95'][1]:.4f}])")
    print(f"  Cross-Attention Weights   : {del_auc['attn_auc']['mean']:.4f} (95% CI: [{del_auc['attn_auc']['ci_95'][0]:.4f}, {del_auc['attn_auc']['ci_95'][1]:.4f}])")
    print(f"  Random Baseline (M={args.n_random_seeds})  : {del_auc['random_auc']['mean']:.4f} (95% CI: [{del_auc['random_auc']['ci_95'][0]:.4f}, {del_auc['random_auc']['ci_95'][1]:.4f}])")
    print(f"  Paired t-test (IG vs Rand): p = {del_auc['p_val_ttest_ig_vs_rand']:.4e}")

    print(f"\nInsertion AUC (Lower is Better):")
    print(f"  Integrated Gradients (IG) : {ins_auc['ig_auc']['mean']:.4f} (95% CI: [{ins_auc['ig_auc']['ci_95'][0]:.4f}, {ins_auc['ig_auc']['ci_95'][1]:.4f}])")
    print(f"  Cross-Attention Weights   : {ins_auc['attn_auc']['mean']:.4f} (95% CI: [{ins_auc['attn_auc']['ci_95'][0]:.4f}, {ins_auc['attn_auc']['ci_95'][1]:.4f}])")
    print(f"  Random Baseline (M={args.n_random_seeds})  : {ins_auc['random_auc']['mean']:.4f} (95% CI: [{ins_auc['random_auc']['ci_95'][0]:.4f}, {ins_auc['random_auc']['ci_95'][1]:.4f}])")
    print(f"  Paired t-test (IG vs Rand): p = {ins_auc['p_val_ttest_ig_vs_rand']:.4e}")

    print(f"\nSample-Level Outlier Summary (Diagnostic):")
    print(f"  IG Deletion Z-score >= 2.0  : {outliers['frac_samples_ig_deletion_outlier_zscore']:.1%}")
    print(f"  IG Deletion Percentile >= 95%: {outliers['frac_samples_ig_deletion_outlier_percentile']:.1%}")
    print(f"  IG Insertion Z-score <= -2.0 : {outliers['frac_samples_ig_insertion_outlier_zscore']:.1%}")
    print(f"  IG Insertion Percentile <= 5%: {outliers['frac_samples_ig_insertion_outlier_percentile']:.1%}")
    print("=" * 70)

    # 6. Generate Figure 3 Plots
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Extract K-step mean curves and CIs across samples
        ig_del_k = np.array([r["ig_deletion_mses"] for r in sample_results])
        attn_del_k = np.array([r["attn_deletion_mses"] for r in sample_results])
        rand_del_k = np.array([np.mean(r["random_deletion_mses"], axis=0) for r in sample_results]) # Average M seeds per sample

        ig_ins_k = np.array([r["ig_insertion_mses"] for r in sample_results])
        attn_ins_k = np.array([r["attn_insertion_mses"] for r in sample_results])
        rand_ins_k = np.array([np.mean(r["random_insertion_mses"], axis=0) for r in sample_results])

        def get_k_cis(matrix):
            means = np.mean(matrix, axis=0)
            lows, highs = [], []
            for col in range(matrix.shape[1]):
                _, l, h = compute_bootstrap_ci(matrix[:, col])
                lows.append(l)
                highs.append(h)
            return means, np.array(lows), np.array(highs)

        ig_d_mean, ig_d_low, ig_d_high = get_k_cis(ig_del_k)
        attn_d_mean, attn_d_low, attn_d_high = get_k_cis(attn_del_k)
        rand_d_mean, rand_d_low, rand_d_high = get_k_cis(rand_del_k)

        ig_i_mean, ig_i_low, ig_i_high = get_k_cis(ig_ins_k)
        attn_i_mean, attn_i_low, attn_i_high = get_k_cis(attn_ins_k)
        rand_i_mean, rand_i_low, rand_i_high = get_k_cis(rand_ins_k)

        # Figure 3A: Deletion Curves
        ax1 = axes[0]
        ax1.plot(k_values, ig_d_mean, 'o-', color='#1f77b4', linewidth=2.5, label='Integrated Gradients')
        ax1.fill_between(k_values, ig_d_low, ig_d_high, color='#1f77b4', alpha=0.2)

        ax1.plot(k_values, attn_d_mean, 's--', color='#ff7f0e', linewidth=2.0, label='Cross-Attention')
        ax1.fill_between(k_values, attn_d_low, attn_d_high, color='#ff7f0e', alpha=0.15)

        ax1.plot(k_values, rand_d_mean, '^:', color='#7f7f7f', linewidth=2.0, label=f'Random Baseline (M={args.n_random_seeds})')
        ax1.fill_between(k_values, rand_d_low, rand_d_high, color='#7f7f7f', alpha=0.15)

        ax1.set_title("Figure 3A: Deletion Curves (MSE vs K)\n(Higher AUC is Better)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Patches Deleted (K)", fontsize=10)
        ax1.set_ylabel("Prediction Error (MSE)", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=9)

        # Figure 3B: Insertion Curves
        ax2 = axes[1]
        ax2.plot(k_values, ig_i_mean, 'o-', color='#1f77b4', linewidth=2.5, label='Integrated Gradients')
        ax2.fill_between(k_values, ig_i_low, ig_i_high, color='#1f77b4', alpha=0.2)

        ax2.plot(k_values, attn_i_mean, 's--', color='#ff7f0e', linewidth=2.0, label='Cross-Attention')
        ax2.fill_between(k_values, attn_i_low, attn_i_high, color='#ff7f0e', alpha=0.15)

        ax2.plot(k_values, rand_i_mean, '^:', color='#7f7f7f', linewidth=2.0, label=f'Random Baseline (M={args.n_random_seeds})')
        ax2.fill_between(k_values, rand_i_low, rand_i_high, color='#7f7f7f', alpha=0.15)

        ax2.set_title("Figure 3B: Insertion Curves (MSE vs K)\n(Lower AUC is Better)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Patches Restored (K)", fontsize=10)
        ax2.set_ylabel("Prediction Error (MSE)", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=9)

        # Figure 3C: Normalized AUC Comparison Bar Chart
        ax3 = axes[2]
        x_indices = np.arange(2)
        width = 0.25

        ig_aucs_val = [del_auc['ig_auc']['mean'], ins_auc['ig_auc']['mean']]
        attn_aucs_val = [del_auc['attn_auc']['mean'], ins_auc['attn_auc']['mean']]
        rand_aucs_val = [del_auc['random_auc']['mean'], ins_auc['random_auc']['mean']]

        ig_errs = [[del_auc['ig_auc']['mean'] - del_auc['ig_auc']['ci_95'][0], ins_auc['ig_auc']['mean'] - ins_auc['ig_auc']['ci_95'][0]],
                   [del_auc['ig_auc']['ci_95'][1] - del_auc['ig_auc']['mean'], ins_auc['ig_auc']['ci_95'][1] - ins_auc['ig_auc']['mean']]]

        ax3.bar(x_indices - width, ig_aucs_val, width, yerr=ig_errs, capsize=4, color='#1f77b4', label='IG')
        ax3.bar(x_indices, attn_aucs_val, width, color='#ff7f0e', label='Attention')
        ax3.bar(x_indices + width, rand_aucs_val, width, color='#7f7f7f', label='Random')

        ax3.set_xticks(x_indices)
        ax3.set_xticklabels(['Deletion AUC\n(Higher=Better)', 'Insertion AUC\n(Lower=Better)'], fontsize=9, fontweight='bold')
        ax3.set_ylabel("Normalized AUC", fontsize=10)
        ax3.set_title("Figure 3C: Normalized AUC Comparison", fontsize=11, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax3.legend(fontsize=9)

        fig.suptitle(
            "Task 1: Causal Context Selection via Deletion & Insertion AUC Benchmarks\n"
            "Note: Unlike standard confidence-based RISE insertion/deletion AUC, here higher deletion-AUC and lower insertion-AUC\n"
            "both indicate superior attribution quality because the target metric is Prediction Error (MSE) rather than model confidence.",
            fontsize=10, fontstyle='italic', y=1.03
        )

        plt.tight_layout()
        plt.savefig(args.output_fig, dpi=300, bbox_inches='tight')
        print(f"[Plot] Saved Figure 3 to {args.output_fig}")

        if not os.environ.get("SAVE_PLOT"):
            plt.show()

    except ImportError:
        print("[Warning] matplotlib not installed, skipping plot generation.")


if __name__ == "__main__":
    main()
