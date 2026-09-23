"""Task 1 with the strict ModelHub loader and shared ImageNet data pipeline.

Evaluates Integrated Gradients (IG) vs Cross-Attention Weights vs Random Patch Baseline (M=20 seeds)
across patch counts K in {1, 3, 5, 10, 20} with 95% bootstrap confidence intervals and paired statistical tests.

This is a complete copy of the Task 1 experiment. Only checkpoint and image
loading differ from the original entry point.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import gc

# Ensure local library takes precedence
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.analysis.attribution import IntegratedGradientsAttribution
from world_model_lens.analysis.ablation_knockout import PatchKnockoutEvaluator, compute_bootstrap_ci
from world_model_lens.data import load_imagenet_image, load_imagenet_subset
from world_model_lens.hub.model_hub import ModelHub


def cleanup_memory():
    """Clear CUDA cache and collect garbage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def sample_context_and_target(
    num_patches: int,
    seed: int,
    visible_fraction: float = 0.20,
):
    """Sample a 20% visible context and one target from the 80% hidden pool."""
    if not 0.0 < visible_fraction < 1.0:
        raise ValueError("visible_fraction must be between zero and one")
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(num_patches)
    n_context = max(1, min(num_patches - 1, round(num_patches * visible_fraction)))
    context_ids = sorted(shuffled[:n_context].tolist())
    target_pool = shuffled[n_context:]
    return context_ids, int(target_pool[0]), int(len(target_pool))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 1: Deletion and Insertion AUC curves for I-JEPA.")
    parser.add_argument(
        "--weights",
        type=str,
        default="meta",
        help="Path to the official checkpoint, or 'meta' for vith14_in1k_ep300.pth.tar."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--subset_size",
        type=int,
        default=1000,
        help="Balanced ImageNet subset size selected before optional slicing.",
    )
    parser.add_argument(
        "--n_classes", type=int, default=50,
        help="Number of ImageNet classes in the balanced subset (default: 50)."
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Samples to evaluate after start_idx (default: all remaining subset samples)."
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for evaluation.")
    parser.add_argument("--n_steps", type=int, default=50, help="Integrated Gradients steps.")
    parser.add_argument("--n_random_seeds", type=int, default=20, help="Random seeds for null distribution (M=20 default).")
    parser.add_argument("--patch_ablation_mode", type=str, default="zero", choices=["zero", "dataset_mean_patch"], help="Patch ablation mode.")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="ImageNet root containing val/, train/, or class directories."
    )
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

    # 1. Load all three official checkpoint components through ModelHub. The
    # strict loader refuses missing target/predictor tensors instead of silently
    # substituting or leaving pretrained parameters randomly initialized.
    weights_path = "vith14_in1k_ep300.pth.tar" if args.weights == "meta" else args.weights
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"I-JEPA checkpoint not found: {weights_path}")
    print(f"[Model] Strict ModelHub load from {weights_path}")
    adapter = ModelHub.load_checkpoint(weights_path, backend="ijepa", device=args.device)
    adapter.eval()
    coverage = getattr(adapter, "checkpoint_coverage", {})
    print(
        "[Model] Loaded context encoder, EMA target encoder, and predictor: "
        + ", ".join(
            f"{name}={row['checkpoint_tensors']}/{row['model_tensors']}"
            for name, row in coverage.items()
        )
    )
    wm = HookedWorldModel(adapter, adapter.config)

    # 2. Build the same balanced deterministic ImageNet subset used by Tasks 4/7.
    print(
        f"\n[Data] Selecting N={args.subset_size} images across "
        f"{args.n_classes} ImageNet classes..."
    )
    manifest = load_imagenet_subset(
        args.data_dir,
        num_samples=args.subset_size,
        num_classes=args.n_classes,
        seed=42,
    )
    if args.start_idx < 0 or args.start_idx >= len(manifest):
        raise ValueError("start_idx must select an item inside the ImageNet subset")
    stop = None if args.n_samples is None else args.start_idx + args.n_samples
    selected = manifest[args.start_idx:stop]
    if not selected:
        raise RuntimeError("The requested ImageNet subset slice is empty")

    manifest_path = Path(args.output_json).with_name("task1_dataset_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    image_size = int(getattr(adapter.config, "img_size", 224))
    num_patches = int(adapter.context_encoder.patch_embed.n_patches)
    dataset = []
    hidden_pool_size = None
    for sample_idx, sample in enumerate(selected, start=args.start_idx):
        image = load_imagenet_image(sample["path"], image_size=image_size)
        context_ids, target_id, hidden_pool_size = sample_context_and_target(
            num_patches, seed=42 + sample_idx
        )
        dataset.append((image, context_ids, target_id, sample["class_name"]))

    print(
        f"[Data] Loaded {len(dataset)} evaluation images at {image_size}x{image_size}; "
        f"{len(dataset[0][1])}/{num_patches} visible context patches and "
        f"{hidden_pool_size}/{num_patches} hidden candidate targets."
    )

    # Compute optional dataset mean patch embedding if requested
    dataset_mean_patch = None
    if args.patch_ablation_mode == "dataset_mean_patch":
        print("[Ablation] Computing dataset mean patch embedding...")
        patch_sum = None
        patch_count = 0
        with torch.no_grad():
            for item in dataset:
                img = item[0].to(args.device)
                embeddings = adapter.context_encoder.patch_embed(img).squeeze(0)
                current_sum = embeddings.sum(dim=0)
                patch_sum = current_sum if patch_sum is None else patch_sum + current_sum
                patch_count += embeddings.shape[0]
        dataset_mean_patch = patch_sum / patch_count

    # 3. Pre-compute Integrated Gradients Attributions
    ig_method = IntegratedGradientsAttribution(adapter, n_steps=args.n_steps)
    cache_path = Path(args.output_json).with_name(
        f"task1_ig_cache_{len(dataset)}_{num_patches}p.pth"
    )
    cached_attributions = []

    if cache_path.exists():
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
            "checkpoint_path": str(Path(weights_path).resolve()),
            "loader": "ModelHub.load_checkpoint",
            "checkpoint_coverage": coverage,
            "n_samples": len(dataset),
            "imagenet_subset_size": args.subset_size,
            "imagenet_num_classes": args.n_classes,
            "dataset_manifest": str(manifest_path),
            "image_size": image_size,
            "num_patches": num_patches,
            "visible_context_fraction": 0.20,
            "visible_context_patches": len(dataset[0][1]),
            "hidden_target_pool_patches": hidden_pool_size,
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
