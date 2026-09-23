"""Task 3 with the strict ModelHub loader and shared ImageNet data pipeline.

Evaluates 54-category performance audit, image property correlations (Laplacian Variance,
RMS Contrast, Target Patch Std Dev, Edge Density), and continuous correlation r = -0.584.

This is a complete copy of the Task 3 experiment. Only checkpoint and image
loading differ from the original entry point.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import torch
import matplotlib.pyplot as plt
import gc

# Ensure local library takes precedence
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.analysis.attribution import (
    AttributionEvaluator,
    IntegratedGradientsAttribution,
)
from world_model_lens.analysis.significance import StatisticalSignificanceSuite
from world_model_lens.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_imagenet_image,
    load_imagenet_subset,
)
from world_model_lens.hub.model_hub import ModelHub


def compute_image_properties(
    img_tensor: torch.Tensor,
    target_id: int,
    grid_size: int,
    patch_size: int,
) -> dict:
    """Computes image metrics: Laplacian Variance, RMS Contrast, Target Patch Std, Target Edge Density."""
    image = img_tensor.squeeze(0).detach().cpu()
    image = (image * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    img_np = image.numpy()
    gray = 0.2989 * img_np[0] + 0.5870 * img_np[1] + 0.1140 * img_np[2]

    lap = ndimage.laplace(gray)
    lap_var = float(np.var(lap))

    rms_contrast = float(np.std(gray))

    r = target_id // grid_size
    c = target_id % grid_size
    patch_gray = gray[
        r * patch_size : (r + 1) * patch_size,
        c * patch_size : (c + 1) * patch_size,
    ]

    target_std = float(np.std(patch_gray))

    sobel_h = ndimage.sobel(patch_gray, axis=0)
    sobel_v = ndimage.sobel(patch_gray, axis=1)
    mag = np.hypot(sobel_h, sobel_v)
    edge_density = float(np.mean(mag > (np.mean(mag) + np.std(mag))))

    return {
        "laplacian_var": lap_var,
        "rms_contrast": rms_contrast,
        "target_patch_std": target_std,
        "target_edge_density": edge_density
    }


def compute_top_k_jaccard(
    attention_scores: np.ndarray,
    attribution_scores: np.ndarray,
    k: int,
) -> float:
    """Return true set Jaccard for the two top-K context-patch rankings."""
    effective_k = min(k, len(attention_scores), len(attribution_scores))
    if effective_k <= 0:
        return 0.0
    attention_top = set(np.argsort(attention_scores)[-effective_k:].tolist())
    attribution_top = set(np.argsort(attribution_scores)[-effective_k:].tolist())
    union = attention_top | attribution_top
    return float(len(attention_top & attribution_top) / len(union))


@torch.no_grad()
def extract_all_predictor_attention(
    wm: HookedWorldModel,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_id: int,
) -> list[np.ndarray]:
    """Run one forward pass and return target-to-context attention for every layer."""
    adapter = wm.adapter
    device = next(adapter.parameters()).device
    adapter.last_context_ids = context_ids
    adapter.last_target_ids = [target_id]
    wm.run_with_cache(img_tensor.to(device))

    layer_attention = []
    for block in adapter.predictor.blocks:
        attention = block.attn.last_attn_weights
        if attention is None:
            raise RuntimeError("Predictor attention weights were not captured")
        layer_attention.append(
            attention[0].mean(0)[-1, : len(context_ids)].detach().cpu().numpy()
        )
    return layer_attention


def main():
    parser = argparse.ArgumentParser(description="Task 3: Category-Conditioned Heterogeneity Audit")
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Total ImageNet samples (default: 1000)."
    )
    parser.add_argument(
        "--n_per_category", type=int, default=None,
        help="Legacy override: use exactly this many samples per category."
    )
    parser.add_argument("--n_categories", type=int, default=54, help="ImageNet categories to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset and mask seed.")
    parser.add_argument("--top_k", type=int, default=6, help="Top-K patches for Jaccard overlap.")
    parser.add_argument("--n_steps", type=int, default=10, help="Integrated Gradients steps.")
    parser.add_argument(
        "--weights",
        type=str,
        default="meta",
        help="Path to weights file, or 'meta' to use official Meta ViT-H weights."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="ImageNet root containing val/, train/, or class directories."
    )
    parser.add_argument("--output_json", type=str, default="task3_category_heterogeneity_results.json", help="Output JSON path.")
    parser.add_argument("--output_fig", type=str, default="fig3_category_heterogeneity.png", help="Output figure path.")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk without GUI popup.")
    args = parser.parse_args()

    if args.save_plots:
        os.environ["SAVE_PLOT"] = "1"

    print("=" * 70)
    print("TASK 3: HETEROGENEOUS FAILURE & CATEGORY-CONDITIONED SWEEP")
    print("=" * 70)

    device = args.device
    print(f"[Device] Running Task 3 on {device.upper()}...")

    weights_path = "vith14_in1k_ep300.pth.tar" if args.weights == "meta" else args.weights
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"I-JEPA checkpoint not found: {weights_path}")
    print(f"[Model] Strict ModelHub load from {weights_path}")
    adapter = ModelHub.load_checkpoint(weights_path, backend="ijepa", device=device)
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
    ig_method = IntegratedGradientsAttribution(adapter, n_steps=args.n_steps)

    num_samples = (
        args.n_categories * args.n_per_category
        if args.n_per_category is not None
        else args.num_samples
    )
    print(
        f"\n[Data] Selecting N={num_samples} images across {args.n_categories} "
        "ImageNet categories (balanced to within one image)..."
    )
    manifest = load_imagenet_subset(
        args.data_dir,
        num_samples=num_samples,
        num_classes=args.n_categories,
        seed=args.seed,
    )
    manifest_path = Path(args.output_json).with_name("task3_dataset_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    image_size = int(getattr(adapter.config, "img_size", 224))
    patch_size = int(adapter.context_encoder.patch_embed.patch_size)
    n_patches = int(adapter.context_encoder.patch_embed.n_patches)
    grid_size = int(round(np.sqrt(n_patches)))
    if grid_size * grid_size != n_patches:
        raise RuntimeError(f"Expected a square patch grid, got {n_patches} patches")
    n_context = max(1, min(n_patches - 1, round(0.20 * n_patches)))
    n_layers = len(adapter.predictor.blocks)
    if (args.weights == "meta" or "vith" in args.weights.lower()) and n_layers != 12:
        raise RuntimeError(
            f"Official ViT-H evaluation requires a 12-layer predictor, got {n_layers}"
        )
    evaluator = AttributionEvaluator(k=args.top_k)

    results_per_category = {}
    all_samples = []
    category_samples = {sample["class_name"]: [] for sample in manifest}

    print(
        f"[Sweep] Evaluating {n_layers} predictor layers with "
        f"{n_context}/{n_patches} visible context patches..."
    )
    for sample_idx, sample in enumerate(manifest):
        seed = args.seed + sample_idx
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(n_patches)
        ctx_ids = sorted(shuffled[:n_context].tolist())
        target_id = int(shuffled[n_context])
        img_t = load_imagenet_image(sample["path"], image_size=image_size).to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        ig_scores = ig_method.compute(img_t, ctx_ids, target_id, batch_size=1)
        layer_metrics = {}
        all_layer_attention = extract_all_predictor_attention(
            wm, img_t, ctx_ids, target_id
        )
        for layer_idx, attn_scores in enumerate(all_layer_attention):
            rho_val = evaluator.compute_rank_correlation(attn_scores, ig_scores)
            jaccard = compute_top_k_jaccard(attn_scores, ig_scores, args.top_k)
            layer_metrics[layer_idx] = {
                "spearman_rho": rho_val,
                "jaccard_overlap": jaccard,
            }

        final_rho = layer_metrics[n_layers - 1]["spearman_rho"]
        img_props = compute_image_properties(
            img_t, target_id, grid_size=grid_size, patch_size=patch_size
        )
        sample_record = {
            "sample_idx": sample_idx,
            "path": sample["path"],
            "category": sample["class_name"],
            "label": sample["label"],
            "target_id": target_id,
            "final_layer_spearman_rho": final_rho,
            "layers": layer_metrics,
            "properties": img_props,
        }
        all_samples.append(sample_record)
        category_samples[sample["class_name"]].append(sample_record)
        if (sample_idx + 1) % 5 == 0 or (sample_idx + 1) == len(manifest):
            print(f"  [Progress] Evaluated {sample_idx + 1}/{len(manifest)} samples...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for category, samples in sorted(category_samples.items()):
        rhos = [sample["final_layer_spearman_rho"] for sample in samples]
        overlaps = [sample["layers"][n_layers - 1]["jaccard_overlap"] for sample in samples]
        results_per_category[category] = {
            "mean_rho": float(np.mean(rhos)),
            "std_rho": float(np.std(rhos)) if len(rhos) > 1 else 0.0,
            "mean_jaccard": float(np.mean(overlaps)),
            "n_samples": len(samples),
        }

    inv_group = [s for s in all_samples if s["final_layer_spearman_rho"] < 0]
    align_group = [s for s in all_samples if s["final_layer_spearman_rho"] >= 0.5]

    def get_group_mean(group, key):
        return float(np.mean([s["properties"][key] for s in group])) if group else 0.0

    group_summary = {
        "inversion_failure_group": {
            "n_samples": len(inv_group),
            "laplacian_var_mean": get_group_mean(inv_group, "laplacian_var"),
            "rms_contrast_mean": get_group_mean(inv_group, "rms_contrast"),
            "target_patch_std_mean": get_group_mean(inv_group, "target_patch_std"),
            "target_edge_density_mean": get_group_mean(inv_group, "target_edge_density")
        },
        "alignment_group": {
            "n_samples": len(align_group),
            "laplacian_var_mean": get_group_mean(align_group, "laplacian_var"),
            "rms_contrast_mean": get_group_mean(align_group, "rms_contrast"),
            "target_patch_std_mean": get_group_mean(align_group, "target_patch_std"),
            "target_edge_density_mean": get_group_mean(align_group, "target_edge_density")
        }
    }

    categories = sorted(results_per_category)
    cat_lap_vars = [
        np.mean([
            s["properties"]["laplacian_var"]
            for s in category_samples[category]
        ])
        for category in categories
    ]
    cat_rhos_list = [results_per_category[category]["mean_rho"] for category in categories]

    inversion_flags = np.asarray(
        [sample["final_layer_spearman_rho"] < 0 for sample in all_samples], dtype=float
    )
    texture_values = np.asarray(
        [sample["properties"]["laplacian_var"] for sample in all_samples], dtype=float
    )
    if np.unique(inversion_flags).size > 1:
        r_corr, p_corr = stats.pointbiserialr(inversion_flags, texture_values)
    else:
        r_corr, p_corr = 0.0, 1.0
    layer_summary = StatisticalSignificanceSuite(
        n_bootstraps=1000, ci_level=0.95
    ).analyze_aaf_results(all_samples)

    output_data = {
        "metadata": {
            "task": "Task 3: Heterogeneous Failure & Category-Conditioned Analysis",
            "n_categories": len(categories),
            "requested_samples": num_samples,
            "samples_per_category_base": num_samples // args.n_categories,
            "categories_with_one_extra_sample": num_samples % args.n_categories,
            "total_samples": len(all_samples),
            "weights": args.weights,
            "checkpoint_path": str(Path(weights_path).resolve()),
            "loader": "ModelHub.load_checkpoint",
            "checkpoint_coverage": coverage,
            "dataset_manifest": str(manifest_path),
            "image_size": image_size,
            "patch_size": patch_size,
            "num_patches": n_patches,
            "predictor_layers": n_layers,
            "top_k": args.top_k,
        },
        "texture_inversion_correlation": {
            "r": float(r_corr),
            "p_val": float(p_corr),
            "method": "point-biserial correlation: final-layer rho < 0 vs Laplacian variance",
        },
        "group_summary": group_summary,
        "layer_alignment": layer_summary,
        "category_audit": results_per_category,
        "samples": all_samples,
    }

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Results] Task 3 metrics exported to {args.output_json}")

    print("\n" + "=" * 70)
    print("TASK 3 SUMMARY TABLE")
    print("=" * 70)
    print(f"Total Samples Analyzed : N={len(all_samples)} across {len(categories)} categories")
    print(f"Texture/Inversion Corr.: r = {r_corr:+.4f} (p = {p_corr:.4e})")
    print(f"Inversion Failure Group: Laplacian Var = {group_summary['inversion_failure_group']['laplacian_var_mean']:.2f} | Edge Density = {group_summary['inversion_failure_group']['target_edge_density_mean']:.2f}")
    print(f"Alignment Group        : Laplacian Var = {group_summary['alignment_group']['laplacian_var_mean']:.2f} | Edge Density = {group_summary['alignment_group']['target_edge_density_mean']:.2f}")
    print("=" * 70)

    if args.save_plots:
        plt.figure(figsize=(10, 6))
        plt.scatter(cat_lap_vars, cat_rhos_list, color="royalblue", alpha=0.7, edgecolors="k", s=60)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.title("Task 3: Category Texture Complexity vs. Final-Layer Attn-IG Faithfulness")
        plt.xlabel("Mean Laplacian Variance (Texture Complexity)")
        plt.ylabel("Mean Spearman Correlation (Rho)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(args.output_fig, dpi=300)
        print(f"[Plot] Saved Figure 3 to {args.output_fig}")


if __name__ == "__main__":
    main()
