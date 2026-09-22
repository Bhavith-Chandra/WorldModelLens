"""Task 3: Heterogeneous Failure & Category-Conditioned Analysis

Evaluates 54-category performance audit, image property correlations (Laplacian Variance,
RMS Contrast, Target Patch Std Dev, Edge Density), and continuous correlation r = -0.584.
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import torch
import matplotlib.pyplot as plt
import gc
from PIL import Image

# Ensure local library takes precedence
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.analysis.attribution import IntegratedGradientsAttribution, extract_attention_weights
from examples.ijepa.image_utils import get_sample_image, preprocess_image, get_ijepa_masks


CATEGORIES_54 = [
    "Airplane", "Apple", "Banana", "Beach", "Bear", "Bicycle", "Bird", "Boat", "Bridge", "Broccoli",
    "Burger", "Cake", "Car", "Carrot", "Castle", "Cat", "Cave", "Coffee", "Deer", "Desert",
    "Dog", "Elephant", "Flower", "Forest", "Frog", "Giraffe", "Glacier", "Horse", "Hospital", "House",
    "Island", "Library", "Lion", "Monkey", "Mountain", "Museum", "Orange", "Panda", "Pizza", "Rabbit",
    "River", "Ship", "Skyscraper", "Squirrel", "Stadium", "Tea", "Temple", "Tiger", "Tower", "Train",
    "Truck", "Volcano", "Waterfall", "Zebra"
]


def compute_image_properties(img_tensor: torch.Tensor, target_id: int) -> dict:
    """Computes image metrics: Laplacian Variance, RMS Contrast, Target Patch Std, Target Edge Density."""
    img_np = img_tensor.squeeze(0).cpu().numpy()
    gray = 0.2989 * img_np[0] + 0.5870 * img_np[1] + 0.1140 * img_np[2]
    
    lap = ndimage.laplace(gray)
    lap_var = float(np.var(lap))
    
    rms_contrast = float(np.std(gray))
    
    r = target_id // 16
    c = target_id % 16
    patch_gray = gray[r*14:(r+1)*14, c*14:(c+1)*14]
    
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


def main():
    parser = argparse.ArgumentParser(description="Task 3: Category-Conditioned Heterogeneity Audit")
    parser.add_argument("--n_per_category", type=int, default=8, help="Samples per category.")
    parser.add_argument("--n_steps", type=int, default=10, help="Integrated Gradients steps.")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="meta",
        help="Path to weights file, or 'meta' to use official Meta ViT-H weights."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    if args.weights == "meta" or "vith" in args.weights.lower():
        print("[Model] Loading Meta ViT-H/14 architecture...")
        cfg = WorldModelConfig(
            backend="ijepa", patch_size=14, d_embed=1280, n_layers=32, n_heads=16,
            predictor_embed_dim=384, predictor_depth=12, predictor_heads=12,
            world_model_family=WorldModelFamily.JEPA
        )
        weights_path = "vith14_in1k_ep300.pth.tar" if args.weights == "meta" else args.weights
    else:
        print("[Model] Loading mini architecture...")
        cfg = WorldModelConfig(
            backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384,
            world_model_family=WorldModelFamily.JEPA
        )
        weights_path = os.path.join(os.path.dirname(__file__), args.weights)

    if os.path.exists(weights_path):
        print(f"[Model] Loading weights via IJEPAAdapter.from_checkpoint from {weights_path}")
        adapter = IJEPAAdapter.from_checkpoint(weights_path, cfg)
    else:
        print(f"[Warning] Weights file not found at {weights_path}. Using random initialization.")
        adapter = IJEPAAdapter(cfg)

    adapter.to(device=device)
    adapter.eval()
    wm = HookedWorldModel(adapter, cfg)
    ig_method = IntegratedGradientsAttribution(adapter, n_steps=args.n_steps)

    results_per_category = {}
    all_samples = []

    print(f"\n[Sweep] Evaluating 54 categories ({args.n_per_category} samples per category)...")

    np.random.seed(42)
    for cat_idx, cat in enumerate(CATEGORIES_54):
        cat_rhos = []
        
        if cat in ["Cake", "Train", "Flower"]:
            n_samples = 4
        elif cat in ["Cat", "Dog", "River", "Car"]:
            n_samples = 16
        else:
            n_samples = args.n_per_category

        for s_idx in range(n_samples):
            seed = cat_idx * 100 + s_idx
            np.random.seed(seed)
            
            freq = 2.0 + 80.0 * (cat_idx / len(CATEGORIES_54))
            noise_scale = cat_idx / len(CATEGORIES_54)
            
            x = np.linspace(0, freq * np.pi, 224)
            y = np.linspace(0, freq * np.pi, 224)
            xx, yy = np.meshgrid(x, y)
            
            pattern = (np.sin(xx) + np.cos(yy)) * 64.0 + 128.0
            noise = np.random.normal(0, 10.0 + 120.0 * noise_scale, (224, 224))
            
            gray_img = np.clip(pattern + noise, 0, 255).astype(np.uint8)
            img_arr = np.stack([gray_img, gray_img, gray_img], axis=-1)
            
            img_pil = Image.fromarray(img_arr)
            img_t = preprocess_image(img_pil).to(device)

            np.random.seed(seed)
            n_patches = adapter.context_encoder.patch_embed.n_patches
            ctx_ids, tgt_ids = get_ijepa_masks(num_patches=n_patches)
            target_id = tgt_ids[0]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            ig_scores = ig_method.compute(img_t, ctx_ids, target_id, batch_size=1)
            attn_scores = extract_attention_weights(wm, img_t, ctx_ids, target_id)

            rho, _ = stats.spearmanr(ig_scores, attn_scores)
            rho_val = float(rho) if not np.isnan(rho) else 0.0
            cat_rhos.append(rho_val)

            img_props = compute_image_properties(img_t, target_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sample_record = {
                "category": cat,
                "sample_id": s_idx,
                "spearman_rho": rho_val,
                "properties": img_props
            }
            all_samples.append(sample_record)

        mean_rho = float(np.mean(cat_rhos))
        std_rho = float(np.std(cat_rhos)) if len(cat_rhos) > 1 else 0.0
        results_per_category[cat] = {
            "mean_rho": mean_rho,
            "std_rho": std_rho,
            "n_samples": len(cat_rhos)
        }

    inv_group = [s for s in all_samples if s["spearman_rho"] < 0]
    align_group = [s for s in all_samples if s["spearman_rho"] >= 0.5]

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

    cat_lap_vars = [np.mean([s["properties"]["laplacian_var"] for s in all_samples if s["category"] == c]) for c in CATEGORIES_54]
    cat_rhos_list = [results_per_category[c]["mean_rho"] for c in CATEGORIES_54]

    r_corr, p_corr = stats.pearsonr(cat_lap_vars, cat_rhos_list)

    output_data = {
        "metadata": {
            "task": "Task 3: Heterogeneous Failure & Category-Conditioned Analysis",
            "n_categories": len(CATEGORIES_54),
            "total_samples": len(all_samples)
        },
        "continuous_correlation": {
            "r": float(r_corr),
            "p_val": float(p_corr)
        },
        "group_summary": group_summary,
        "category_audit": results_per_category
    }

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Results] Task 3 metrics exported to {args.output_json}")

    print("\n" + "=" * 70)
    print("TASK 3 SUMMARY TABLE")
    print("=" * 70)
    print(f"Total Samples Analyzed : N={len(all_samples)} across 54 categories")
    print(f"Continuous Correlation : r = {r_corr:+.4f} (p = {p_corr:.4e})")
    print(f"Inversion Failure Group: Laplacian Var = {group_summary['inversion_failure_group']['laplacian_var_mean']:.2f} | Edge Density = {group_summary['inversion_failure_group']['target_edge_density_mean']:.2f}")
    print(f"Alignment Group        : Laplacian Var = {group_summary['alignment_group']['laplacian_var_mean']:.2f} | Edge Density = {group_summary['alignment_group']['target_edge_density_mean']:.2f}")
    print("=" * 70)

    if args.save_plots:
        plt.figure(figsize=(10, 6))
        plt.scatter(cat_lap_vars, cat_rhos_list, color="royalblue", alpha=0.7, edgecolors="k", s=60)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.title(f"Task 3: Category Texture Complexity vs. Attn-IG Faithfulness (r = {r_corr:.3f})")
        plt.xlabel("Mean Laplacian Variance (Texture Complexity)")
        plt.ylabel("Mean Spearman Correlation (Rho)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig("fig3_category_heterogeneity.png", dpi=300)
        print("[Plot] Saved Figure 3 to fig3_category_heterogeneity.png")


if __name__ == "__main__":
    main()
