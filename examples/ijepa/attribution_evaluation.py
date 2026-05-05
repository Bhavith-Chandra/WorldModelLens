"""Attribution Evaluation Example

Demonstrates how to use the AttributionEvaluator to compute dataset-level
aggregate metrics for causal sensitivity in I-JEPA patch predictions.
"""

import os
import sys
import torch
import numpy as np
import gc
import argparse
# Ensure local library takes precedence over any stale site-packages install
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.analysis.attribution import (
    IntegratedGradientsAttribution,
    GradientXInputAttribution,
    AttributionEvaluator,
)
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks

def cleanup_memory():
    """Clear CUDA cache and collect garbage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Evaluate I-JEPA patch attribution.")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="ijepa_mini.pth",
        help="Path to weights file, or 'meta' to use the official Meta ViT-H weights."
    )
    parser.add_argument("--n_samples", type=int, default=50, help="Total samples to evaluate.")
    parser.add_argument("--n_steps", type=int, default=50, help="Integrated Gradients steps.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to local image dataset.")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk instead of showing them.")
    args = parser.parse_args()

    if args.save_plots:
        os.environ["SAVE_PLOT"] = "1"

    # Initial cleanup
    cleanup_memory()

    print("Initializing model...")
    # Setup config based on weights
    if args.weights == "meta" or "vith" in args.weights.lower():
        print("Using Meta ViT-H configuration.")
        config = WorldModelConfig(
            backend="ijepa", d_embed=1280, n_layers=32, n_heads=16, predictor_embed_dim=384
        )
        weights_path = "vith14_in1k_ep300.pth.tar" if args.weights == "meta" else args.weights
    else:
        print("Using default mini configuration.")
        config = WorldModelConfig(
            backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384
        )
        weights_path = os.path.join(os.path.dirname(__file__), args.weights)

    adapter = IJEPAAdapter(config)

    # Load weights if available
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        adapter.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True), strict=False
        )
    else:
        print(f"Warning: Weights not found at {weights_path}. Using random initialization.")

    adapter.eval()
    wm = HookedWorldModel(adapter, config)

    # Prepare validation dataset
    print("Preparing validation images...")
    np.random.seed(42) # Ensure deterministic patch sampling for cache consistency
    
    # Category-aware local image loading
    image_sources = []
    category_map = {} # category -> list of paths
    
    if args.data_dir and os.path.isdir(args.data_dir):
        # Scan subdirectories for categories
        for root, dirs, files in os.walk(args.data_dir):
            category = os.path.basename(root)
            if category == os.path.basename(args.data_dir):
                category = "unspecified"
            
            valid_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if valid_files:
                if category not in category_map:
                    category_map[category] = []
                category_map[category].extend(valid_files)
        
        n_categories = len(category_map)
        if n_categories > 0:
            print(f"Found {n_categories} categories in {args.data_dir}")
            samples_per_category = max(1, args.n_samples // n_categories)
            
            for cat, paths in category_map.items():
                # Sample evenly from each category
                sampled_paths = paths[:samples_per_category]
                image_sources.extend(sampled_paths)
            
            print(f"Sampled {len(image_sources)} images across {n_categories} categories.")
    
    if not image_sources:
        # Fallback to current behavior if no categorical data found
        print("No categorical local images found. Using default URLs.")
        image_sources = [
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
            "https://upload.wikimedia.org/wikipedia/commons/e/e0/JPEG_example_JPG_RIP_025.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
        ]
    
    # Pre-run Validation: Load and verify images ONCE, cache the tensors
    print("Verifying image access and loading tensors...")
    loaded_images = [] # list of (src, img_tensor)
    for src in image_sources:
        try:
            raw_img = get_sample_image(src)
            img_tensor = preprocess_image(raw_img)
            loaded_images.append((src, img_tensor))
        except Exception as e:
            print(f"Warning: Could not load {src}: {e}")
    
    if not loaded_images:
        print("CRITICAL: No images accessible. Run will likely fail.")
        # Hard stop — can't build dataset without images
        return

    dataset = []
    samples_per_image = max(1, args.n_samples // len(loaded_images))
    
    for src, img_tensor in loaded_images:
        target_ids = list(range(10, 190, max(1, 190 // samples_per_image)))[:samples_per_image]
        
        for tid in target_ids:
            context_ids, _ = get_ijepa_masks(num_context=80)
            if tid in context_ids:
                context_ids.remove(tid)
            dataset.append((img_tensor, context_ids, tid))

    # Cap to exactly n_samples to ensure cache stays compatible across runs
    dataset = dataset[:args.n_samples]
    print(f"Created dataset with {len(dataset)} samples.")

    # Initialize evaluator
    evaluator = AttributionEvaluator(k=6)

    # Layers to evaluate
    predictor_depth = len(adapter.predictor.blocks)
    if args.weights == "meta" or predictor_depth > 10:
        layers_to_test = [predictor_depth // 3, 2 * predictor_depth // 3, predictor_depth - 1]
    else:
        layers_to_test = [0, predictor_depth // 2, predictor_depth - 1]
    
    print(f"Predictor depth: {predictor_depth}")

    # Evaluate Integrated Gradients (Pre-compute once for caching)
    cache_path = "ig_cache.pth"
    if os.path.exists(cache_path):
        print(f"\n--- Loading cached IG results from {cache_path} ---")
        # weights_only=False is required here because the cache contains numpy arrays
        cached_attributions = torch.load(cache_path, weights_only=False)
        if len(cached_attributions) != len(dataset):
            print(f"Warning: Cache size ({len(cached_attributions)}) mismatch with current dataset. Re-computing...")
            cached_attributions = []
    else:
        cached_attributions = []

    # Instantiate ig_method once — used for both caching and on-the-fly repair during sweep
    ig_method = IntegratedGradientsAttribution(adapter, n_steps=args.n_steps)

    if not cached_attributions:
        print(f"\n--- Pre-computing Integrated Gradients ({args.n_steps} steps) ---")
        
        for i, (img_tensor, context_ids, target_id) in enumerate(dataset):
            if i % 5 == 0:
                print(f"  [Progress] Computing IG for sample {i}/{len(dataset)}...")
                cleanup_memory()
                # Save progress to disk every 5 samples
                torch.save(cached_attributions, cache_path)
                
            attr = ig_method.compute(img_tensor, context_ids, target_id)
            cached_attributions.append(attr)
        
        # Final save
        torch.save(cached_attributions, cache_path)
        print(f"Done. IG results cached to {cache_path}")

    # Prepare results container
    results_to_save = {
        "metadata": {
            "weights": args.weights,
            "n_samples": len(dataset),
            "n_steps": args.n_steps,
            "predictor_depth": predictor_depth
        },
        "layer_results": {}
    }

    print("\n--- Running Multi-Layer Sweep (using cached IG) ---")
    
    for layer_idx in layers_to_test:
        print(f"\n>>> Results for Predictor Layer {layer_idx} <<<")
        ig_results = evaluator.evaluate_dataset(
            wm, ig_method, dataset, 
            alignment_threshold=0.6, 
            failure_threshold=0.3, 
            layer_idx=layer_idx,
            precomputed_attributions=cached_attributions
        )

        print(f"  N: {ig_results['n_samples']}")
        print(f"  Mean Jaccard Overlap: {ig_results['mean_overlap']:.3f} ± {ig_results['ci_overlap']:.3f} (95% CI)")
        print(f"  Mean Spearman Rank Corr: {ig_results['mean_spearman']:.3f} ± {ig_results['ci_spearman']:.3f} (95% CI)")
        print(f"  High Alignment Rate (O_k >= 0.6): {ig_results['alignment_rate']:.1%}")
        print(f"  Failure Rate (O_k <= 0.3): {ig_results['failure_rate']:.1%}")
        print(f"  Severe Failure Rate (O_k < 0.1): {ig_results['low_overlap_rate']:.1%}")
        print(f"  Ranking Inversion Rate (Spearman < 0): {ig_results['negative_spearman_rate']:.1%}")
        
        print(f"  --- Per-Head Analysis ---")
        print(f"  Mean O_k across heads: {ig_results['mean_overlap_across_heads']:.3f}")
        print(f"  Var O_k across heads: {ig_results['mean_var_overlap_heads']:.4f}")
        print(f"  Mean Spearman across heads: {ig_results['mean_corr_across_heads']:.3f}")
        print(f"  Var Spearman across heads: {ig_results['mean_var_corr_heads']:.4f}")

        # Record results for this layer
        results_to_save["layer_results"][str(layer_idx)] = {
            "mean_overlap": float(ig_results['mean_overlap']),
            "mean_spearman": float(ig_results['mean_spearman']),
            "failure_rate": float(ig_results['failure_rate']),
            "negative_spearman_rate": float(ig_results['negative_spearman_rate']),
            "mean_overlap_across_heads": float(ig_results['mean_overlap_across_heads']),
            "mean_corr_across_heads": float(ig_results['mean_corr_across_heads'])
        }

    # Save all layers to disk
    import json
    with open("attribution_results.json", "w") as f:
        json.dump(results_to_save, f, indent=4)
    print("\nResults saved to attribution_results.json")

    # Plotting for the final layer
    try:
        import matplotlib.pyplot as plt
        
        evaluator.plot_overlap_distributions(
            ig_results, 
            title=f"IG vs Attention: Overlap (Layer {layers_to_test[-1]})",
            save_path="ig_overlap_dist.png" if os.environ.get("SAVE_PLOT") else None
        )
        
        evaluator.plot_rank_correlation_distribution(
            ig_results,
            title=f"IG vs Attention: Spearman (Layer {layers_to_test[-1]})",
            save_path="ig_spearman_dist.png" if os.environ.get("SAVE_PLOT") else None
        )

        # Find the sample with the worst Spearman correlation
        worst_idx = np.argmin(ig_results["raw_correlations"])
        worst_corr = ig_results["raw_correlations"][worst_idx]
        worst_attn = ig_results["raw_attn_list"][worst_idx]
        worst_attr = ig_results["raw_attr_list"][worst_idx]
        
        print(f"\n--- Qualitative Example: Ranking Inversion (Layer {layers_to_test[-1]}) ---")
        worst_target_id = dataset[worst_idx][2]
        worst_img_tensor = dataset[worst_idx][0]
        
        print(f"  Target Patch ID: {worst_target_id}")
        print(f"  Spearman Correlation: {worst_corr:.3f}")
        
        target_gt = ig_method._get_target_gt(worst_img_tensor, worst_target_id)
        actual_emb = adapter.context_encoder.patch_embed(worst_img_tensor)
        worst_score = ig_method._forward_score(
            actual_emb, dataset[worst_idx][1], worst_target_id, target_gt.squeeze(0)
        )
        print(f"  Prediction Impact (MSE Score): {-worst_score.item():.4f}")

        evaluator.plot_rank_scatter(
            worst_attn, worst_attr,
            title=f"Attention vs IG Rank (Target {worst_target_id}, Layer {layers_to_test[-1]})",
            save_path="ig_rank_scatter.png" if os.environ.get("SAVE_PLOT") else None
        )

        if not os.environ.get("SAVE_PLOT"):
            plt.show()
        else:
            print("\nSaved distribution and scatter plots to disk.")

    except ImportError:
        print("\nmatplotlib not installed, skipping plots.")


if __name__ == "__main__":
    main()
