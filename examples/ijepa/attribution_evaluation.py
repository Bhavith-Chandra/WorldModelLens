"""Attribution Evaluation Example

Demonstrates how to use the AttributionEvaluator to compute dataset-level
aggregate metrics for causal sensitivity in I-JEPA patch predictions.
"""

import os
import torch
import numpy as np

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.analysis.attribution import (
    IntegratedGradientsAttribution,
    GradientXInputAttribution,
    AttributionEvaluator,
)
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate I-JEPA patch attribution.")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="ijepa_mini.pth",
        help="Path to weights file, or 'meta' to use the official Meta ViT-H weights."
    )
    args = parser.parse_args()

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
    print("Preparing validation dataset...")
    
    # We will use a few different validation images to ensure diversity
    image_urls = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/JPEG_example_JPG_RIP_025.jpg/250px-JPEG_example_JPG_RIP_025.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Pug_600.jpg/300px-Pug_600.jpg"
    ]
    
    dataset = []
    # We want N=50 samples for a fast demo
    samples_per_image = 50 // len(image_urls)
    
    for url in image_urls:
        raw_img = get_sample_image(url)
        img_tensor = preprocess_image(raw_img)
        
        # Pick targets per image to reach N=50
        target_ids = list(range(10, 190, max(1, 190 // samples_per_image)))[:samples_per_image]
        
        for tid in target_ids:
            # Get random context masks
            context_ids, _ = get_ijepa_masks(num_context=80)
            if tid in context_ids:
                context_ids.remove(tid)
            dataset.append((img_tensor, context_ids, tid))

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

    # Evaluate Integrated Gradients
    print("\n--- Evaluating Integrated Gradients (50 steps) ---")
    ig_method = IntegratedGradientsAttribution(adapter, n_steps=50)
    
    for layer_idx in layers_to_test:
        print(f"\n>>> Results for Predictor Layer {layer_idx} <<<")
        ig_results = evaluator.evaluate_dataset(
            wm, ig_method, dataset, alignment_threshold=0.6, failure_threshold=0.3, layer_idx=layer_idx
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
