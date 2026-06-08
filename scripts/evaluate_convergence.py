#!/usr/bin/env python3
"""Evaluate Layer-wise Predictor Convergence.

This script runs the LayerCKAAnalyzer on a provided model checkpoint
and generates a markdown report and convergence curve plots.
"""

import os
import sys
import argparse
import torch
import json
import matplotlib.pyplot as plt

# Ensure local library is on path
sys.path.insert(0, os.path.abspath("."))

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.analysis.layer_cka import LayerCKAAnalyzer
from examples.ijepa.image_utils import get_sample_image, preprocess_image

def main():
    parser = argparse.ArgumentParser(description="Evaluate Layer-wise Predictor Convergence.")
    parser.add_argument("--weights", type=str, default="ijepa_mini.pth", help="Path to weights file.")
    parser.add_argument("--output_dir", type=str, default="convergence_report", help="Output directory for reports.")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples to evaluate on.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing model...")
    if "vith" in args.weights.lower() or args.weights == "meta":
        config = WorldModelConfig(backend="ijepa", d_embed=1280, n_layers=32, n_heads=16, predictor_embed_dim=384)
    else:
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384)

    adapter = IJEPAAdapter(config)
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        adapter.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=False)
    else:
        print(f"Warning: Weights not found at {args.weights}. Using random init.")
    
    adapter.eval()
    wm = HookedWorldModel(adapter, config)

    print("Loading test images...")
    # Load some default test images
    urls = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    ]
    tensors = []
    for url in urls:
        try:
            img = get_sample_image(url)
            tensors.append(preprocess_image(img))
        except Exception as e:
            print(f"Failed to load {url}: {e}")
            
    if not tensors:
        print("Using random synthetic data due to image load failure.")
        batch = torch.randn(args.n_samples, 3, 224, 224)
    else:
        # Repeat to match n_samples
        batch = torch.cat(tensors, dim=0)
        if batch.size(0) < args.n_samples:
            repeats = (args.n_samples // batch.size(0)) + 1
            batch = batch.repeat(repeats, 1, 1, 1)[:args.n_samples]
        elif batch.size(0) > args.n_samples:
            batch = batch[:args.n_samples]

    print("Running Layer CKA Analysis...")
    analyzer = LayerCKAAnalyzer(wm)
    # Target encoder provides the representation target, but the predictor layers are what converge
    # For standard I-JEPA convergence, we can evaluate how context encoder blocks converge
    result = analyzer.analyze_layers(
        batch,
        layer_hook_pattern="context_encoder.blocks.{}.hook_resid_post"
    )

    print("Generating report...")
    fig = analyzer.plot_convergence(result)
    plot_path = os.path.join(args.output_dir, "layer_cka_convergence.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "semantic_convergence_score": float(result.semantic_convergence_score),
        "initial_layer_cka": float(result.avg_cka_per_layer[0]),
        "final_layer_cka": float(result.avg_cka_per_layer[-1])
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    report_md = f"""# Layer-wise Predictor Convergence Report

## Summary
* **Weights:** `{args.weights}`
* **Semantic Convergence Score:** `{metrics['semantic_convergence_score']:.3f}`
* **Initial Layer CKA:** `{metrics['initial_layer_cka']:.3f}`
* **Final Layer CKA:** `{metrics['final_layer_cka']:.3f}`

## Convergence Curve
![Convergence Curve](layer_cka_convergence.png)

*A positive convergence score indicates that patch representations are successfully converging to stable semantic meaning across transformer layers.*
"""
    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write(report_md)

    print(f"Done! Report saved to {args.output_dir}/report.md")

if __name__ == "__main__":
    main()
