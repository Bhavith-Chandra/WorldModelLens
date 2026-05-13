#!/usr/bin/env python3
"""Example: Layer-by-Layer CKA Analysis for Vision Models.

This example demonstrates how to analyze patch representation convergence
through transformer layers using Centered Kernel Alignment (CKA).

Shows how patch embeddings evolve from low-level features to semantic meaning.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.analysis.layer_cka import LayerCKAAnalyzer
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.types import WorldModelFamily


def create_sample_images(batch_size: int = 4, img_size: int = 224) -> torch.Tensor:
    """Create sample images for analysis."""
    # Create some simple synthetic images with patterns
    images = []

    for i in range(batch_size):
        if i == 0:
            # Vertical stripes
            img = torch.zeros(3, img_size, img_size)
            img[:, :, ::16] = 1.0
        elif i == 1:
            # Horizontal stripes
            img = torch.zeros(3, img_size, img_size)
            img[:, ::16, :] = 1.0
        elif i == 2:
            # Checkerboard pattern
            img = torch.zeros(3, img_size, img_size)
            img[:, ::32, ::32] = 1.0
            img[:, 16::32, 16::32] = 1.0
        else:
            # Random noise
            img = torch.randn(3, img_size, img_size) * 0.1 + 0.5

        images.append(img.clamp(0, 1))

    return torch.stack(images)


def main():
    """Run layer-by-layer CKA analysis."""

    print("🔍 Layer-by-Layer CKA Analysis for Vision Models")
    print("=" * 50)

    # Create a simple I-JEPA model for demonstration
    config = WorldModelConfig(
        world_model_family=WorldModelFamily.JEPA,
        embed_dim=192,
        n_heads=3,
        n_layers=6,
        patch_size=16,
        num_patches=196,  # 224//16 = 14, 14*14 = 196
        backend="ijepa",
        encoder_type="vit",
    )

    # Initialize model
    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter, config)

    print(f"Model: {config.world_model_family} with {config.n_layers} layers")
    print(f"Patch size: {config.patch_size}x{config.patch_size}")
    print(f"Number of patches: {config.num_patches}")

    # Create sample images
    print("\n📸 Creating sample images...")
    images = create_sample_images(batch_size=4)
    print(f"Input shape: {images.shape}")

    # Initialize analyzer
    analyzer = LayerCKAAnalyzer(wm)

    # Run layer-by-layer CKA analysis
    print("\n🧮 Computing layer-by-layer CKA...")
    result = analyzer.analyze_layers(
        images,
        layer_hook_pattern="context_encoder.blocks.{}.hook_resid_post",
        max_layers=None,  # Analyze all layers
    )

    print("Analysis Results:")
    print(f"- Layers analyzed: {len(result.layer_names)}")
    print(".3f")
    print(".3f")

    # Plot convergence analysis
    print("\n📊 Plotting convergence analysis...")
    fig = analyzer.plot_convergence(result)
    plt.savefig("layer_cka_convergence.png", dpi=300, bbox_inches="tight")
    print("Saved: layer_cka_convergence.png")

    # Plot patch embeddings PCA
    print("\n📈 Plotting patch embedding PCA...")
    fig_pca = analyzer.plot_patch_embeddings_pca(
        result, images, layer_hook_pattern="context_encoder.blocks.{}.hook_resid_post"
    )
    plt.savefig("patch_embeddings_pca.png", dpi=300, bbox_inches="tight")
    print("Saved: patch_embeddings_pca.png")

    print("\n✅ Analysis complete!")
    print("\nKey Insights:")
    print(".3f")
    print(".3f")
    print(".3f")
    if result.semantic_convergence_score > 0.5:
        print("🎯 Strong semantic convergence detected!")
    elif result.semantic_convergence_score > 0.3:
        print("📈 Moderate semantic convergence observed.")
    else:
        print("🤔 Limited semantic convergence - representations may not be converging.")

    # Show per-patch convergence statistics
    patch_conv = result.patch_convergence
    print("\nPatch Convergence Stats:")
    print(".3f")
    print(".3f")
    print(".3f")
    print(f"- Most converged patch: #{patch_conv.argmax()} (score: {patch_conv.max():.3f})")
    print(f"- Least converged patch: #{patch_conv.argmin()} (score: {patch_conv.min():.3f})")


if __name__ == "__main__":
    main()
