"""Layer-by-layer representation analysis for context encoders.

This module provides tools for analyzing how patch representations evolve
through transformer layers, showing convergence to semantic meaning using CKA.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from sklearn.decomposition import PCA

from world_model_lens.analysis.continual_learning import ContinualLearningAuditor


@dataclass
class LayerCKAResult:
    """Results of layer-by-layer CKA analysis."""

    layer_names: List[str]
    cka_matrix: np.ndarray  # [n_layers-1, n_patches] - CKA between consecutive layers
    avg_cka_per_layer: np.ndarray  # [n_layers-1] - average CKA across patches
    patch_convergence: np.ndarray  # [n_patches] - convergence score per patch
    semantic_convergence_score: float  # overall convergence metric


class LayerCKAAnalyzer:
    """Analyze layer-by-layer convergence of patch representations using CKA.

    This analyzer hooks into transformer layers to extract patch embeddings
    at each layer, then computes CKA similarity between consecutive layers
    to show how representations converge to semantic meaning.

    Example:
        analyzer = LayerCKAAnalyzer(world_model)

        # Run analysis on an image batch
        obs = torch.randn(4, 3, 224, 224)  # batch of images
        result = analyzer.analyze_layers(obs)

        # Plot convergence
        analyzer.plot_convergence(result)
    """

    def __init__(self, world_model: Any):
        """Initialize analyzer.

        Args:
            world_model: HookedWorldModel instance with vision encoder
        """
        self.wm = world_model

    def _centered_kernel_alignment(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute CKA (Centered Kernel Alignment)."""
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)

        n = a_flat.shape[0]

        # Center the matrices
        a_centered = a_flat - a_flat.mean(dim=0)
        b_centered = b_flat - b_flat.mean(dim=0)

        # Compute Gram matrices
        ka = a_centered @ a_centered.T / n
        kb = b_centered @ b_centered.T / n

        # Compute HSIC
        hsic_aa = (ka * ka).sum() / (n * n)
        hsic_bb = (kb * kb).sum() / (n * n)
        hsic_ab = (ka * kb).sum() / (n * n)

        # CKA
        cka = hsic_ab / (torch.sqrt(hsic_aa * hsic_bb) + 1e-8)

        return cka.item()

    def analyze_layers(
        self,
        observations: torch.Tensor,
        layer_hook_pattern: str = "target_encoder.blocks.{}.hook_resid_post",
        max_layers: Optional[int] = None,
    ) -> LayerCKAResult:
        """Analyze CKA between consecutive transformer layers.

        Args:
            observations: Input images [B, C, H, W]
            layer_hook_pattern: Pattern for layer hook names (e.g., "target_encoder.blocks.{}.hook_resid_post")
            max_layers: Maximum number of layers to analyze (None = all)

        Returns:
            LayerCKAResult with CKA analysis
        """
        # Extract representations from all layers
        layer_representations = self._extract_layer_representations(
            observations, layer_hook_pattern, max_layers
        )

        if len(layer_representations) < 2:
            raise ValueError("Need at least 2 layers for CKA analysis")

        layer_names = list(layer_representations.keys())
        representations = list(layer_representations.values())

        # Compute CKA between consecutive layers for each patch
        n_layers = len(representations)
        n_patches = representations[0].shape[1]  # [B, n_patches, dim]

        cka_matrix = np.zeros((n_layers - 1, n_patches))

        for layer_idx in range(n_layers - 1):
            layer_a = representations[layer_idx]  # [B, n_patches, dim]
            layer_b = representations[layer_idx + 1]  # [B, n_patches, dim]

            # Compute CKA for each patch independently
            for patch_idx in range(n_patches):
                patch_a = layer_a[:, patch_idx, :]  # [B, dim]
                patch_b = layer_b[:, patch_idx, :]  # [B, dim]

                cka = self._centered_kernel_alignment(patch_a, patch_b)
                cka_matrix[layer_idx, patch_idx] = cka

        # Average CKA per layer transition
        avg_cka_per_layer = cka_matrix.mean(axis=1)

        # Compute patch convergence scores (how much each patch's representation stabilizes)
        patch_convergence = self._compute_patch_convergence(cka_matrix)

        # Overall semantic convergence (average CKA increase)
        semantic_convergence_score = self._compute_semantic_convergence(avg_cka_per_layer)

        return LayerCKAResult(
            layer_names=layer_names,
            cka_matrix=cka_matrix,
            avg_cka_per_layer=avg_cka_per_layer,
            patch_convergence=patch_convergence,
            semantic_convergence_score=semantic_convergence_score,
        )

    def _extract_layer_representations(
        self,
        observations: torch.Tensor,
        layer_hook_pattern: str,
        max_layers: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        """Extract patch representations from each layer using hooks.

        Args:
            observations: Input images [B, C, H, W]
            layer_hook_pattern: Hook pattern with {} for layer index
            max_layers: Max layers to extract

        Returns:
            Dict mapping layer names to representations [B, n_patches, dim]
        """
        # Determine number of layers available
        if hasattr(self.wm.adapter, "context_encoder"):
            encoder = self.wm.adapter.context_encoder
        elif hasattr(self.wm.adapter, "encoder"):
            encoder = self.wm.adapter.encoder
        else:
            raise ValueError("Could not find encoder in world model adapter")

        if hasattr(encoder, "blocks"):
            if encoder.blocks is None:
                raise ValueError("Encoder does not have transformer blocks")
            n_layers = len(encoder.blocks)
        else:
            raise ValueError("Encoder does not have transformer blocks")

        if max_layers is not None:
            n_layers = min(n_layers, max_layers)

        # Build filter for encoder block names
        block_hook_patterns = []
        for i in range(n_layers):
            hook_name = layer_hook_pattern.format(i)
            block_hook_patterns.append(hook_name)

        # Run forward pass with caching
        with torch.no_grad():
            traj, cache = self.wm.run_with_cache(observations, names_filter=block_hook_patterns)

        # Extract layer outputs from cache
        layer_outputs = {}
        for i in range(n_layers):
            hook_name = layer_hook_pattern.format(i)
            if (hook_name, 0) in cache:
                layer_outputs[f"layer_{i}"] = cache[hook_name, 0]

        return layer_outputs

    def _compute_patch_convergence(self, cka_matrix: np.ndarray) -> np.ndarray:
        """Compute convergence score for each patch.

        Higher scores indicate patches that converge more (higher CKA between layers).

        Args:
            cka_matrix: [n_transitions, n_patches] CKA values

        Returns:
            Convergence scores [n_patches]
        """
        # Average CKA across layer transitions for each patch
        return cka_matrix.mean(axis=0)

    def _compute_semantic_convergence(self, avg_cka_per_layer: np.ndarray) -> float:
        """Compute overall semantic convergence score.

        Measures how much representations become more similar as they progress through layers.

        Args:
            avg_cka_per_layer: Average CKA per layer transition

        Returns:
            Convergence score (higher = more convergence)
        """
        if len(avg_cka_per_layer) < 2:
            return 0.0

        # Measure the trend: are CKA values increasing?
        # Positive slope indicates increasing similarity (convergence)
        x = np.arange(len(avg_cka_per_layer))
        slope = np.polyfit(x, avg_cka_per_layer, 1)[0]

        # Normalize to [0, 1] range
        return max(0.0, min(1.0, slope + 0.5))  # assuming slope range [-0.5, 0.5]

    def plot_convergence(
        self,
        result: LayerCKAResult,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot layer-by-layer convergence analysis.

        Args:
            result: LayerCKAResult from analyze_layers
            figsize: Figure size
            save_path: Optional path to save plot

        Returns:
            matplotlib Figure or None if matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
            return None
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Average CKA per layer transition
        layer_transitions = [
            "{}→{}".format(result.layer_names[i], result.layer_names[i + 1])
            for i in range(len(result.layer_names) - 1)
        ]

        axes[0, 0].plot(result.avg_cka_per_layer, "bo-", linewidth=2, markersize=6)
        axes[0, 0].set_xticks(range(len(layer_transitions)))
        axes[0, 0].set_xticklabels(layer_transitions, rotation=45, ha="right")
        axes[0, 0].set_ylabel("Average CKA")
        axes[0, 0].set_title("Layer-by-Layer Representation Similarity")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: CKA matrix heatmap
        im = axes[0, 1].imshow(result.cka_matrix.T, aspect="auto", cmap="viridis")
        axes[0, 1].set_xticks(range(len(layer_transitions)))
        axes[0, 1].set_xticklabels(layer_transitions, rotation=45, ha="right")
        axes[0, 1].set_ylabel("Patch Index")
        axes[0, 1].set_title("CKA Matrix (Patches × Layers)")
        plt.colorbar(im, ax=axes[0, 1])

        # Plot 3: Patch convergence distribution
        axes[1, 0].hist(result.patch_convergence, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(
            result.patch_convergence.mean(),
            color="red",
            linestyle="--",
            label="Mean: {:.3f}".format(result.patch_convergence.mean()),
        )
        axes[1, 0].set_xlabel("Convergence Score")
        axes[1, 0].set_ylabel("Number of Patches")
        axes[1, 0].set_title("Patch Convergence Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Convergence trend analysis
        axes[1, 1].text(
            0.1,
            0.8,
            "{:.3f}".format(result.semantic_convergence_score),
            fontsize=14,
            fontweight="bold",
        )
        axes[1, 1].text(
            0.1, 0.6, "Final CKA: {:.3f}".format(result.avg_cka_per_layer[-1]), fontsize=12
        )
        axes[1, 1].text(
            0.1,
            0.4,
            "CKA Range: {:.3f} - {:.3f}".format(
                result.avg_cka_per_layer.min(), result.avg_cka_per_layer.max()
            ),
            fontsize=12,
        )

        # Add convergence arrow
        if result.avg_cka_per_layer[-1] > result.avg_cka_per_layer[0]:
            axes[1, 1].text(0.1, 0.2, "↗ Representations Converging", fontsize=12, color="green")
        else:
            axes[1, 1].text(0.1, 0.2, "↘ Representations Diverging", fontsize=12, color="red")

        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title("Semantic Convergence Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_patch_embeddings_pca(
        self,
        result: LayerCKAResult,
        observations: torch.Tensor,
        layer_hook_pattern: str = "target_encoder.blocks.{}.hook_resid_post",
        n_components: int = 2,
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot PCA of patch embeddings to visualize semantic convergence.

        Args:
            result: LayerCKAResult from analyze_layers
            observations: Original input images
            layer_hook_pattern: Hook pattern for layer extraction
            n_components: PCA components (2 or 3)
            save_path: Optional save path

        Returns:
            matplotlib Figure or None if matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
            return None
        # Extract representations from first and last layers
        layer_reps = self._extract_layer_representations(
            observations, layer_hook_pattern, max_layers=None
        )

        if len(layer_reps) == 0:
            raise ValueError("No layer representations extracted. Check layer_hook_pattern.")

        first_layer = list(layer_reps.values())[0]  # [B, n_patches, dim]
        last_layer = list(layer_reps.values())[-1]  # [B, n_patches, dim]

        # Flatten batch and patches for PCA
        first_flat = first_layer.reshape(-1, first_layer.shape[-1]).cpu().numpy()
        last_flat = last_layer.reshape(-1, last_layer.shape[-1]).cpu().numpy()

        # Fit PCA on first layer, transform both
        pca = PCA(n_components=n_components)
        pca.fit(first_flat)

        first_pca = pca.transform(first_flat)
        last_pca = pca.transform(last_flat)

        # Plot
        if n_components == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # First layer
            ax1.scatter(first_pca[:, 0], first_pca[:, 1], alpha=0.6, c="blue", label="First Layer")
            ax1.set_title("Patch Embeddings - First Layer")
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Last layer
            ax2.scatter(last_pca[:, 0], last_pca[:, 1], alpha=0.6, c="red", label="Last Layer")
            ax2.set_title("Patch Embeddings - Last Layer")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Set same axis limits
            xlims = (
                min(ax1.get_xlim()[0], ax2.get_xlim()[0]),
                max(ax1.get_xlim()[1], ax2.get_xlim()[1]),
            )
            ylims = (
                min(ax1.get_ylim()[0], ax2.get_ylim()[0]),
                max(ax1.get_ylim()[1], ax2.get_ylim()[1]),
            )

            ax1.set_xlim(xlims)
            ax2.set_xlim(xlims)
            ax1.set_ylim(ylims)
            ax2.set_ylim(ylims)

        elif n_components == 3:
            fig = plt.figure(figsize=(16, 6))

            # First layer
            ax1 = fig.add_subplot(121, projection="3d")
            ax1.scatter(
                first_pca[:, 0],
                first_pca[:, 1],
                first_pca[:, 2],
                alpha=0.6,
                c="blue",
                label="First Layer",
            )
            ax1.set_title("Patch Embeddings - First Layer")
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            ax1.set_zlabel("PC3")
            ax1.legend()

            # Last layer
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.scatter(
                last_pca[:, 0],
                last_pca[:, 1],
                last_pca[:, 2],
                alpha=0.6,
                c="red",
                label="Last Layer",
            )
            ax2.set_title("Patch Embeddings - Last Layer")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_zlabel("PC3")
            ax2.legend()

        else:
            raise ValueError("n_components must be 2 or 3, got {}".format(n_components))

        plt.suptitle(
            "Patch Embedding Convergence (CKA Score: {:.3f})".format(
                result.semantic_convergence_score
            )
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
