"""Patch attribution and causal sensitivity evaluation for world models.

This module provides gradient-based attribution methods (Integrated Gradients,
Gradient x Input, SmoothGrad) to score the causal importance of context patches
for predicting a specific target. It also provides an AttributionEvaluator to
compare these causal scores against attention weights and compute aggregate
metrics across datasets.

Note:
    This module focuses strictly on *attribution methods*. Causal evaluation
    components (e.g., ablation, counterfactual interventions) will be introduced
    in a subsequent module.

    SHAP is intentionally excluded due to computational constraints at patch-level
    resolution; current baselines represent a standard gradient-based attribution
    triad.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


class BaseAttribution(ABC):
    """Abstract base class for patch attribution methods.

    All attribution methods must implement the compute() method to return
    scalar attribution scores for a set of context patches with respect
    to a specific target.
    """

    def __init__(self, adapter: Any):
        """Initialize the attribution method.

        Args:
            adapter: Loaded World Model adapter (e.g., IJEPAAdapter). Must be
                     in evaluation mode to prevent stochasticity.
        """
        self.adapter = adapter
        self.adapter.eval()

    @abstractmethod
    def compute(
        self,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
    ) -> np.ndarray:
        """Compute attribution scores for each context patch.

        Args:
            img_tensor: Input image tensor of shape [1, C, H, W].
            context_ids: List of integer patch indices used as context.
            target_id: Integer patch index of the target to predict.

        Returns:
            np.ndarray: Scalar attribution scores of shape [len(context_ids)].
                        Higher absolute values indicate higher importance.
        """
        pass

    def _forward_score(
        self,
        patch_emb: torch.Tensor,
        context_ids: List[int],
        target_id: int,
        target_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scalar prediction score given pre-positional embeddings.

        Args:
            patch_emb: [1, N_all, C] raw patch embeddings (no positional encoding).
            context_ids: List of context patch indices.
            target_id: Target patch index.
            target_gt: [1, C] ground truth target embedding.

        Returns:
            torch.Tensor: Scalar score (negative MSE). Higher is better.
        """
        device = patch_emb.device
        ctx_ids_t = torch.tensor(context_ids, device=device)
        
        # Select context patches and add positional embeddings
        ctx_emb = patch_emb[:, ctx_ids_t, :]
        pos = self.adapter.context_encoder.pos_embed[:, ctx_ids_t, :]
        ctx_with_pos = ctx_emb + pos

        # Process through blocks
        ctx_latents = self.adapter.context_encoder.forward_blocks(ctx_with_pos)

        # Predict target
        pred = self.adapter.predictor(ctx_latents, context_ids, [target_id])

        # Score is negative MSE
        return -F.mse_loss(pred.squeeze(1), target_gt)

    @torch.no_grad()
    def _get_target_gt(self, img_tensor: torch.Tensor, target_id: int) -> torch.Tensor:
        """Get ground truth target embedding from the target encoder."""
        target_reps = self.adapter.target_encoder(img_tensor)
        return target_reps[:, [target_id], :]


class IntegratedGradientsAttribution(BaseAttribution):
    """Integrated Gradients (IG) attribution method.

    Computes the path integral of gradients from a baseline embedding
    (on-manifold mean) to the actual patch embeddings.
    """

    def __init__(self, adapter: Any, n_steps: int = 50):
        """Initialize IG attribution.

        Args:
            adapter: Loaded World Model adapter.
            n_steps: Number of Riemann sum steps.
        """
        super().__init__(adapter)
        self.n_steps = n_steps

    def compute(
        self,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
    ) -> np.ndarray:
        img_tensor = img_tensor.float().to(next(self.adapter.parameters()).device)

        with torch.no_grad():
            actual_emb = self.adapter.context_encoder.patch_embed(img_tensor)
            baseline_emb = actual_emb.mean(dim=1, keepdim=True).expand_as(actual_emb)

        target_gt = self._get_target_gt(img_tensor, target_id)
        target_gt_flat = target_gt.squeeze(0)

        accumulated_grads = torch.zeros_like(actual_emb)

        for step in range(self.n_steps):
            alpha = step / self.n_steps
            interp_emb = (baseline_emb + alpha * (actual_emb - baseline_emb)).detach()
            interp_emb.requires_grad_(True)

            score = self._forward_score(interp_emb, context_ids, target_id, target_gt_flat)
            score.backward()

            if interp_emb.grad is not None:
                accumulated_grads += interp_emb.grad.detach()
            
            interp_emb.grad = None

        delta = actual_emb.detach() - baseline_emb.detach()
        ig_per_dim = delta * (accumulated_grads / self.n_steps)
        ig_per_patch = ig_per_dim.abs().sum(dim=-1).squeeze(0)

        return ig_per_patch[context_ids].cpu().numpy()


class GradientXInputAttribution(BaseAttribution):
    """Gradient * Input attribution method.

    A fast baseline computing the element-wise product of gradients and inputs.
    """

    def compute(
        self,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
    ) -> np.ndarray:
        img_tensor = img_tensor.float().to(next(self.adapter.parameters()).device)

        with torch.no_grad():
            actual_emb = self.adapter.context_encoder.patch_embed(img_tensor)
        
        target_gt = self._get_target_gt(img_tensor, target_id)
        target_gt_flat = target_gt.squeeze(0)

        emb_to_grad = actual_emb.detach().clone()
        emb_to_grad.requires_grad_(True)

        score = self._forward_score(emb_to_grad, context_ids, target_id, target_gt_flat)
        score.backward()

        grad = emb_to_grad.grad.detach()
        grad_x_input = (grad * actual_emb.detach()).abs().sum(dim=-1).squeeze(0)

        return grad_x_input[context_ids].cpu().numpy()


class SmoothGradAttribution(BaseAttribution):
    """SmoothGrad attribution method.

    Averages the Gradient * Input results over multiple noisy inputs to
    reduce local gradient noise and improve attribution clarity.
    """

    def __init__(self, adapter: Any, n_samples: int = 50, noise_level: float = 0.1):
        """Initialize SmoothGrad.

        Args:
            adapter: Loaded World Model adapter.
            n_samples: Number of noisy samples to average.
            noise_level: Standard deviation of the Gaussian noise as a fraction
                         of the input span.
        """
        super().__init__(adapter)
        self.n_samples = n_samples
        self.noise_level = noise_level

    def compute(
        self,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
    ) -> np.ndarray:
        img_tensor = img_tensor.float().to(next(self.adapter.parameters()).device)

        with torch.no_grad():
            actual_emb = self.adapter.context_encoder.patch_embed(img_tensor)
        
        target_gt = self._get_target_gt(img_tensor, target_id)
        target_gt_flat = target_gt.squeeze(0)

        std_dev = self.noise_level * (actual_emb.max() - actual_emb.min()).item()
        accumulated_grads = torch.zeros_like(actual_emb)

        for _ in range(self.n_samples):
            noise = torch.randn_like(actual_emb) * std_dev
            noisy_emb = (actual_emb + noise).detach()
            noisy_emb.requires_grad_(True)

            score = self._forward_score(noisy_emb, context_ids, target_id, target_gt_flat)
            score.backward()

            if noisy_emb.grad is not None:
                accumulated_grads += noisy_emb.grad.detach()
            
            noisy_emb.grad = None

        avg_grad = accumulated_grads / self.n_samples
        smooth_grad_x_input = (avg_grad * actual_emb.detach()).abs().sum(dim=-1).squeeze(0)

        return smooth_grad_x_input[context_ids].cpu().numpy()


@torch.no_grad()
def extract_attention_weights(
    wm: Any, 
    img_tensor: torch.Tensor, 
    context_ids: List[int], 
    target_id: int, 
    layer_idx: int = -1,
    head_idx: Optional[Union[int, str]] = None
) -> np.ndarray:
    """Helper to extract attention weights for a given target.
    
    Args:
        wm: HookedWorldModel instance.
        img_tensor: [1, 3, H, W] image.
        context_ids: List of context patch indices.
        target_id: Target patch index.
        layer_idx: Index of the predictor block to extract attention from.
        head_idx: If int, extract that head. If None, average across heads.
                  If "all", return all heads as [n_heads, n_ctx].
    """
    wm.adapter.last_context_ids = context_ids
    wm.adapter.last_target_ids = [target_id]
    
    wm.run_with_cache(img_tensor)
    
    # Get the specific block
    n_blocks = len(wm.adapter.predictor.blocks)
    actual_layer = layer_idx if layer_idx >= 0 else n_blocks + layer_idx
    attn = wm.adapter.predictor.blocks[actual_layer].attn.last_attn_weights # [1, n_heads, n_queries, n_keys]
    
    # Target patch is always at the end of queries
    # Keys correspond to [context_patches, target_patches]
    n_ctx = len(context_ids)
    
    if head_idx is None:
        # Average across heads
        target_to_context_attn = attn[0].mean(0)[-1, :n_ctx].cpu().numpy()
    elif head_idx == "all":
        # Return all heads
        target_to_context_attn = attn[0][:, -1, :n_ctx].cpu().numpy() # [n_heads, n_ctx]
    else:
        # Extract specific head
        target_to_context_attn = attn[0][head_idx, -1, :n_ctx].cpu().numpy()
        
    return target_to_context_attn


class AttributionEvaluator:
    """Evaluates attribution scores against attention weights.

    Demonstrates that attention-based interpretations can be systematically
    unfaithful in a measurable subset of cases. Computes dataset-level
    aggregate metrics including Jaccard overlap and Spearman rank correlation.
    """

    def __init__(self, k: int = 6):
        """Initialize the evaluator.

        Args:
            k: The number of top patches to consider for Jaccard overlap.
        """
        self.k = k

    def compute_jaccard_overlap(self, attn_weights: np.ndarray, attr_scores: np.ndarray) -> float:
        """Compute the Jaccard similarity between top-K ranked patches.
        
        Args:
            attn_weights: Array of attention weights.
            attr_scores: Array of causal attribution scores.

        Returns:
            float: Overlap score between 0.0 and 1.0.
        """
        if self.k > len(attn_weights):
            k = len(attn_weights)
        else:
            k = self.k

        top_attn_idx = np.argsort(attn_weights)[::-1][:k]
        top_attr_idx = np.argsort(attr_scores)[::-1][:k]

        intersection = len(set(top_attn_idx) & set(top_attr_idx))
        return intersection / k

    def compute_rank_correlation(self, attn_weights: np.ndarray, attr_scores: np.ndarray) -> float:
        """Compute Spearman rank correlation between attention and attribution.

        Args:
            attn_weights: Array of attention weights.
            attr_scores: Array of causal attribution scores.

        Returns:
            float: Spearman correlation coefficient [-1.0, 1.0].
        """
        if spearmanr is None:
            raise ImportError("scipy is required for Spearman correlation. Install with `pip install scipy`.")
        
        # We want to measure how well attention ranks correlate with attribution ranks
        corr, _ = spearmanr(attn_weights, attr_scores)
        return float(corr) if not np.isnan(corr) else 0.0

    def evaluate_sample_heads(
        self,
        wm: Any,
        attribution_method: BaseAttribution,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_id: int,
        layer_idx: int = -1,
    ) -> Dict[str, Any]:
        """Compute metrics for every individual head in a given layer.
        
        Args:
            wm: HookedWorldModel instance.
            attribution_method: BaseAttribution instance.
            img_tensor: Input image.
            context_ids: Context patches.
            target_id: Target patch.
            layer_idx: Layer to extract attention from.
            
        Returns:
            Dictionary with per-head metrics and their stats across heads.
        """
        # 1. Compute causal attribution (once per sample)
        attr_scores = attribution_method.compute(img_tensor, context_ids, target_id)
        
        # 2. Extract all heads for this layer [n_heads, n_ctx]
        all_heads_attn = extract_attention_weights(
            wm, img_tensor, context_ids, target_id, layer_idx=layer_idx, head_idx="all"
        )
        
        n_heads = all_heads_attn.shape[0]
        head_overlaps = []
        head_correlations = []
        
        for i in range(n_heads):
            attn_weights = all_heads_attn[i]
            head_overlaps.append(self.compute_jaccard_overlap(attn_weights, attr_scores))
            head_correlations.append(self.compute_rank_correlation(attn_weights, attr_scores))
            
        head_overlaps = np.array(head_overlaps)
        head_correlations = np.array(head_correlations)
        
        # Also compute for the head-averaged attention map (standard metric)
        avg_attn = np.mean(all_heads_attn, axis=0)
        avg_overlap = self.compute_jaccard_overlap(avg_attn, attr_scores)
        avg_corr = self.compute_rank_correlation(avg_attn, attr_scores)
        
        return {
            "head_overlaps": head_overlaps,
            "head_correlations": head_correlations,
            "mean_overlap_across_heads": float(np.mean(head_overlaps)),
            "var_overlap_across_heads": float(np.var(head_overlaps)),
            "mean_corr_across_heads": float(np.mean(head_correlations)),
            "var_corr_across_heads": float(np.var(head_correlations)),
            "averaged_map_overlap": avg_overlap,
            "averaged_map_corr": avg_corr,
            "attr_scores": attr_scores,
            "avg_attn": avg_attn
        }

    def evaluate_dataset(
        self,
        wm: Any,
        attribution_method: BaseAttribution,
        dataset: List[Tuple[torch.Tensor, List[int], int]],
        alignment_threshold: float = 0.7,
        failure_threshold: float = 0.3,
        layer_idx: int = -1,
    ) -> Dict[str, Any]:
        """Evaluate attribution against attention across a dataset.

        Args:
            wm: HookedWorldModel instance.
            attribution_method: Initialized BaseAttribution instance.
            dataset: List of (img_tensor, context_ids, target_id) tuples.
            alignment_threshold: Overlap >= this is considered high alignment.
            failure_threshold: Overlap <= this is considered near-zero/failure.
            layer_idx: The layer to extract attention from.

        Returns:
            Dictionary containing metrics, raw scores, and confidence intervals.
        """
        overlaps = []
        correlations = []
        raw_attn = []
        raw_attr = []
        
        # Track per-head stats across the dataset
        all_sample_head_results = []

        n_samples = len(dataset)
        for i, (img_tensor, context_ids, target_id) in enumerate(dataset):
            if i % 5 == 0:
                print(f"  [Progress] Evaluating sample {i}/{n_samples}...")
            # We use the head-averaged map for the main dataset metrics
            head_results = self.evaluate_sample_heads(
                wm, attribution_method, img_tensor, context_ids, target_id, layer_idx=layer_idx
            )
            
            overlap = head_results["averaged_map_overlap"]
            corr = head_results["averaged_map_corr"]
            
            overlaps.append(overlap)
            correlations.append(corr)
            
            raw_attn.append(head_results["avg_attn"])
            raw_attr.append(head_results["attr_scores"])
            all_sample_head_results.append(head_results)

        overlaps = np.array(overlaps)
        correlations = np.array(correlations)
        # n_samples defined above

        failures = np.mean(overlaps <= failure_threshold)
        alignments = np.mean(overlaps >= alignment_threshold)
        low_overlap = np.mean(overlaps < 0.1)
        negative_spearman = np.mean(correlations < 0)

        # 95% Confidence Intervals
        ci_overlap = 1.96 * np.std(overlaps) / np.sqrt(n_samples) if n_samples > 0 else 0.0
        ci_spearman = 1.96 * np.std(correlations) / np.sqrt(n_samples) if n_samples > 0 else 0.0

        # Aggregate per-head stats across dataset
        mean_overlap_heads = np.mean([r["mean_overlap_across_heads"] for r in all_sample_head_results])
        mean_var_overlap_heads = np.mean([r["var_overlap_across_heads"] for r in all_sample_head_results])
        mean_corr_heads = np.mean([r["mean_corr_across_heads"] for r in all_sample_head_results])
        mean_var_corr_heads = np.mean([r["var_corr_across_heads"] for r in all_sample_head_results])

        return {
            "n_samples": n_samples,
            "mean_overlap": float(np.mean(overlaps)),
            "var_overlap": float(np.var(overlaps)),
            "ci_overlap": float(ci_overlap),
            "mean_spearman": float(np.mean(correlations)),
            "var_spearman": float(np.var(correlations)),
            "ci_spearman": float(ci_spearman),
            "failure_rate": float(failures),
            "low_overlap_rate": float(low_overlap),
            "alignment_rate": float(alignments),
            "negative_spearman_rate": float(negative_spearman),
            "raw_overlaps": overlaps,
            "raw_correlations": correlations,
            "raw_attn_list": raw_attn,
            "raw_attr_list": raw_attr,
            # Per-head stats
            "mean_overlap_across_heads": mean_overlap_heads,
            "mean_var_overlap_heads": mean_var_overlap_heads,
            "mean_corr_across_heads": mean_corr_heads,
            "mean_var_corr_heads": mean_var_corr_heads,
        }

    def plot_overlap_distributions(
        self,
        results: Dict[str, Any],
        title: str = "Jaccard Overlap Distribution ($O_k$)",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot the distribution of Jaccard overlap scores."""
        if plt is None:
            print("matplotlib is not installed. Skipping plot.")
            return None

        overlaps = results["raw_overlaps"]
        mean_o = results["mean_overlap"]
        var_o = results["var_overlap"]
        n_samples = results["n_samples"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(overlaps, bins=np.linspace(0, 1, self.k + 2), alpha=0.7, color="blue", edgecolor="black")
        ax.axvline(mean_o, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_o:.2f}")
        
        ax.set_title(f"{title}\nN={n_samples} | Variance={var_o:.4f}")
        ax.set_xlabel(f"Overlap Score (Top-{self.k})")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig

    def plot_rank_correlation_distribution(
        self,
        results: Dict[str, Any],
        title: str = "Spearman Rank Correlation Distribution",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot the distribution of Spearman rank correlations."""
        if plt is None:
            return None

        corrs = results["raw_correlations"]
        mean_corr = results["mean_spearman"]
        n_samples = results["n_samples"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(corrs, bins=20, range=(-1, 1), alpha=0.7, color="green", edgecolor="black")
        ax.axvline(mean_corr, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_corr:.2f}")
        
        ax.set_title(f"{title}\nN={n_samples} | Variance={results['var_spearman']:.4f}")
        ax.set_xlabel("Spearman Rank Correlation")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-1.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig

    def plot_rank_scatter(
        self,
        attn_weights: np.ndarray,
        attr_scores: np.ndarray,
        title: str = "Attention Rank vs Causal Attribution Rank",
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot a scatter plot showing attention rank vs IG rank.
        
        This explicitly visualizes ranking inversion/decorrelation, highlighting
        when high-attention patches have low actual causal importance.
        """
        if plt is None:
            return None

        # Convert scores to ranks (1 is best/highest, N is worst/lowest)
        # argsort() twice gives the rank.
        # Since we want highest score = rank 1, we invert before ranking
        attn_ranks = len(attn_weights) - np.argsort(np.argsort(attn_weights))
        attr_ranks = len(attr_scores) - np.argsort(np.argsort(attr_scores))

        corr = self.compute_rank_correlation(attn_weights, attr_scores)

        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Plot ideal diagonal
        max_rank = len(attn_weights)
        ax.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.3, label="Perfect Alignment")
        
        # Scatter points
        ax.scatter(attn_ranks, attr_ranks, alpha=0.7, c='purple', s=40, edgecolors='white')
        
        # Formatting
        ax.set_title(f"{title}\nSpearman Correlation: {corr:.3f}")
        ax.set_xlabel("Attention Rank (1 = Highest Weight)")
        ax.set_ylabel("Attribution Rank (1 = Highest Causal Impact)")
        
        # Invert axes so rank 1 is at the top/right (or bottom/left depending on preference, 
        # usually origin is rank 1)
        ax.set_xlim(max_rank + 1, 0)
        ax.set_ylim(max_rank + 1, 0)
        
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
