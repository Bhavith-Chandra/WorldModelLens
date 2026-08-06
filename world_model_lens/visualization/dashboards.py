"""Reusable dashboard layout functions for WorldModelLens examples.

Each function assembles a matplotlib figure from analysis results and
optionally saves it to disk.  The examples call these instead of inlining
matplotlib code.
"""

from typing import Any, Dict, List, Optional
import pathlib

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from world_model_lens.visualization.cache_plots import CacheSignalPlotter
from world_model_lens.visualization.intervention_plots import InterventionVisualizer
from world_model_lens.visualization.prediction_plots import PredictionVisualizer


# Quickstart

def plot_quickstart_dashboard(
    traj: Any,
    cache: Any,
    imagined_traj: Any,
    n_cat: int,
    n_cls: int,
    wm: Any,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Six-panel overview: hidden-state PCA, surprise timeline, per-category
    heatmap, activation/head timelines, 3D state trajectory, counterfactual
    divergence heatmap.

    Args:
        traj: LatentTrajectory from run_with_cache.
        cache: ActivationCache from run_with_cache.
        imagined_traj: LatentTrajectory from wm.imagine().
        n_cat: Number of latent categories (from config).
        n_cls: Number of classes per category (from config).
        wm: HookedWorldModel instance (used for InterventionVisualizer).
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    T = traj.length

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("WorldModelLens Quickstart Dashboard", fontsize=14)

    # 1. Hidden-state PCA
    h_tensors = torch.stack([cache["h", t] for t in range(T)])
    h_centered = h_tensors - h_tensors.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(h_centered, q=2)
    pca_proj = (h_centered @ Vt).detach().numpy()

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(pca_proj[:, 0], pca_proj[:, 1], c=np.arange(T), cmap="viridis")
    for i in range(T):
        ax1.annotate(str(i), (pca_proj[i, 0], pca_proj[i, 1]), fontsize=7)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Hidden-State PCA")

    # 2. Surprise Timeline
    kl_vals = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_vals.append((p * (p.log() - q.log())).sum().item())

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(range(T), kl_vals, marker="o")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("KL")
    ax2.set_title("Surprise Timeline")

    # 3. Per-Category Surprise Heatmap
    kl_per_cat = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_per_cat.append((p * (p.log() - q.log())).sum(dim=-1).detach().numpy())
    heatmap_matrix = np.stack(kl_per_cat, axis=0)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(heatmap_matrix.T, aspect="auto", origin="lower", cmap="hot")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Latent Category")
    ax3.set_title("Per-Category Surprise Heatmap")

    # 4. Activation / Head Timelines
    h_signal = CacheSignalPlotter.plot_cache_signal(cache, "h")
    reward_data = CacheSignalPlotter.plot_reward_timeline(traj)
    value_data = CacheSignalPlotter.plot_value_timeline(traj)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(h_signal["timesteps"], h_signal["norms"], label="||h_t||")
    if len(reward_data["predicted"]) > 0:
        ax4.plot(reward_data["timesteps"], reward_data["predicted"], label="Reward Pred")
    if len(value_data["value_pred"]) > 0:
        ax4.plot(value_data["timesteps"], value_data["value_pred"], label="Value Pred")
    ax4.set_xlabel("Timestep")
    ax4.legend(fontsize=7)
    ax4.set_title("Activation / Head Timelines")

    # 5. State Trajectory (3D)
    pred_viz = PredictionVisualizer(wm)
    traj_3d = pred_viz.state_trajectory_3d(traj, components=[0, 1, 2])
    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    if len(traj_3d["x"]) > 1:
        ax5.plot(traj_3d["x"], traj_3d["y"], traj_3d["z"], marker="o", markersize=3)
    ax5.set_xlabel("Dim 0")
    ax5.set_ylabel("Dim 1")
    ax5.set_zlabel("Dim 2")
    ax5.set_title("State Trajectory (3D)")

    # 6. Counterfactual Divergence heatmap
    iv = InterventionVisualizer(wm)
    heatmap = iv.intervention_heatmap(traj, imagined_traj)
    divergence_curve = iv.divergence_curve(traj, imagined_traj)
    ts = sorted(divergence_curve.keys())
    divs = [divergence_curve[t] for t in ts]
    cf_matrix = np.zeros((heatmap.shape[1], len(ts)))
    for i, t in enumerate(ts):
        if t < heatmap.shape[0]:
            cf_matrix[:, i] = heatmap[t, : cf_matrix.shape[0]]

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(cf_matrix, aspect="auto", cmap="RdBu_r", origin="lower")
    ax6.plot(
        range(len(ts)),
        [d * cf_matrix.shape[0] for d in divs],
        color="navy",
        label="Divergence",
    )
    ax6.legend(fontsize=7)
    ax6.set_xlabel("Timestep")
    ax6.set_ylabel("Dims / Divergence")
    ax6.set_title("Counterfactual Divergence")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Probing

def plot_probing_dashboard(
    sweep_result: Any,
    activations: torch.Tensor,
    labels: np.ndarray,
    concepts: Dict[str, np.ndarray],
    cache: Any,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Four-panel probing dashboard: PCA scatter, probe accuracy bars,
    hidden-state norms, label distribution.

    Args:
        sweep_result: ProbeSweepResult from LatentProber.sweep().
        activations: Collected z_posterior activations [N, d_z].
        labels: Integer concept labels per activation.
        concepts: Dict of concept_name → binary label array.
        cache: Last-run ActivationCache (for hidden state norms).
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WorldModelLens - Linear Probing Dashboard", fontsize=14)

    # 1. PCA of activations colored by label
    acts_centered = activations - activations.mean(0)
    _, _, Vt = torch.pca_lowrank(acts_centered, q=2)
    pca_proj = (acts_centered @ Vt).detach().numpy()

    ax = axes[0, 0]
    sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Label (0=reward, 1=novel, 2=value)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("z_posterior PCA (colored by concept label)")

    # 2. Probe accuracy per concept
    ax = axes[0, 1]
    concept_names = list(concepts.keys())
    accuracies = [sweep_result.results.get(f"{c}_z_posterior", None) for c in concept_names]
    acc_vals = [r.accuracy if r is not None else 0.0 for r in accuracies]
    bars = ax.bar(concept_names, acc_vals, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe Accuracy per Concept")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.legend(fontsize=8)
    for bar, acc in zip(bars, acc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Hidden state norms for last episode
    ax = axes[1, 0]
    h_signal = CacheSignalPlotter.plot_cache_signal(cache, "h")
    ax.plot(h_signal["timesteps"], h_signal["norms"], marker="o", color="darkorange")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Hidden State Norms (last episode)")

    # 4. Label distribution
    ax = axes[1, 1]
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "reward_region", 1: "novel_state", 2: "high_value"}
    ax.bar([label_names.get(int(u), str(u)) for u in unique], counts, color="mediumseagreen")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution (across all episodes)")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Activation Patching

def plot_patching_dashboard(
    wm: Any,
    sweep_result: Any,
    clean_traj: Any,
    corrupted_traj: Any,
    clean_cache: Any,
    corrupted_cache: Any,
    corruption_start_t: int = 5,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Three-panel patching dashboard: recovery rate heatmap, clean vs
    corrupted divergence curves, hidden-state norm comparison.

    Args:
        wm: HookedWorldModel instance.
        sweep_result: PatchSweepResult from TemporalPatcher.full_sweep().
        clean_traj: Baseline LatentTrajectory.
        corrupted_traj: Corrupted LatentTrajectory.
        clean_cache: Baseline ActivationCache.
        corrupted_cache: Corrupted ActivationCache.
        corruption_start_t: Timestep where corruption began (draws vertical line).
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WorldModelLens - Activation Patching Dashboard", fontsize=14)

    # 1. Recovery rate heatmap
    ax = axes[0]
    recovery_matrix = sweep_result.recovery_matrix().cpu().numpy()
    im = ax.imshow(recovery_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(sweep_result.components)))
    ax.set_yticklabels(sweep_result.components)
    ax.set_xticks(range(len(sweep_result.timesteps)))
    ax.set_xticklabels(sweep_result.timesteps)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Component")
    ax.set_title("Recovery Rate Heatmap")
    plt.colorbar(im, ax=ax, label="Recovery Rate")

    # 2. Clean vs corrupted divergence
    ax = axes[1]
    iv = InterventionVisualizer(wm)
    divergence_curve = iv.divergence_curve(clean_traj, corrupted_traj)
    ts = sorted(divergence_curve.keys())
    divs = [divergence_curve[t] for t in ts]
    cum_divs = []
    running = 0.0
    for d in divs:
        running += d
        cum_divs.append(running)
    ax.plot(ts, divs, marker="o", label="Step divergence", color="tomato")
    ax.plot(ts, cum_divs, marker="s", linestyle="--", label="Cumulative", color="navy")
    ax.axvline(corruption_start_t, color="gray", linestyle=":", linewidth=1.5, label="Corruption starts")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE Divergence")
    ax.set_title("Clean vs Corrupted Divergence")
    ax.legend(fontsize=8)

    # 3. Hidden state norms: clean vs corrupted
    ax = axes[2]
    h_clean = CacheSignalPlotter.plot_cache_signal(clean_cache, "h")
    h_corrupt = CacheSignalPlotter.plot_cache_signal(corrupted_cache, "h")
    ax.plot(h_clean["timesteps"], h_clean["norms"], marker="o", label="Clean", color="steelblue")
    ax.plot(
        h_corrupt["timesteps"],
        h_corrupt["norms"],
        marker="s",
        linestyle="--",
        label="Corrupted",
        color="darkorange",
    )
    ax.axvline(corruption_start_t, color="gray", linestyle=":", linewidth=1.5, label="Corruption starts")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Hidden State Norms: Clean vs Corrupted")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Imagination Branching

def plot_branching_dashboard(
    branches: List[Any],
    real_traj_cache: Any,
    fork_at: int,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Three-panel branching dashboard: per-branch divergence curves, PCA
    scatter of all branch states, real-trajectory hidden-state norms with
    fork marker.

    Args:
        branches: List of LatentTrajectory objects (imagined branches).
        real_traj_cache: ActivationCache from the real trajectory run.
        fork_at: Timestep where branches diverged from the real trajectory.
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WorldModelLens - Imagination Branching Dashboard", fontsize=14)

    # 1. Branch divergence curves vs branch 0
    ax = axes[0]
    ref_states = torch.stack([s.state for s in branches[0].states])
    colors = plt.cm.tab10(np.linspace(0, 1, len(branches) - 1))
    for i, branch in enumerate(branches[1:], 1):
        branch_states = torch.stack([s.state for s in branch.states])
        min_len = min(len(ref_states), len(branch_states))
        div = (ref_states[:min_len] - branch_states[:min_len]).norm(dim=-1).detach().numpy()
        ax.plot(div, label=f"Branch 0 vs {i}", color=colors[i - 1])
    ax.set_xlabel("Timestep (post-fork)")
    ax.set_ylabel("L2 Divergence")
    ax.set_title("Branch Divergence from Reference")
    ax.legend(fontsize=8)

    # 2. PCA of all branch states
    ax = axes[1]
    all_states = []
    branch_labels = []
    for i, branch in enumerate(branches):
        for s in branch.states:
            all_states.append(s.state.flatten())
            branch_labels.append(i)
    all_states_t = torch.stack(all_states)
    centered = all_states_t - all_states_t.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(centered, q=2)
    pca_proj = (centered @ Vt).detach().numpy()
    branch_labels = np.array(branch_labels)
    sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=branch_labels, cmap="tab10", s=15, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Branch index")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Branch States PCA")

    # 3. Real trajectory hidden-state norms with fork marker
    ax = axes[2]
    h_signal = CacheSignalPlotter.plot_cache_signal(real_traj_cache, "h")
    ax.plot(h_signal["timesteps"], h_signal["norms"], marker="o", color="steelblue", label="||h_t||")
    ax.axvline(fork_at, color="red", linestyle="--", linewidth=1.5, label=f"Fork at t={fork_at}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Real Trajectory Hidden State Norms")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Belief Analysis

def plot_belief_dashboard(
    wm: Any,
    traj: Any,
    cache: Any,
    saliency_result: Any,
    hallucination_result: Any,
    n_cat: int,
    n_cls: int,
    imagined_traj: Any,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Four-panel belief dashboard: surprise timeline with peaks, per-category
    KL heatmap, saliency bars, real-vs-imagined divergence with hallucination
    markers.

    Args:
        wm: HookedWorldModel instance.
        traj: Real LatentTrajectory.
        cache: ActivationCache from real trajectory.
        saliency_result: SaliencyResult from BeliefAnalyzer.latent_saliency().
        hallucination_result: HallucinationResult from BeliefAnalyzer.detect_hallucinations().
        n_cat: Number of latent categories.
        n_cls: Number of classes per category.
        imagined_traj: Imagined LatentTrajectory.
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    T = traj.length

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WorldModelLens - Belief Analysis Dashboard", fontsize=14)

    # 1. Surprise timeline
    ax = axes[0, 0]
    kl_vals = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_vals.append((p * (p.log() - q.log())).sum().item())
    ax.plot(range(T), kl_vals, marker="o", color="tomato")
    if hallucination_result is not None and hasattr(hallucination_result, "hallucination_timesteps"):
        peaks = [t for t in hallucination_result.hallucination_timesteps if t < T]
        if peaks:
            ax.scatter(peaks, [kl_vals[t] for t in peaks], color="navy", zorder=5, s=60, label="Peaks")
            ax.legend(fontsize=8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Surprise Timeline")

    # 2. Per-category surprise heatmap
    ax = axes[0, 1]
    kl_per_cat = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_per_cat.append((p * (p.log() - q.log())).sum(dim=-1).detach().numpy())
    heatmap_matrix = np.stack(kl_per_cat, axis=0)
    im = ax.imshow(heatmap_matrix.T, aspect="auto", origin="lower", cmap="hot")
    plt.colorbar(im, ax=ax, label="KL per category")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Latent Category")
    ax.set_title("Per-Category Surprise Heatmap")

    # 3. Saliency bars
    ax = axes[1, 0]
    h_sal = saliency_result.h_saliency.detach().numpy()
    z_sal = saliency_result.z_saliency.detach().numpy()
    top_h = np.argsort(np.abs(h_sal))[-16:][::-1]
    top_z = np.argsort(np.abs(z_sal))[-16:][::-1]
    x = np.arange(16)
    ax.bar(x - 0.2, np.abs(h_sal[top_h]), width=0.4, label="h saliency", color="steelblue")
    ax.bar(x + 0.2, np.abs(z_sal[top_z]), width=0.4, label="z saliency", color="darkorange")
    ax.set_xlabel("Top-16 dimensions")
    ax.set_ylabel("|Saliency|")
    ax.set_title("Saliency at t=5 (reward_pred target)")
    ax.legend(fontsize=8)

    # 4. Real vs imagined divergence
    ax = axes[1, 1]
    iv = InterventionVisualizer(wm)
    divergence_curve = iv.divergence_curve(traj, imagined_traj)
    ts = sorted(divergence_curve.keys())
    divs = [divergence_curve[t] for t in ts]
    ax.plot(ts, divs, marker="o", color="purple")
    if hallucination_result is not None and hallucination_result.hallucination_timesteps:
        for ht in hallucination_result.hallucination_timesteps:
            if ht < len(ts):
                ax.axvline(ht, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("State Divergence (MSE)")
    severity = getattr(hallucination_result, "severity_score", 0.0)
    ax.set_title(f"Real vs Imagined Divergence\n(severity={severity:.3f})")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Disentanglement

def plot_disentanglement_dashboard(
    disentanglement_result: Any,
    cache: Any,
    factors: Dict[str, torch.Tensor],
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Three-panel disentanglement dashboard: MIG/DCI/SAP score bars,
    factor-dimension assignment heatmap, z_posterior PCA colored by a factor.

    Args:
        disentanglement_result: DisentanglementResult from BeliefAnalyzer.disentanglement_score().
        cache: ActivationCache (used to get z_posterior sequence).
        factors: Dict of factor_name → per-timestep float tensor.
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WorldModelLens - Disentanglement Analysis Dashboard", fontsize=14)

    # 1. MIG/DCI/SAP scores
    ax = axes[0]
    metric_names = ["MIG", "DCI", "SAP"]
    metric_vals = [disentanglement_result.scores.get(m, 0.0) for m in metric_names]
    bars = ax.bar(metric_names, metric_vals, color=["steelblue", "darkorange", "mediumseagreen"])
    ax.set_ylabel("Score")
    ax.set_title(f"Disentanglement Metrics\n(total={disentanglement_result.total_score:.4f})")
    for bar, val in zip(bars, metric_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Factor-dimension assignment heatmap
    ax = axes[1]
    factor_names = list(disentanglement_result.factor_dim_assignment.keys())
    n_dims_show = 32
    heatmap_data = np.zeros((len(factor_names), n_dims_show))
    for i, fname in enumerate(factor_names):
        for d in disentanglement_result.factor_dim_assignment[fname]:
            if d < n_dims_show:
                heatmap_data[i, d] = 1.0
    if heatmap_data.sum() == 0:
        z_seq = cache["z_posterior"]
        var_per_dim = z_seq.var(dim=0).detach().numpy()[:n_dims_show]
        heatmap_data = np.tile(var_per_dim, (len(factor_names), 1))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(factor_names)))
    ax.set_yticklabels(factor_names)
    ax.set_xlabel("Latent Dimension")
    ax.set_title("Factor-Dimension Assignment")
    plt.colorbar(im, ax=ax)

    # 3. PCA colored by first factor
    ax = axes[2]
    z_seq = cache["z_posterior"]
    z_centered = z_seq - z_seq.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(z_centered, q=2)
    pca_proj = (z_centered @ Vt).detach().numpy()

    color_factor = list(factors.keys())[-1]  # last factor (e.g. reward_level)
    color_vals = factors[color_factor].numpy()
    sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=color_vals, cmap="RdYlGn", s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label=color_factor)
    n = len(pca_proj)
    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        ax.annotate(str(i), (pca_proj[i, 0], pca_proj[i, 1]), fontsize=7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"z_posterior PCA (colored by {color_factor})")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Video Model

def plot_video_model_dashboard(
    traj: Any,
    frames: torch.Tensor,
    preds: torch.Tensor,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Three-panel video model dashboard: latent state PCA, state norms over
    time, input vs predicted frame norms.

    Args:
        traj: LatentTrajectory from run_with_cache.
        frames: Input video frames [T, C, H, W].
        preds: Predicted frames [n_pred, C, H, W].
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    T = traj.length

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("WorldModelLens - Video Prediction Dashboard", fontsize=14)

    # 1. Latent state PCA colored by timestep
    state_tensors = torch.stack([traj.states[t].state for t in range(T)])
    state_centered = state_tensors - state_tensors.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(state_centered, q=2)
    pca_proj = (state_centered @ Vt).detach().numpy()

    ax = axes[0]
    sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=np.arange(T), cmap="viridis")
    plt.colorbar(sc, ax=ax, label="Timestep")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent State PCA")

    # 2. Hidden state norms over time
    ax = axes[1]
    norms = [traj.states[t].state.norm().item() for t in range(T)]
    ax.plot(range(T), norms, marker="o", color="steelblue")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||state||")
    ax.set_title("Latent State Norms")

    # 3. Input vs predicted frame norms
    ax = axes[2]
    n_inp = frames.shape[0]
    n_pred = preds.shape[0]
    ax.plot(range(n_inp), [frames[t].norm().item() for t in range(n_inp)],
            marker="o", label="Input frames", color="darkorange")
    ax.plot(range(n_pred), [preds[t].norm().item() for t in range(n_pred)],
            marker="s", label="Predicted frames", color="tomato")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||frame||")
    ax.set_title("Input vs Predicted Frame Norms")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Toy Video World Model

def plot_toy_video_dashboard(
    traj: Any,
    cache: Any,
    mem_result: Any,
    # surprise_result=None,  # BeliefAnalyzer.surprise_timeline() result
    # traj_metrics=None,    # GeometryAnalyzer.trajectory_metrics() result
    # manifold_result=None, # GeometryAnalyzer.manifold_analysis() result
    # dep_result=None,      # TemporalMemoryProber.temporal_dependencies() result
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Three-panel toy-video dashboard: latent PCA of moving-pattern video,
    z_posterior activation norms, memory retention.

    Args:
        traj: LatentTrajectory from run_with_cache.
        cache: ActivationCache from run_with_cache.
        mem_result: MemoryRetentionResult from TemporalMemoryProber.memory_retention().
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    T = len(traj.states)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("WorldModelLens - Toy Video World Model Dashboard", fontsize=14)

    # 1. Latent state PCA colored by timestep (or surprise if available)
    state_tensors = torch.stack([traj.states[t].state for t in range(T)])
    state_centered = state_tensors - state_tensors.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(state_centered, q=2)
    pca_proj = (state_centered @ Vt).detach().numpy()

    ax = axes[0]
    # if surprise_result is not None:
    #     # Color by surprise values
    #     surprise_vals = [surprise_result.mean_surprise if hasattr(surprise_result, 'mean_surprise') else 0] * T  # placeholder
    #     sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=surprise_vals, cmap="plasma")
    #     plt.colorbar(sc, ax=ax, label="Surprise")
    # else:
    sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=np.arange(T), cmap="plasma")
    plt.colorbar(sc, ax=ax, label="Timestep")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent State PCA (moving pattern)")

    # 2. z_posterior activation norms over time
    ax = axes[1]
    z_keys = sorted({t for (n, t) in cache.keys() if n == "z_posterior"})
    norms = [cache.get("z_posterior", t).norm().item() for t in z_keys]
    ax.plot(z_keys, norms, marker="o", color="tomato")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||z_posterior||")
    ax.set_title("Latent Activation Norms")
    # if traj_metrics is not None:
    #     # Add trajectory metrics as text
    #     dist = getattr(traj_metrics, 'mean_trajectory_distance', 0)
    #     coh = getattr(traj_metrics, 'temporal_coherence', 0)
    #     ax.text(0.05, 0.95, f'Distance: {dist:.3f}\nCoherence: {coh:.3f}', 
    #             transform=ax.transAxes, fontsize=8, verticalalignment='top',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Memory retention
    ax = axes[2]
    deps = mem_result.temporal_dependencies
    lags = list(deps.keys())
    vals = [deps[l] for l in lags]
    if isinstance(lags[0], str):
        ax.bar(lags, vals, color="steelblue")
        ax.set_ylabel("Retention")
    else:
        ax.plot(lags, vals, marker="o", color="steelblue")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Retention")
    title = f"Memory Retention (capacity={mem_result.memory_capacity:.2f})"
    # if manifold_result is not None:
    #     dim = getattr(manifold_result, 'intrinsic_dimensionality_estimate', 'N/A')
    #     title += f"\nIntrinsic Dim: {dim}"
    # if dep_result is not None and hasattr(dep_result, 'autocorrelations'):
    #     autocorr = dep_result['autocorrelations']
    #     if len(autocorr) > 0:
    #         top_corr = max(autocorr[:5]) if len(autocorr) >= 5 else max(autocorr)
    #         title += f"\nTop Autocorr: {top_corr:.3f}"
    ax.set_title(title)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Scientific Dynamics

def plot_scientific_dynamics_dashboard(
    lorenz_traj_raw: torch.Tensor,
    traj_lorenz: Any,
    traj_pendulum: Any,
    dep_result: Dict,
    mem_result: Any,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Four-panel scientific-dynamics dashboard: 3D Lorenz attractor, latent
    PCA of Lorenz run, Lorenz vs pendulum latent norm comparison, temporal
    autocorrelations.

    Args:
        lorenz_traj_raw: Raw Lorenz trajectory [T, 3] (for 3D plot).
        traj_lorenz: LatentTrajectory from Lorenz run.
        traj_pendulum: LatentTrajectory from pendulum run.
        dep_result: Dict from TemporalMemoryProber.temporal_dependencies().
        mem_result: MemoryRetentionResult (for dominant period label).
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    fig = plt.figure(figsize=(18, 4))
    fig.suptitle("WorldModelLens - Scientific Dynamics Dashboard", fontsize=14)

    # 1. Lorenz attractor (raw 3D trajectory)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.plot(
        lorenz_traj_raw[:, 0].numpy(),
        lorenz_traj_raw[:, 1].numpy(),
        lorenz_traj_raw[:, 2].numpy(),
        lw=0.5,
        color="steelblue",
    )
    ax1.set_title("Lorenz Attractor")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # 2. Lorenz latent PCA
    ax2 = fig.add_subplot(1, 4, 2)
    T_lor = len(traj_lorenz.states)
    state_tensors = torch.stack([traj_lorenz.states[t].state for t in range(T_lor)])
    state_centered = state_tensors - state_tensors.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(state_centered, q=2)
    pca_proj = (state_centered @ Vt).detach().numpy()
    sc = ax2.scatter(pca_proj[:, 0], pca_proj[:, 1], c=np.arange(T_lor), cmap="viridis", s=10)
    plt.colorbar(sc, ax=ax2, label="Timestep")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Lorenz Latent PCA")

    # 3. Lorenz vs pendulum latent norms
    ax3 = fig.add_subplot(1, 4, 3)
    lor_norms = [traj_lorenz.states[t].state.norm().item() for t in range(T_lor)]
    T_pen = len(traj_pendulum.states)
    pen_norms = [traj_pendulum.states[t].state.norm().item() for t in range(T_pen)]
    ax3.plot(range(T_lor), lor_norms, label="Lorenz", color="steelblue")
    ax3.plot(range(T_pen), pen_norms, label="Pendulum", color="darkorange")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("||state||")
    ax3.set_title("Lorenz vs Pendulum\nLatent Norms")
    ax3.legend(fontsize=8)

    # 4. Temporal autocorrelations
    ax4 = fig.add_subplot(1, 4, 4)
    autocorrs = dep_result["autocorrelations"]
    if isinstance(autocorrs, dict):
        lags_sorted = sorted(autocorrs.keys())
        ac_vals = [autocorrs[l] for l in lags_sorted]
        ax4.stem(lags_sorted, ac_vals, basefmt=" ")
    else:
        ac_arr = np.array(autocorrs)
        ax4.stem(range(len(ac_arr)), ac_arr, basefmt=" ")
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("Autocorrelation")
    ax4.set_title(f"Temporal Autocorrelations\n(dominant period={dep_result['dominant_period']})")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


# Causal Engine

def plot_causal_engine_dashboard(
    wm: Any,
    div_curve: Dict[int, float],
    tree: Any,
    cache: Any,
    intervention_t: int = 5,
    output_path: Optional[pathlib.Path] = None,
) -> Any:
    """Three-panel causal-engine dashboard: cumulative divergence curve,
    per-branch intervention divergence bars, baseline hidden-state norms.

    Args:
        wm: HookedWorldModel instance (passed through for future use).
        div_curve: Dict[timestep, cumulative_divergence] from engine.trace_divergence().
        tree: BranchTree from engine.build_branch_tree().
        cache: ActivationCache from baseline run.
        intervention_t: Timestep of the intervention (draws vertical line).
        output_path: If given, save the figure here.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WorldModelLens - Causal Engine Dashboard", fontsize=14)

    # 1. Cumulative divergence curve
    ax = axes[0]
    ts = sorted(div_curve.keys())
    divs = [div_curve[t] for t in ts]
    ax.plot(ts, divs, marker="o", color="tomato")
    ax.axvline(intervention_t, color="gray", linestyle="--", linewidth=1.5,
               label=f"Intervention at t={intervention_t}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Divergence")
    ax.set_title("Baseline vs Counterfactual\nCumulative Divergence")
    ax.legend(fontsize=8)

    # 2. Branch divergence bar chart
    ax = axes[1]
    branch_labels = [br.intervention.description for br in tree.branches]
    branch_divs = [br.divergence for br in tree.branches]
    colors = ["steelblue", "darkorange", "mediumseagreen"]
    bars = ax.bar(range(len(branch_labels)), branch_divs,
                  color=colors[: len(branch_labels)])
    ax.set_xticks(range(len(branch_labels)))
    ax.set_xticklabels(branch_labels, fontsize=8, rotation=10)
    ax.set_ylabel("Trajectory Distance")
    ax.set_title("Divergence per Intervention Branch")
    for bar, val in zip(bars, branch_divs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 3. Baseline hidden state norms
    ax = axes[2]
    h_signal = CacheSignalPlotter.plot_cache_signal(cache, "h")
    ax.plot(h_signal["timesteps"], h_signal["norms"], marker="o", color="steelblue")
    ax.axvline(intervention_t, color="red", linestyle="--", linewidth=1.5,
               label=f"Intervention at t={intervention_t}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Baseline Hidden State Norms")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig
