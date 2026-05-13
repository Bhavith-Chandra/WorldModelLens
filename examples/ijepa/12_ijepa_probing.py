from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import (
    OUTPUT_DIR,
    PatchLabelBuilder,
    build_disentanglement_factors,
    build_patch_heatmap,
    build_patch_labels,
    collect_ijepa_artifacts,
    get_patch_rect,
    get_sample_image,
    load_ijepa_world_model,
    preprocess_image,
)
from world_model_lens import LatentProber
from world_model_lens.analysis.metrics import DisentanglementEvaluationSuite
from world_model_lens.probing.prober import ProbeResult

try:
    from world_model_lens.probing.semantic_probes import IJEPASemanticAligner, SemanticProber

    _SEMANTIC_AVAILABLE = True
except Exception:
    IJEPASemanticAligner = None
    SemanticProber = None
    _SEMANTIC_AVAILABLE = False

_COMPONENT_COLORS = {
    "encoder.out": "#4C72B0",
    "target_encoder_out": "#55A868",
    "predictor_out": "#DD8452",
}
_CONCEPT_LABELS = {
    "top_half": "Top Half",
    "left_half": "Left Half",
    "center_band": "Center Band",
}
_CLIP_CONCEPT_PROMPTS = {
    "top_half": (
        "image patches from the upper half of a picture",
        "image patches from the lower half of a picture",
    ),
    "left_half": (
        "image patches from the left side of a picture",
        "image patches from the right side of a picture",
    ),
    "center_band": (
        "image patches from the center of a picture",
        "image patches from the edges of a picture",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint-first I-JEPA probing example.")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the I-JEPA checkpoint."
    )
    parser.add_argument("--image", type=str, default=None, help="Path to a specific image file.")
    parser.add_argument(
        "--imagenet-root",
        type=str,
        default=None,
        help="Optional ImageNet root. The first image found under val/train will be used.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Optional torch device, e.g. cpu or cuda."
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=1,
        help="Number of images for multi-image aggregate probing (default 1 = single-image mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory where plots and text summaries will be written.",
    )
    parser.add_argument(
        "--dino-model",
        type=str,
        default="dino_vitb16",
        choices=["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"],
        help="DINO backbone for semantic probes.",
    )
    parser.add_argument(
        "--semantic-projector-epochs",
        type=int,
        default=300,
        help="Training epochs for DINO/CLIP latent projectors used by semantic probes.",
    )
    return parser.parse_args()


_GEOMETRIC_PROBE_SPECS = [
    ("top_half", "logistic"),
    ("left_half", "logistic"),
    ("center_band", "logistic"),
    ("patch_index_norm", "ridge"),
    ("row_norm", "ridge"),
    ("col_norm", "ridge"),
    ("quadrant", "logistic"),
    ("is_boundary", "logistic"),
]
_CONTENT_PROBE_SPECS = [
    ("brightness", "ridge"),
    ("edge_density", "ridge"),
    ("texture_variance", "ridge"),
    ("saturation", "ridge"),
]


def _run_probes(
    prober: LatentProber,
    component_name: str,
    activations: torch.Tensor,
    labels: dict[str, np.ndarray],
    include_content: bool = False,
) -> dict[str, object]:
    """Train all probes given a pre-built labels dict."""
    specs = _GEOMETRIC_PROBE_SPECS + (_CONTENT_PROBE_SPECS if include_content else [])
    results: dict[str, object] = {}
    for concept_name, probe_type in specs:
        if concept_name not in labels:
            continue
        concept_labels = np.asarray(labels[concept_name])
        if np.unique(concept_labels).size < 2:
            results[concept_name] = ProbeResult(
                accuracy=1.0,
                r2=None,
                direction=torch.zeros(activations.shape[-1]),
                feature_weights=torch.zeros(activations.shape[-1]),
                confusion_matrix=None,
                concept_name=concept_name,
                activation_name=component_name,
                probe_type=probe_type,
                training_samples=int(concept_labels.shape[0]),
                test_samples=0,
                cv_scores=[],
                cv_mean=1.0,
                cv_std=0.0,
                regularization_alpha=1.0,
            )
            print(
                f"  [probe] {component_name} / {concept_name}: skipped degenerate labels "
                f"(single class)"
            )
            continue
        results[concept_name] = prober.train_probe(
            activations=activations,
            labels=concept_labels,
            concept_name=concept_name,
            activation_name=component_name,
            probe_type=probe_type,
        )
    return results


def _probe_component(
    prober: LatentProber,
    component_name: str,
    activations: torch.Tensor,
    patch_ids: list[int],
    grid_size: int,
    raw_image=None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    builder = PatchLabelBuilder(patch_ids, grid_size)
    labels = builder.build_all(raw_image) if raw_image is not None else builder.build_geometric()
    results = _run_probes(
        prober, component_name, activations, labels, include_content=(raw_image is not None)
    )
    return results, labels


def _plot_probe_overview(
    artifacts,
    component_results,
    component_labels,
    output_path: Path,
    disentanglement_result=None,
) -> None:
    component_names = list(component_results.keys())
    n_comps = len(component_names)
    fig = plt.figure(figsize=(24, 15))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[0.85, 2.2], wspace=0.08, left=0.03, right=0.97, top=0.91, bottom=0.05
    )
    ax_img = fig.add_subplot(outer[0])
    n_right_rows = 3 if disentanglement_result is not None else 2
    height_ratios = [1.1, 1.1, 0.6] if disentanglement_result is not None else [1.1, 1.1]
    inner = outer[1].subgridspec(
        n_right_rows, 2, height_ratios=height_ratios, hspace=0.55, wspace=0.38
    )
    ax_bar = fig.add_subplot(inner[0, 0])
    ax_r2 = fig.add_subplot(inner[0, 1])
    pca_inner = inner[1, :].subgridspec(1, n_comps, wspace=0.38)
    pca_axes = [fig.add_subplot(pca_inner[i]) for i in range(n_comps)]
    # --- Image panel ---
    ax_img.imshow(artifacts.raw_image.resize((224, 224)))
    for patch_id in artifacts.context_ids:
        x, y, w, h = get_patch_rect(patch_id, artifacts.grid_size, image_size=224)
        ax_img.add_patch(
            patches.Rectangle(
                (x, y), w, h, linewidth=0.8, edgecolor="#4CAF50", facecolor="#4CAF50", alpha=0.15
            )
        )
    for patch_id in artifacts.target_ids:
        x, y, w, h = get_patch_rect(patch_id, artifacts.grid_size, image_size=224)
        ax_img.add_patch(
            patches.Rectangle(
                (x, y), w, h, linewidth=1.5, edgecolor="#F44336", facecolor="#F44336", alpha=0.20
            )
        )
    ax_img.legend(
        handles=[
            patches.Patch(
                facecolor="#4CAF50",
                alpha=0.6,
                label=f"Context ({len(artifacts.context_ids)} patches)",
            ),
            patches.Patch(
                facecolor="#F44336",
                alpha=0.6,
                label=f"Target ({len(artifacts.target_ids)} patches)",
            ),
        ],
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )
    ax_img.set_title("Context / Target Patch Layout", fontweight="bold", pad=8)
    ax_img.axis("off")
    # --- Classification accuracy bars with cv_std error bars ---
    concepts = ["top_half", "left_half", "center_band"]
    bar_w = 0.22
    x = np.arange(len(concepts))
    for i, cname in enumerate(component_names):
        color = _COMPONENT_COLORS.get(cname, f"C{i}")
        accs = [component_results[cname][c].accuracy for c in concepts]
        errs = [component_results[cname][c].cv_std for c in concepts]
        offset = (i - n_comps / 2 + 0.5) * bar_w
        bars = ax_bar.bar(
            x + offset,
            accs,
            bar_w,
            label=cname.replace("_", " "),
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "#444", "capthick": 1.2},
        )
        for bar, acc, err in zip(bars, accs, errs, strict=False):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                acc + err + 0.025,
                f"{acc:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )
    ax_bar.axhline(
        0.5, color="#888", linestyle="--", linewidth=1.0, alpha=0.7, label="chance (0.5)"
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([_CONCEPT_LABELS[c] for c in concepts], fontsize=9)
    ax_bar.set_ylim(0.0, 1.32)
    ax_bar.set_ylabel("Accuracy +/- CV std", fontsize=9)
    ax_bar.set_title(
        "Linear Probe Accuracy\n(spatial concept decoding)", fontweight="bold", pad=6, fontsize=9
    )
    ax_bar.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax_bar.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax_bar.spines[["top", "right"]].set_visible(False)
    # --- R² subplot for patch_index_norm (positional regression) ---
    r2_vals, r2_colors, r2_labels = [], [], []
    for cname in component_names:
        result = component_results[cname].get("patch_index_norm")
        if result is not None and result.r2 is not None:
            r2 = float(result.r2)
            r2_vals.append(r2)
            r2_colors.append(_COMPONENT_COLORS.get(cname, "gray") if r2 >= 0 else "#d62728")
            r2_labels.append(cname.replace("_", "\n"))
    if r2_vals:
        bars_r2 = ax_r2.bar(
            np.arange(len(r2_vals)),
            r2_vals,
            color=r2_colors,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            width=0.5,
        )
        for bar, val in zip(bars_r2, r2_vals, strict=False):
            ypos = val + 0.03 if val >= 0 else val - 0.07
            va = "bottom" if val >= 0 else "top"
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:.3f}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )
        ax_r2.axhline(0.0, color="#555", linewidth=1.0)
        ax_r2.axhline(1.0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_r2.text(
            len(r2_vals) - 0.45, 1.02, "R^2 = 1 (perfect)", fontsize=7, color="#888", va="bottom"
        )
        ax_r2.set_xticks(np.arange(len(r2_labels)))
        ax_r2.set_xticklabels(r2_labels, fontsize=9)
        ymin = min(min(r2_vals) - 0.2, -0.5)
        ax_r2.set_ylim(ymin, 1.18)
        ax_r2.set_ylabel("R^2", fontsize=9)
        ax_r2.set_title(
            "Ridge Probe R^2: patch_index_norm\n(can absolute patch position be decoded?)",
            fontweight="bold",
            pad=6,
            fontsize=9,
        )
        ax_r2.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax_r2.spines[["top", "right"]].set_visible(False)
    # --- PCA for all components, shared colorbar on last axis ---
    norm = mcolors.Normalize(vmin=0, vmax=1)
    for ax_pca, cname in zip(pca_axes, component_names, strict=False):
        activations = artifacts.components[cname].activations
        labels_arr = component_labels[cname]["top_half"]
        acts_c = activations - activations.mean(0)
        _, _, vt = torch.pca_lowrank(acts_c, q=2)
        proj = (acts_c @ vt).detach().cpu().numpy()
        ax_pca.scatter(
            proj[:, 0],
            proj[:, 1],
            c=labels_arr,
            cmap="coolwarm",
            norm=norm,
            s=28,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
        )
        acc = component_results[cname]["top_half"].accuracy
        ax_pca.set_title(
            f"{cname.replace('_', ' ').title()}\ntop_half probe={acc:.2f}",
            fontweight="bold",
            pad=4,
            fontsize=8,
        )
        ax_pca.set_xlabel("PC 1", fontsize=8)
        ax_pca.set_ylabel("PC 2", fontsize=8)
        ax_pca.tick_params(labelsize=7)
        ax_pca.grid(alpha=0.25, linewidth=0.7)
        ax_pca.spines[["top", "right"]].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    cb = fig.colorbar(sm, ax=pca_axes[-1], shrink=0.8, pad=0.04)
    cb.set_label("Top-half\n(1=upper, 0=lower)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    # --- Disentanglement: DCI and MIG/SAP on separate scales ---
    if disentanglement_result is not None:
        ax_dci = fig.add_subplot(inner[2, 0])
        ax_mig = fig.add_subplot(inner[2, 1])
        scores = disentanglement_result.summary_scores
        dci_keys = [m for m in scores if m.startswith("DCI")]
        sparse_keys = [m for m in scores if not m.startswith("DCI")]
        dci_vals = [scores[m] for m in dci_keys]
        dci_labels = [m.replace("DCI_", "") for m in dci_keys]
        dci_colors = [
            _COMPONENT_COLORS["target_encoder_out"]
            if v >= 0.5
            else _COMPONENT_COLORS["predictor_out"]
            for v in dci_vals
        ]
        bdci = ax_dci.barh(dci_labels, dci_vals, color=dci_colors, alpha=0.85, edgecolor="white")
        ax_dci.axvline(0.5, color="#888", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_dci.set_xlim(0, 1.2)
        for bar, val in zip(bdci, dci_vals, strict=False):
            ax_dci.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )
        ax_dci.set_title("DCI Metrics (0–1 scale)", fontweight="bold", pad=4, fontsize=9)
        ax_dci.grid(axis="x", alpha=0.3, linewidth=0.7)
        ax_dci.spines[["top", "right"]].set_visible(False)
        ax_dci.tick_params(labelsize=8)
        sparse_vals = [scores[m] for m in sparse_keys]
        sparse_max = max(sparse_vals) * 1.6 if sparse_vals else 0.02
        bsp = ax_mig.barh(sparse_keys, sparse_vals, color="#9e9e9e", alpha=0.85, edgecolor="white")
        ax_mig.set_xlim(0, max(sparse_max, 0.02))
        for bar, val in zip(bsp, sparse_vals, strict=False):
            ax_mig.text(
                val + sparse_max * 0.04,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )
        ax_mig.set_title("MIG / SAP (near-zero = entangled)", fontweight="bold", pad=4, fontsize=9)
        ax_mig.grid(axis="x", alpha=0.3, linewidth=0.7)
        ax_mig.spines[["top", "right"]].set_visible(False)
        ax_mig.tick_params(labelsize=8)
    fig.suptitle("I-JEPA Patch Probing Overview", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_component_heatmaps(artifacts, output_path: Path) -> None:
    component_names = [
        name
        for name in ["encoder.out", "predictor_out", "target_encoder_out"]
        if name in artifacts.components
    ]
    n = max(1, len(component_names))
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 0.5, 5.5))
    axes = np.atleast_1d(axes)
    # Compute global min/max across all components for consistent color scale
    all_energies = []
    for component_name in component_names:
        component = artifacts.components[component_name]
        energy = component.activations.norm(dim=1)
        all_energies.append(energy)
    all_energies_tensor = torch.cat(all_energies)
    global_emin = float(all_energies_tensor.min())
    global_emax = float(all_energies_tensor.max())
    for ax, component_name in zip(axes, component_names, strict=False):
        component = artifacts.components[component_name]
        energy = component.activations.norm(dim=1)
        heatmap = build_patch_heatmap(component.patch_ids, energy, artifacts.grid_size)
        ax.imshow(artifacts.raw_image.resize((224, 224)), alpha=0.60)
        im = ax.imshow(
            heatmap,
            cmap="inferno",
            alpha=0.65,
            extent=(0, 224, 224, 0),
            interpolation="nearest",
            vmin=global_emin,
            vmax=global_emax,
        )
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("L2 norm", fontsize=8)
        cb.ax.tick_params(labelsize=8)
        emin, emax = float(energy.min()), float(energy.max())
        erange = emax - emin
        pct = erange / emax * 100 if emax > 0 else 0
        title = component_name.replace("_", " ").title()
        ax.set_title(
            f"{title}\nDelta={erange:.4f}  ({pct:.2f}% of max)",
            fontweight="bold",
            pad=6,
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle(
        "Per-Patch Activation Energy\n(L2 norm of latent feature vector per patch)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    artifacts,
    metadata,
    checkpoint_path: Path,
    component_results,
    disentanglement_result,
    semantic_results,
    dino_results,
    output_path: Path,
) -> None:
    lines = []
    lines.append("I-JEPA Patch Probing Summary")
    lines.append("=" * 40)
    lines.append(f"checkpoint: {checkpoint_path}")
    lines.append(f"image_source: {metadata.get('source', 'unknown')}")
    if "path" in metadata:
        lines.append(f"image_path: {metadata['path']}")
    if "label_name" in metadata:
        lines.append(f"label_name: {metadata['label_name']}")
    lines.append(f"context_patches: {len(artifacts.context_ids)}")
    lines.append(f"target_patches: {len(artifacts.target_ids)}")
    lines.append("")
    lines.append("Probe Results")
    lines.append("-" * 40)
    for component_name, result_dict in component_results.items():
        lines.append(component_name)
        for concept_name, result in result_dict.items():
            metric_name = "r2" if result.r2 is not None else "accuracy"
            metric_value = result.r2 if result.r2 is not None else result.accuracy
            lines.append(
                f"  {concept_name}: probe={result.probe_type} {metric_name}={metric_value:.4f} cv_mean={result.cv_mean:.4f} cv_std={result.cv_std:.4f} alpha={result.regularization_alpha:.4g}"
            )
        lines.append("")
    lines.append("Disentanglement Summary")
    lines.append("-" * 40)
    lines.append(f"evaluated_components: {len(disentanglement_result.component_results)}")
    for metric_name, score in disentanglement_result.summary_scores.items():
        lines.append(f"{metric_name}: {score:.4f}")
    if disentanglement_result.component_errors:
        lines.append("")
        lines.append("Disentanglement Errors")
        lines.append("-" * 40)
        for component_name, error_text in disentanglement_result.component_errors.items():
            lines.append(f"{component_name}: {error_text}")
    lines.append("")
    if dino_results is not None:
        lines.append("DINO Semantic Alignment")
        lines.append("-" * 40)
        for component_name, result in dino_results.items():
            summary = "  ".join(
                f"{concept}={score:.4f}"
                for concept, score in result.concept_feature_alignments.items()
            )
            lines.append(
                f"{component_name}: r2_projection={result.r2_projection:.4f} {summary}".rstrip()
            )
        lines.append("")
    if semantic_results is not None:
        lines.append("CLIP Semantic Alignment")
        lines.append("-" * 40)
        for component_name, result in semantic_results.items():
            summary = "  ".join(
                f"{concept}={score:.4f}"
                for concept, score in result.concept_text_alignments.items()
            )
            lines.append(
                f"{component_name}: r2_projection={result.r2_projection:.4f} {summary}".rstrip()
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_probe_heatmap(
    component_results: dict,
    output_path: Path,
    title: str = "Probe Accuracy / R^2 Heatmap",
    n_images: int = 1,
) -> None:
    """Heatmap of probe scores: rows = concepts, columns = components."""
    component_names = list(component_results.keys())
    # Collect all concept names present across components
    all_concepts: list[str] = []
    seen: set[str] = set()
    for res in component_results.values():
        for c in res:
            if c not in seen:
                all_concepts.append(c)
                seen.add(c)
    matrix = np.full((len(all_concepts), len(component_names)), np.nan)
    has_r2 = False
    for j, cname in enumerate(component_names):
        for i, concept in enumerate(all_concepts):
            r = component_results[cname].get(concept)
            if r is None:
                continue
            if r.r2 is not None:
                matrix[i, j] = r.r2
                has_r2 = True
            else:
                matrix[i, j] = r.accuracy
    fig, ax = plt.subplots(
        figsize=(max(6, len(component_names) * 2.2), max(5, len(all_concepts) * 0.55))
    )
    # Use symmetric scale for R² (which can be negative); accuracy stays 0-1
    if has_r2 and np.nanmin(matrix) < 0:
        vmin, vmax = -1, 1
        cmap = "RdBu_r"
    else:
        vmin, vmax = 0, 1
        cmap = "RdYlGn"
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(component_names)))
    ax.set_xticklabels([c.replace("_", "\n") for c in component_names], fontsize=9)
    ax.set_yticks(range(len(all_concepts)))
    ax.set_yticklabels(all_concepts, fontsize=8)
    for i in range(len(all_concepts)):
        for j in range(len(component_names)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.25 < v < 0.85 else "white",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy (clf) / R^2 (reg)", fontsize=8)
    suffix = f"  [{n_images} image{'s' if n_images > 1 else ''}]"
    ax.set_title(f"{title}{suffix}", fontweight="bold", fontsize=10)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_dino_semantic_probes(
    artifacts,
    dino_model: str = "dino_vitb16",
    projector_epochs: int = 300,
) -> dict | None:
    """Run DINO semantic alignment probes across all I-JEPA components."""
    if not _SEMANTIC_AVAILABLE:
        print("  [dino-semantic] semantic probes unavailable - skipping DINO alignment probes.")
        return None
    try:
        prober = SemanticProber(model_name=dino_model)
    except Exception as exc:
        print(f"  [dino-semantic] Could not initialise SemanticProber: {exc}")
        return None
    results = {}
    for component_name, component in artifacts.components.items():
        labels = build_patch_labels(component.patch_ids, artifacts.grid_size)
        cls_labels = {k: v for k, v in labels.items() if k in _CLIP_CONCEPT_PROMPTS}
        try:
            result = prober.run_patch_alignment(
                raw_image=artifacts.raw_image,
                activations=component.activations,
                patch_ids=component.patch_ids,
                grid_size=artifacts.grid_size,
                concept_labels=cls_labels,
                component_name=component_name,
                projector_epochs=projector_epochs,
            )
            results[component_name] = result
            print(
                f"  [dino-semantic] {component_name}: R^2={result.r2_projection:.3f}  "
                + "  ".join(f"{c}={v:.3f}" for c, v in result.concept_feature_alignments.items())
            )
        except Exception as exc:
            print(f"  [dino-semantic] {component_name} failed: {exc}")
    return results or None


def _plot_dino_alignment(dino_results: dict, output_path: Path) -> None:
    """Two-panel plot: DINO concept-direction cosine similarity + projector R^2."""
    component_names = list(dino_results.keys())
    concepts = list(_CLIP_CONCEPT_PROMPTS.keys())
    n_comps = len(component_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bar_w = 0.22
    x = np.arange(len(concepts))
    for i, cname in enumerate(component_names):
        res = dino_results[cname]
        vals = [res.concept_feature_alignments.get(c, 0.0) for c in concepts]
        offset = (i - n_comps / 2 + 0.5) * bar_w
        color = _COMPONENT_COLORS.get(cname, f"C{i}")
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=cname.replace("_", " "),
            color=color,
            alpha=0.88,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals, strict=False):
            yoff = 0.012 if val >= 0 else -0.03
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + yoff,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )
    ax.axhline(0, color="#555", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_CONCEPT_LABELS[c] for c in concepts], fontsize=9)
    ax.set_ylabel("Cosine Similarity (DINO concept vs projected I-JEPA direction)", fontsize=9)
    ax.set_title(
        "DINO Feature - I-JEPA Concept Direction Alignment\n"
        "(positive = semantically aligned with DINO patch features)",
        fontweight="bold",
        fontsize=9,
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    r2_vals = [dino_results[c].r2_projection for c in component_names]
    colors = [_COMPONENT_COLORS.get(c, "gray") for c in component_names]
    bars2 = ax2.bar(
        range(len(component_names)),
        r2_vals,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars2, r2_vals, strict=False):
        yoff = 0.02 if val >= 0 else -0.05
        va = "bottom" if val >= 0 else "top"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + yoff,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_xticks(range(len(component_names)))
    ax2.set_xticklabels([c.replace("_", "\n") for c in component_names], fontsize=9)
    r2_min = min(r2_vals) if r2_vals else 0
    r2_max = max(r2_vals) if r2_vals else 1
    ymin = min(r2_min - 0.15, -0.1)
    ymax = max(r2_max + 0.15, 1.15)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("R^2  (I-JEPA latents -> DINO patch embeddings)", fontsize=9)
    ax2.set_title(
        "DINO Projector R^2\n(how much of DINO patch structure lives in I-JEPA?)",
        fontweight="bold",
        fontsize=9,
    )
    ax2.axhline(1.0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "DINO Semantic Alignment Probes on I-JEPA Patch Latents", fontsize=13, fontweight="bold"
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.13, wspace=0.35)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_semantic_probes(artifacts, projector_epochs: int = 300) -> dict | None:
    """Run CLIP semantic alignment probes across all I-JEPA components.
    Returns a dict mapping component name -> SemanticAlignmentResult,
    or None when the transformers package is unavailable.
    """
    if not _SEMANTIC_AVAILABLE:
        print("  [semantic] transformers not installed — skipping CLIP alignment probes.")
        return None
    try:
        aligner = IJEPASemanticAligner()
    except Exception as exc:
        print(f"  [semantic] Could not initialise IJEPASemanticAligner: {exc}")
        return None
    results = {}
    for component_name, component in artifacts.components.items():
        labels = build_patch_labels(component.patch_ids, artifacts.grid_size)
        cls_labels = {k: v for k, v in labels.items() if k in _CLIP_CONCEPT_PROMPTS}
        try:
            result = aligner.run(
                raw_image=artifacts.raw_image,
                activations=component.activations,
                patch_ids=component.patch_ids,
                grid_size=artifacts.grid_size,
                concept_labels=cls_labels,
                concept_prompts=_CLIP_CONCEPT_PROMPTS,
                component_name=component_name,
                projector_epochs=projector_epochs,
            )
            results[component_name] = result
            print(
                f"  [semantic] {component_name}: R²={result.r2_projection:.3f}  "
                + "  ".join(f"{c}={v:.3f}" for c, v in result.concept_text_alignments.items())
            )
        except Exception as exc:
            print(f"  [semantic] {component_name} failed: {exc}")
    return results or None


def _plot_semantic_alignment(semantic_results: dict, output_path: Path) -> None:
    """Two-panel plot: concept-text cosine similarity + ridge projection R^2."""
    component_names = list(semantic_results.keys())
    concepts = list(_CLIP_CONCEPT_PROMPTS.keys())
    n_comps = len(component_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: concept-text cosine similarity bars
    ax = axes[0]
    bar_w = 0.22
    x = np.arange(len(concepts))
    for i, cname in enumerate(component_names):
        res = semantic_results[cname]
        vals = [res.concept_text_alignments.get(c, 0.0) for c in concepts]
        offset = (i - n_comps / 2 + 0.5) * bar_w
        color = _COMPONENT_COLORS.get(cname, f"C{i}")
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=cname.replace("_", " "),
            color=color,
            alpha=0.88,
            edgecolor="white",
        )
        for bar, val in zip(bars, vals, strict=False):
            yoff = 0.012 if val >= 0 else -0.03
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + yoff,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )
    ax.axhline(0, color="#555", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_CONCEPT_LABELS[c] for c in concepts], fontsize=9)
    ax.set_ylabel("Cosine Similarity (CLIP text vs projected I-JEPA direction)", fontsize=9)
    ax.set_title(
        "CLIP Text - I-JEPA Concept Direction Alignment\n"
        "(positive = semantically aligned with CLIP's spatial understanding)",
        fontweight="bold",
        fontsize=9,
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    # Right: Ridge projection R² per component
    ax2 = axes[1]
    r2_vals = [semantic_results[c].r2_projection for c in component_names]
    colors = [_COMPONENT_COLORS.get(c, "gray") for c in component_names]
    bars2 = ax2.bar(
        range(len(component_names)), r2_vals, color=colors, alpha=0.88, edgecolor="white", width=0.5
    )
    for bar, val in zip(bars2, r2_vals, strict=False):
        yoff = 0.02 if val >= 0 else -0.05
        va = "bottom" if val >= 0 else "top"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + yoff,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_xticks(range(len(component_names)))
    ax2.set_xticklabels([c.replace("_", "\n") for c in component_names], fontsize=9)
    r2_min = min(r2_vals) if r2_vals else 0
    r2_max = max(r2_vals) if r2_vals else 1
    ymin = min(r2_min - 0.15, -0.1)
    ymax = max(r2_max + 0.15, 1.15)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("R^2  (I-JEPA latents -> CLIP image patch embeddings)", fontsize=9)
    ax2.set_title(
        "CrossModal Projector R^2\n(how much of CLIP patch structure lives in I-JEPA?)",
        fontweight="bold",
        fontsize=9,
    )
    ax2.axhline(1.0, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "CLIP Semantic Alignment Probes on I-JEPA Patch Latents", fontsize=13, fontweight="bold"
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.13, wspace=0.35)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_patch_semantic_probes(artifacts, projector_epochs: int = 300) -> dict | None:
    """Run per-patch CLIP text similarity probes across all I-JEPA components."""
    if not _SEMANTIC_AVAILABLE:
        print("  [patch-semantic] transformers not installed — skipping patch text probes.")
        return None
    try:
        from world_model_lens.probing.semantic_probes import CLIPTextProber

        prober = CLIPTextProber()
    except Exception as exc:
        print(f"  [patch-semantic] Could not initialise CLIPTextProber: {exc}")
        return None
    results = {}
    for component_name, component in artifacts.components.items():
        try:
            result = prober.compute_patch_text_alignment(
                latents=component.activations,
                concept_prompts=_CLIP_CONCEPT_PROMPTS,
                patch_ids=component.patch_ids,
                component_name=component_name,
                raw_image=artifacts.raw_image,
                grid_size=artifacts.grid_size,
                projector_epochs=projector_epochs,
            )
            results[component_name] = result
            summary = "  ".join(
                f"{concept}={result.concept_scores[concept].mean():.3f}"
                for concept in _CLIP_CONCEPT_PROMPTS
                if concept in result.concept_scores
            )
            print(f"  [patch-semantic] {component_name}: {summary}")
        except Exception as exc:
            print(f"  [patch-semantic] {component_name} failed: {exc}")
    return results or None


def _plot_patch_semantic_alignment(
    patch_semantic_results: dict, artifacts, output_path: Path
) -> None:
    """Grid of per-patch CLIP text similarity heatmaps."""
    import matplotlib.gridspec as gridspec

    component_names = list(patch_semantic_results.keys())
    concepts = list(_CLIP_CONCEPT_PROMPTS.keys())
    n_rows = len(component_names)
    n_cols = len(concepts)

    cell_w, cell_h = 4.0, 4.0
    cbar_w = 0.5
    fig_w = cell_w * n_cols + cbar_w + 0.4
    fig_h = cell_h * n_rows + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))

    # grid: data cells + one narrow colorbar column
    gs = gridspec.GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [cbar_w / (cell_w * n_cols)],
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.03,
        hspace=0.30,
        wspace=0.08,
    )
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(n_rows)])
    cax = fig.add_subplot(gs[:, n_cols])

    max_abs = 0.0
    for result in patch_semantic_results.values():
        for concept in concepts:
            scores = result.concept_scores.get(concept)
            if scores is not None and len(scores) > 0:
                max_abs = max(max_abs, float(np.max(np.abs(scores))))
    if max_abs <= 1e-8:
        max_abs = 1.0

    cmap = plt.get_cmap("coolwarm")
    cmap.set_bad(color="#e8e8e8")  # NaN → light grey

    im_ref = None
    for i, component_name in enumerate(component_names):
        component = artifacts.components[component_name]
        result = patch_semantic_results[component_name]
        for j, concept in enumerate(concepts):
            ax = axes[i, j]
            scores = result.concept_scores.get(concept)
            if scores is None:
                ax.set_visible(False)
                continue

            # NaN for unsampled patches so they render as grey, not neutral blue/red
            heatmap = build_patch_heatmap(
                component.patch_ids, scores, artifacts.grid_size, fill_value=float("nan")
            )

            ax.imshow(artifacts.raw_image.resize((224, 224)), alpha=0.25)
            im_ref = ax.imshow(
                heatmap,
                cmap=cmap,
                vmin=-max_abs,
                vmax=max_abs,
                alpha=0.85,
                extent=(0, 224, 224, 0),
                interpolation="nearest",
            )
            mean_score = float(np.mean(scores))
            ax.set_title(
                f"{component_name.replace('_', ' ').title()}\n"
                f"{_CONCEPT_LABELS[concept]}  mean={mean_score:+.3f}",
                fontsize=8.5,
                fontweight="bold",
                pad=4,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#aaa")
            ax.set_xticks([])
            ax.set_yticks([])

    if im_ref is not None:
        fig.colorbar(im_ref, cax=cax)
        cax.set_ylabel("cosine(pos text) - cosine(neg text)", fontsize=8, labelpad=6)
        cax.tick_params(labelsize=7)

    fig.suptitle(
        "Per-Patch CLIP Text Similarity Maps\n(grey = patches not in component)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wm, adapter, config, checkpoint_path = load_ijepa_world_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    raw_image, metadata = get_sample_image(
        image_path=args.image,
        imagenet_root=args.imagenet_root,
        image_size=getattr(config, "img_size", 224),
    )
    image_tensor = preprocess_image(raw_image, image_size=getattr(config, "img_size", 224))
    artifacts = collect_ijepa_artifacts(wm, adapter, raw_image, image_tensor)
    prober = LatentProber(seed=42)
    component_results = {}
    component_labels = {}
    for component_name, component in artifacts.components.items():
        results, labels = _probe_component(
            prober=prober,
            component_name=component_name,
            activations=component.activations,
            patch_ids=component.patch_ids,
            grid_size=artifacts.grid_size,
            raw_image=raw_image,
        )
        component_results[component_name] = results
        component_labels[component_name] = labels
    from experiment_utils import collect_manual_components_and_attention

    manual_components, _ = collect_manual_components_and_attention(
        adapter,
        image_tensor,
        artifacts.context_ids,
        artifacts.target_ids,
    )
    disentanglement_component_names = sorted(
        name
        for name in manual_components
        if name.startswith("context_encoder.")
        or name.startswith("target_encoder.")
        or name.startswith("predictor.")
    )
    disentanglement_cache = {
        name: manual_components[name].activations for name in disentanglement_component_names
    }
    disentanglement_factors = {
        name: build_disentanglement_factors(
            manual_components[name].patch_ids,
            artifacts.grid_size,
            image_size=getattr(config, "img_size", 224),
        )
        for name in disentanglement_component_names
    }
    disentangler = DisentanglementEvaluationSuite()
    disentanglement_result = disentangler.evaluate_components(
        cache=disentanglement_cache,
        factors=disentanglement_factors,
        components=disentanglement_component_names,
        metrics=["MIG", "DCI", "SAP"],
    )
    # --- Single-image heatmap (all concepts including content labels) ---
    _plot_probe_heatmap(
        component_results,
        output_dir / "ijepa_probe_heatmap.png",
        title="Single-Image Patch Probe Scores (geometric + appearance labels)",
        n_images=1,
    )
    # --- Multi-image aggregate probing ---
    if args.n_images > 1:
        print(f"Building multi-image dataset ({args.n_images} images)...")
        from dataset import build_multi_image_dataset

        dataset = build_multi_image_dataset(wm, adapter, n_synthetic=args.n_images)
        multi_results: dict = {}
        for comp_name, comp_data in dataset.components.items():
            multi_results[comp_name] = _run_probes(
                prober,
                comp_name,
                comp_data.activations,
                comp_data.labels,
                include_content=True,
            )
        _plot_probe_heatmap(
            multi_results,
            output_dir / "ijepa_multi_image_probe_heatmap.png",
            title="Multi-Image Aggregate Patch Probe Scores",
            n_images=dataset.n_images,
        )
        print(f"Multi-image probing done — {dataset.n_patches_total} total patches.")
    print("Running DINO semantic alignment probes...")
    dino_results = _run_dino_semantic_probes(
        artifacts,
        dino_model=args.dino_model,
        projector_epochs=args.semantic_projector_epochs,
    )
    print("Running CLIP semantic alignment probes...")
    semantic_results = _run_semantic_probes(
        artifacts,
        projector_epochs=args.semantic_projector_epochs,
    )
    print("Running per-patch CLIP text similarity probes...")
    patch_semantic_results = _run_patch_semantic_probes(
        artifacts,
        projector_epochs=args.semantic_projector_epochs,
    )
    _plot_probe_overview(
        artifacts=artifacts,
        component_results=component_results,
        component_labels=component_labels,
        output_path=output_dir / "ijepa_probing_overview.png",
        disentanglement_result=disentanglement_result,
    )
    _plot_component_heatmaps(artifacts, output_path=output_dir / "ijepa_component_heatmaps.png")
    if dino_results is not None:
        _plot_dino_alignment(
            dino_results,
            output_path=output_dir / "ijepa_dino_semantic_alignment.png",
        )
    if semantic_results is not None:
        _plot_semantic_alignment(
            semantic_results,
            output_path=output_dir / "ijepa_semantic_alignment.png",
        )
    if patch_semantic_results is not None:
        _plot_patch_semantic_alignment(
            patch_semantic_results,
            artifacts=artifacts,
            output_path=output_dir / "ijepa_patch_text_alignment.png",
        )
    _write_summary(
        artifacts=artifacts,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        component_results=component_results,
        disentanglement_result=disentanglement_result,
        semantic_results=semantic_results,
        dino_results=dino_results,
        output_path=output_dir / "ijepa_probing_summary.txt",
    )
    print("=" * 70)
    print("I-JEPA probing scripts are configured.")
    print(f"checkpoint: {checkpoint_path}")
    print(f"components: {list(artifacts.components.keys())}")
    print(f"outputs: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
