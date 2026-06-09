"""dashboard.py — Gradio web dashboard for the I-JEPA interpretability suite.

Wraps `orchestrator.run_full_suite` in a 4-panel UI:
  1. Input + Masking      — uploaded image with context/target patch overlay
  2. Attribution          — top-K attention weights + spatial heatmap
  3. Circuit              — energy-threshold circuit and minimal formal circuit
  4. Causal Validation    — faithfulness ablation/reconstruction AOPC curves

Run:
    python -m examples.ijepa.dashboard
    # or
    python examples/ijepa/dashboard.py --checkpoint ijepa_mini.pth
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "gradio is required for the dashboard. Install with: pip install gradio"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import run_full_suite
from image_utils import get_sample_image


GRID = 14
PATCH_PX = 16
IMG_PX = GRID * PATCH_PX  # 224


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _resize_to_canvas(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((IMG_PX, IMG_PX))


def _patch_xy(patch_id: int) -> Tuple[int, int]:
    return (patch_id % GRID) * PATCH_PX, (patch_id // GRID) * PATCH_PX


def _render_masking_panel(
    img: Image.Image,
    context_ids: List[int],
    target_ids: List[int],
) -> Image.Image:
    base = _resize_to_canvas(img)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.array(base))

    overlay = np.zeros((IMG_PX, IMG_PX, 4), dtype=np.float32)
    for pid in context_ids:
        x, y = _patch_xy(pid)
        overlay[y : y + PATCH_PX, x : x + PATCH_PX] = (0.13, 0.59, 0.95, 0.30)
    for pid in target_ids:
        x, y = _patch_xy(pid)
        overlay[y : y + PATCH_PX, x : x + PATCH_PX] = (1.0, 0.27, 0.23, 0.55)
    ax.imshow(overlay)

    for pid in target_ids:
        x, y = _patch_xy(pid)
        ax.text(
            x + PATCH_PX / 2, y + PATCH_PX / 2, str(pid),
            ha="center", va="center", color="white",
            fontsize=7, fontweight="bold",
        )

    ax.set_title(
        f"Masking — {len(context_ids)} context (blue) / {len(target_ids)} target (red)",
        fontsize=10,
    )
    ax.axis("off")
    return _fig_to_pil(fig)


def _render_attribution_panel(
    img: Image.Image,
    results: Dict[str, Any],
) -> Image.Image:
    attribution = results.get("attribution", {}).get("targets", {})
    if not attribution:
        return _placeholder("Attribution — no data")

    target_keys = list(attribution.keys())
    tid = target_keys[0]
    final = attribution[tid].get("final", {})
    top_patches = final.get("top_context_patches", [])
    top_weights = final.get("top_weights", [])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Bar chart
    ax = axes[0]
    if top_weights:
        ax.bar(range(len(top_weights)), top_weights, color="#2196F3")
        ax.set_xticks(range(len(top_weights)))
        ax.set_xticklabels([f"p{p}" for p in top_patches], rotation=45, fontsize=8)
        ax.set_title(f"Top-K Attribution — Target {tid} (Final Layer)", fontsize=10)
        ax.set_ylabel("Attention Weight")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_title("No attribution weights")

    # Spatial heatmap (final layer attention painted onto patch grid)
    ax = axes[1]
    base = _resize_to_canvas(img)
    ax.imshow(np.array(base))

    heat = np.zeros((GRID, GRID), dtype=np.float32)
    if top_patches and top_weights:
        max_w = max(top_weights) or 1.0
        for pid, w in zip(top_patches, top_weights):
            heat[pid // GRID, pid % GRID] = w / max_w

    heat_full = np.kron(heat, np.ones((PATCH_PX, PATCH_PX)))
    ax.imshow(heat_full, cmap="hot", alpha=0.55, vmin=0, vmax=1)

    for tgt in results.get("metadata", {}).get("target_ids", []):
        x, y = _patch_xy(tgt)
        ax.add_patch(plt.Rectangle((x, y), PATCH_PX, PATCH_PX,
                                   fill=False, edgecolor="lime", linewidth=2))
    ax.set_title("Attribution Heatmap (target = green box)", fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    return _fig_to_pil(fig)


def _render_circuit_panel(results: Dict[str, Any]) -> Image.Image:
    circuit = results.get("circuit", {}).get("per_target", {})
    formal = results.get("formal_circuit", {}).get("per_target", {})
    if not circuit and not formal:
        return _placeholder("Circuit — no data")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Energy-threshold circuit size per target
    ax = axes[0]
    if circuit:
        tids = list(circuit.keys())
        sizes = [circuit[t]["circuit_size"] for t in tids]
        coverages = [circuit[t]["energy_coverage"] for t in tids]
        bars = ax.bar(tids, sizes, color="#4CAF50")
        for bar, cov in zip(bars, coverages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{cov:.1%}",
                ha="center", fontsize=8,
            )
        ax.set_title("Energy-Threshold Circuit (label = energy coverage)", fontsize=10)
        ax.set_xlabel("Target Patch ID")
        ax.set_ylabel("# Circuit Patches")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_title("No circuit data")

    # Formal minimal circuit performance curve
    ax = axes[1]
    if formal:
        first = next(iter(formal.values()))
        curve = first.get("performance_curve", [])
        threshold = first.get("threshold", 0.85)
        if curve:
            ax.plot(range(1, len(curve) + 1), curve, "o-", color="#FF5722")
            ax.axhline(threshold, color="gray", linestyle="--",
                       label=f"threshold = {threshold:.0%}")
            ax.set_title(
                f"Formal Minimal Circuit — Target {next(iter(formal))} "
                f"(K={first['circuit_size']}, sparsity={first['sparsity']:.1%})",
                fontsize=10,
            )
            ax.set_xlabel("Patches Added (attribution order)")
            ax.set_ylabel("Performance")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    else:
        ax.set_title("No formal circuit data")

    plt.tight_layout()
    return _fig_to_pil(fig)


def _render_causal_panel(results: Dict[str, Any]) -> Image.Image:
    faith = results.get("faithfulness", {}).get("per_target", {})
    if not faith:
        return _placeholder("Causal Validation — no data")

    first_key = next(iter(faith))
    f = faith[first_key]
    k_vals = f["ablation"]["k_values"]
    n_top = len(f["ablation"]["top"])
    n_rec = len(f["reconstruction"]["top"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(k_vals[:n_top], f["ablation"]["top"], "r-o",
            markersize=4, label="Top-K removed")
    ax.plot(k_vals[:n_top], f["ablation"]["random"], color="gray",
            linestyle="--", label="Random removed")
    ax.set_title(f"Ablation MSE — Target {first_key}", fontsize=10)
    ax.set_xlabel("Patches Removed")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(k_vals[:n_rec], f["reconstruction"]["top"], "g-o",
            markersize=4, label="Top-K kept")
    ax.plot(k_vals[:n_rec], f["reconstruction"]["random"], color="gray",
            linestyle="--", label="Random kept")
    ax.set_title(
        f"Reconstruction MSE — AOPC = {f['aopc']:.4f}", fontsize=10,
    )
    ax.set_xlabel("Patches Kept")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return _fig_to_pil(fig)


def _placeholder(text: str) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    ax.axis("off")
    return _fig_to_pil(fig)


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def _format_metrics(results: Dict[str, Any]) -> str:
    meta = results.get("metadata", {})
    faith = results.get("faithfulness", {})
    formal = results.get("formal_circuit", {})

    lines = [
        f"**Seed:** {meta.get('seed')}",
        f"**Targets:** {meta.get('target_ids')}",
        f"**Context patches:** {meta.get('context_size')}",
        f"**Elapsed:** {meta.get('elapsed_seconds')}s",
        "",
        f"**Mean AOPC (faithfulness):** {faith.get('mean_aopc', float('nan')):.4f}",
        f"**Mean circuit size:** {formal.get('mean_circuit_size', float('nan')):.1f}",
        f"**Mean sparsity:** {formal.get('mean_sparsity', 0.0):.1%}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

def _run_and_render(
    image: Optional[Image.Image],
    checkpoint_path: str,
    seed: int,
    num_context: int,
    num_targets: int,
):
    if image is None:
        image = get_sample_image()

    output_dir = tempfile.mkdtemp(prefix="ijepa_dashboard_")

    cp = checkpoint_path.strip() if checkpoint_path else ""
    cp_arg = cp if cp and os.path.exists(cp) else None

    results = run_full_suite(
        image=image,
        checkpoint_path=cp_arg,
        seed=int(seed),
        output_dir=output_dir,
        num_context=int(num_context),
    )

    # Trim auto-targets to user-requested count
    meta_targets = results["metadata"]["target_ids"][: max(1, int(num_targets))]
    results["metadata"]["target_ids"] = meta_targets

    context_ids, _ = _seeded_context(int(seed), int(num_context))

    panel1 = _render_masking_panel(image, context_ids, meta_targets)
    panel2 = _render_attribution_panel(image, results)
    panel3 = _render_circuit_panel(results)
    panel4 = _render_causal_panel(results)
    metrics_md = _format_metrics(results)

    json_path = os.path.join(output_dir, "orchestrator_results.json")
    summary_path = results.get("figures", {}).get("summary")

    files = [p for p in [json_path, summary_path] if p and os.path.exists(p)]

    return panel1, panel2, panel3, panel4, metrics_md, files


def _seeded_context(seed: int, num_context: int, num_patches: int = 196) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_patches)
    return sorted(indices[:num_context].tolist()), sorted(indices[num_context:num_context + 3].tolist())


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(default_checkpoint: str = "ijepa_mini.pth") -> gr.Blocks:
    with gr.Blocks(title="I-JEPA Interpretability Dashboard") as demo:
        gr.Markdown(
            "# I-JEPA Interpretability Dashboard\n"
            "Upload an image, configure masking, and run the full orchestrator."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="Input Image", type="pil", height=280)
                checkpoint_in = gr.Textbox(
                    label="Checkpoint Path",
                    value=default_checkpoint,
                    info="Path to .pth (leave default for ijepa_mini.pth)",
                )
                seed_in = gr.Slider(0, 9999, value=42, step=1, label="Seed")
                num_context_in = gr.Slider(20, 160, value=80, step=10, label="# Context Patches")
                num_targets_in = gr.Slider(1, 5, value=2, step=1, label="# Target Patches")
                run_btn = gr.Button("Run Full Suite", variant="primary")
                metrics_out = gr.Markdown(label="Metrics")
                files_out = gr.File(label="Export Report", file_count="multiple")

            with gr.Column(scale=2):
                with gr.Row():
                    panel1 = gr.Image(label="1. Input + Masking", type="pil", height=360)
                    panel2 = gr.Image(label="2. Attribution", type="pil", height=360)
                with gr.Row():
                    panel3 = gr.Image(label="3. Circuit", type="pil", height=360)
                    panel4 = gr.Image(label="4. Causal Validation", type="pil", height=360)

        run_btn.click(
            fn=_run_and_render,
            inputs=[image_in, checkpoint_in, seed_in, num_context_in, num_targets_in],
            outputs=[panel1, panel2, panel3, panel4, metrics_out, files_out],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the I-JEPA interpretability Gradio dashboard.")
    parser.add_argument("--checkpoint", default="ijepa_mini.pth")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(default_checkpoint=args.checkpoint)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
