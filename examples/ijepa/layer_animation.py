"""layer_animation.py — Idea V2 / Roadmap 5C #17.

Renders a GIF (and optional per-frame PNGs) showing how the predictor's
attention attribution over context patches evolves across its layers
for a single target patch. Each frame is a 14x14 spatial heatmap
overlaid on the input image, with the target patch outlined in green.

Reuses orchestrator.ExecutionEngine + SharedLatentState — no second
forward pass per layer; all per-layer attention weights are captured
in a single build_state() call.

Usage:
    python examples/ijepa/layer_animation.py \
        --checkpoint ijepa_mini.pth \
        --target-patch 97 \
        --output layer_animation.gif
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import ExecutionEngine, _load_model, _seeded_masks
from image_utils import get_sample_image, preprocess_image
from world_model_lens.core.config import WorldModelConfig


GRID = 14
PATCH_PX = 16
IMG_PX = GRID * PATCH_PX  # 224


def _heat_grid(weights: np.ndarray, context_ids: List[int]) -> np.ndarray:
    grid = np.zeros((GRID, GRID), dtype=np.float32)
    for w, pid in zip(weights, context_ids):
        grid[pid // GRID, pid % GRID] = w
    m = grid.max()
    if m > 0:
        grid /= m
    return grid


def _frame(
    base: np.ndarray,
    weights: np.ndarray,
    context_ids: List[int],
    target_id: int,
    layer_idx: int,
    n_layers: int,
    title_suffix: str,
) -> Image.Image:
    grid = _heat_grid(weights, context_ids)
    heat_full = np.kron(grid, np.ones((PATCH_PX, PATCH_PX)))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(base)
    ax.imshow(heat_full, cmap="hot", alpha=0.55, vmin=0, vmax=1)

    tx, ty = (target_id % GRID) * PATCH_PX, (target_id // GRID) * PATCH_PX
    ax.add_patch(plt.Rectangle((tx, ty), PATCH_PX, PATCH_PX,
                               fill=False, edgecolor="lime", linewidth=2.5))

    ax.set_title(
        f"Predictor Layer {layer_idx + 1}/{n_layers} — target patch {target_id}"
        f"{title_suffix}",
        fontsize=11,
    )
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_animation(
    image: Optional[Image.Image],
    checkpoint_path: Optional[str],
    target_patch: Optional[int],
    seed: int,
    num_context: int,
    output_gif: str,
    frame_dir: Optional[str],
    duration_ms: int,
) -> str:
    if image is None:
        image = get_sample_image()
    img_tensor = preprocess_image(image)
    base_np = np.array(image.convert("RGB").resize((IMG_PX, IMG_PX)))

    config = WorldModelConfig(
        backend="ijepa", d_embed=192, n_layers=6, n_heads=3,
        predictor_embed_dim=192,
    )
    wm, adapter = _load_model(checkpoint_path, config)

    context_ids, auto_targets = _seeded_masks(seed, num_context=num_context)
    if target_patch is None:
        target_patch = auto_targets[0]
    if target_patch in context_ids:
        context_ids = [c for c in context_ids if c != target_patch]

    engine = ExecutionEngine(adapter, wm)
    state = engine.build_state(img_tensor, context_ids, [target_patch], seed)

    layers = sorted(state.predictor_attn[target_patch].keys())
    n_layers = len(layers)
    ctx_for_target = [c for c in context_ids if c != target_patch]

    print(f"[Animation] {n_layers} layers — target patch {target_patch}")

    frames: List[Image.Image] = []
    for li in layers:
        w = state.predictor_attn[target_patch][li]
        entropy = float(-(w * np.log(w + 1e-9)).sum())
        suffix = f"  |  entropy={entropy:.2f}"
        frame = _frame(base_np, w, ctx_for_target, target_patch,
                       li, n_layers, suffix)
        frames.append(frame)

        if frame_dir:
            os.makedirs(frame_dir, exist_ok=True)
            frame.save(os.path.join(frame_dir, f"layer_{li:02d}.png"))

    if frames:
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"[Animation] Saved GIF → {output_gif}")
    else:
        print("[Animation] No frames produced")

    return output_gif


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-layer predictor attribution as a GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", default=None, help="Path to image (default: sample dog)")
    parser.add_argument("--checkpoint", default="ijepa_mini.pth")
    parser.add_argument("--target-patch", type=int, default=None,
                        help="Target patch ID (0..195). Default: first seeded target.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-context", type=int, default=80)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_animation.gif"))
    parser.add_argument("--frame-dir", default=None,
                        help="If set, also write per-layer PNGs to this directory")
    parser.add_argument("--duration-ms", type=int, default=600,
                        help="Per-frame duration in milliseconds")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB") if args.image else None
    cp = args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None

    build_animation(
        image=img,
        checkpoint_path=cp,
        target_patch=args.target_patch,
        seed=args.seed,
        num_context=args.num_context,
        output_gif=args.output,
        frame_dir=args.frame_dir,
        duration_ms=args.duration_ms,
    )


if __name__ == "__main__":
    main()
