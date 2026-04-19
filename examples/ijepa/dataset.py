"""Multi-image probing dataset builder for I-JEPA."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from utils import PatchLabelBuilder, collect_ijepa_artifacts, preprocess_image


@dataclass
class ComponentProbingData:
    """Aggregated activations and labels for one I-JEPA component across N images."""

    activations: torch.Tensor       # [N_patches_total, D]
    labels: Dict[str, np.ndarray]   # concept -> [N_patches_total]


@dataclass
class MultiImageProbingDataset:
    """Patch activations + labels aggregated across multiple images."""

    components: Dict[str, ComponentProbingData]
    n_images: int
    concept_names: List[str]
    component_names: List[str]

    @property
    def n_patches_total(self) -> int:
        if not self.components:
            return 0
        return next(iter(self.components.values())).activations.shape[0]


# ---------------------------------------------------------------------------
# Synthetic image generator
# ---------------------------------------------------------------------------


def _make_synthetic_images(n: int, image_size: int = 224) -> List[Image.Image]:
    """Generate n varied synthetic PIL images covering different pixel statistics.

    Six image types cycle: angle-gradient, noise, checkerboard, radial,
    smooth blobs, horizontal stripes.
    """
    rng = np.random.default_rng(seed=0)
    images = []

    for i in range(n):
        kind = i % 6

        if kind == 0:
            angle = (i / max(n, 1)) * np.pi
            g = np.linspace(0, 255, image_size, dtype=np.float32)
            xv, yv = np.meshgrid(g, g)
            r  = np.clip(np.cos(angle) * xv + np.sin(angle) * yv, 0, 255)
            gr = np.clip(np.sin(angle) * xv + np.cos(angle) * yv, 0, 255)
            b  = (xv + yv) / 2
            arr = np.stack([r, gr, b], axis=-1).astype(np.uint8)

        elif kind == 1:
            arr = rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8)

        elif kind == 2:
            sz = max(8, 8 + (i // 6) * 4)
            x = np.arange(image_size) // sz
            y = np.arange(image_size) // sz
            xv, yv = np.meshgrid(x, y)
            val = ((xv + yv) % 2 * 200 + 55).astype(np.uint8)
            arr = np.stack([val, val, val], axis=-1)

        elif kind == 3:
            c = image_size // 2
            xs = np.arange(image_size, dtype=np.float32) - c
            ys = np.arange(image_size, dtype=np.float32) - c
            xv, yv = np.meshgrid(xs, ys)
            dist = np.sqrt(xv ** 2 + yv ** 2)
            val = (dist / (dist.max() + 1e-8) * 255).astype(np.uint8)
            hue = ((np.arctan2(yv, xv) + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            arr = np.stack([val, hue, (255 - val).astype(np.uint8)], axis=-1)

        elif kind == 4:
            small = rng.random((image_size // 16, image_size // 16, 3))
            small = ((small - small.min()) / (small.max() - small.min() + 1e-8) * 255).astype(np.uint8)
            resample = getattr(getattr(Image, "Resampling", None), "BILINEAR", Image.BILINEAR)
            arr = np.array(
                Image.fromarray(small).resize((image_size, image_size), resample)
            )

        else:
            sh = max(8, 8 + (i % 8) * 3)
            row_idx = np.arange(image_size)
            val = ((row_idx // sh) % 2 * 200 + 55).astype(np.uint8)
            base = np.tile(val[:, None], (1, image_size))
            shift = (i * 30) % 256
            arr = np.stack([
                base,
                np.roll(base, shift, axis=1).astype(np.uint8),
                (255 - base).astype(np.uint8),
            ], axis=-1)

        images.append(Image.fromarray(arr.astype(np.uint8), mode="RGB"))

    return images


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_multi_image_dataset(
    wm: Any,
    adapter: Any,
    image_paths: Optional[List[Path]] = None,
    n_synthetic: int = 100,
    image_size: int = 224,
    verbose: bool = True,
) -> MultiImageProbingDataset:
    """Build aggregated patch activations + content/geometric labels across N images.

    If image_paths is provided those images are used; otherwise n_synthetic varied
    synthetic images are generated (no ImageNet required).

    Each image contributes one row per patch to every component's activation matrix.
    Labels are computed per-patch per-image via PatchLabelBuilder.build_all().

    Args:
        wm: HookedWorldModel.
        adapter: IJEPAAdapter (must be in eval mode).
        image_paths: Optional list of image Paths.  None → synthetic images.
        n_synthetic: How many synthetic images to generate when image_paths is None.
        image_size: Pixel size used for preprocessing.
        verbose: Print progress every 20 images.

    Returns:
        MultiImageProbingDataset ready to pass to LatentProber.train_probe().
    """
    if image_paths is not None:
        raw_images: List[Image.Image] = [
            Image.open(p).convert("RGB").resize((image_size, image_size))
            for p in image_paths
        ]
    else:
        raw_images = _make_synthetic_images(n_synthetic, image_size)

    acc_acts: Dict[str, List[torch.Tensor]] = {}
    acc_labels: Dict[str, Dict[str, List[np.ndarray]]] = {}
    n_ok = 0

    for idx, raw_image in enumerate(raw_images):
        if verbose and idx % 20 == 0:
            print(f"  [dataset] {idx + 1}/{len(raw_images)} ...")
        try:
            tensor = preprocess_image(raw_image, image_size=image_size)
            artifacts = collect_ijepa_artifacts(wm, adapter, raw_image, tensor)
        except Exception as exc:
            if verbose:
                print(f"  [dataset] image {idx} skipped: {exc}")
            continue

        for comp_name, comp in artifacts.components.items():
            builder = PatchLabelBuilder(comp.patch_ids, artifacts.grid_size, image_size)
            lbl = builder.build_all(raw_image)

            if comp_name not in acc_acts:
                acc_acts[comp_name] = []
                acc_labels[comp_name] = {k: [] for k in lbl}

            acc_acts[comp_name].append(comp.activations.cpu())
            for k, v in lbl.items():
                acc_labels[comp_name][k].append(v)

        n_ok += 1

    if n_ok == 0:
        raise RuntimeError("No images processed successfully — check model / adapter.")

    components: Dict[str, ComponentProbingData] = {
        comp_name: ComponentProbingData(
            activations=torch.cat(acc_acts[comp_name], dim=0),
            labels={k: np.concatenate(acc_labels[comp_name][k]) for k in acc_labels[comp_name]},
        )
        for comp_name in acc_acts
    }

    first = next(iter(components.values()))
    if verbose:
        print(
            f"  [dataset] done — {n_ok} images | "
            f"{first.activations.shape[0]} patches | "
            f"{len(first.labels)} concepts"
        )

    return MultiImageProbingDataset(
        components=components,
        n_images=n_ok,
        concept_names=list(first.labels.keys()),
        component_names=list(components.keys()),
    )
