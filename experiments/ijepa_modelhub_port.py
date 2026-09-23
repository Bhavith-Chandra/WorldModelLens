"""Shared adapters for full-resident, canonical ModelHub I-JEPA experiments.

This module contains no experiment logic. It translates the canonical adapter
returned by ModelHub into the narrow compatibility surface used by the
preserved Task 5/6 experiment implementations.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from world_model_lens.data import load_imagenet_image, load_imagenet_subset
from world_model_lens.hub.model_hub import ModelHub


def _architecture(adapter: Any) -> dict[str, int]:
    encoder = adapter.context_encoder
    predictor = adapter.predictor
    grid = int(encoder.patch_embed.grid_size)
    return {
        "embed_dim": int(encoder.norm.normalized_shape[0]),
        "depth": len(encoder.blocks),
        "num_heads": int(encoder.blocks[0].attn.num_heads),
        "patch_size": int(encoder.patch_embed.patch_size),
        "img_size": int(adapter.config.img_size),
        "num_patches": int(encoder.patch_embed.n_patches),
        "grid_size": grid,
        "predictor_embed_dim": int(predictor.norm.normalized_shape[0]),
        "predictor_depth": len(predictor.blocks),
        "predictor_heads": int(predictor.blocks[0].attn.num_heads),
    }


def _coverage(adapter: Any) -> list[dict[str, Any]]:
    report = getattr(adapter, "checkpoint_coverage", None)
    if not isinstance(report, dict):
        raise RuntimeError("Canonical ModelHub loader did not attach checkpoint coverage")
    return [{"component": name, **entry} for name, entry in report.items()]


def load_modelhub_vith(
    checkpoint: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    slim_dir: str | None = None,
    verbose: bool = True,
) -> Any:
    """Load a full official ViT-H/14 adapter and expose the legacy result shape."""
    if slim_dir is not None and verbose:
        print("[load] --slim_dir is ignored by the full-resident ModelHub port")

    started = time.time()
    adapter = ModelHub.load_checkpoint(checkpoint, backend="ijepa", device=str(device))
    if dtype == torch.float16:
        adapter.half()
    elif dtype == torch.bfloat16:
        adapter.bfloat16()
    elif dtype == torch.float32:
        adapter.float()
    else:
        raise ValueError(f"Unsupported model dtype: {dtype}")

    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    import attribution_patching_vith14 as task5

    architecture = _architecture(adapter)
    coverage = _coverage(adapter)
    metadata = {
        "path": os.path.abspath(checkpoint),
        "file_size_bytes": os.path.getsize(checkpoint),
        "epoch": -1,
        "loss": float("nan"),
        "batch_size": -1,
        "world_size": -1,
        "encoder_params": int(sum(p.numel() for p in adapter.context_encoder.parameters())),
        "predictor_params": int(sum(p.numel() for p in adapter.predictor.parameters())),
        "load_seconds": round(time.time() - started, 1),
        "loader": "ModelHub.load_checkpoint",
    }
    loaded = task5.LoadedIJEPA(
        encoder=adapter.context_encoder,
        predictor=adapter.predictor,
        slim={},
        arch=architecture,
        coverage=coverage,
        checkpoint_meta=metadata,
    )
    loaded.adapter = adapter
    loaded.context_encoder = adapter.context_encoder
    loaded.target_encoder = adapter.target_encoder

    if verbose:
        print(f"[load] canonical ModelHub architecture: {architecture}")
        print(
            "[load] strict checkpoint coverage: "
            + ", ".join(
                f"{row['component']}={row['checkpoint_tensors']}/{row['model_tensors']}"
                for row in coverage
            )
        )
    return loaded


def _manifest_samples(path: Path, count: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("samples", raw.get("images", raw.get("items")))
    if not isinstance(raw, list):
        raise ValueError("Dataset manifest must be a list or contain samples/images/items")

    samples: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            image_path = Path(item)
            record: dict[str, Any] = {"path": str(image_path)}
        elif isinstance(item, dict) and "path" in item:
            record = dict(item)
            image_path = Path(str(record["path"]))
        else:
            raise ValueError("Every manifest entry must be a path or an object with a path")

        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        record["path"] = str(image_path)
        samples.append(record)

    if len(samples) < count:
        raise ValueError(f"Manifest contains {len(samples)} images; experiment needs {count}")
    return samples[:count]


def _balanced_class_count(num_samples: int, limit: int = 50) -> int:
    for candidate in range(min(limit, num_samples), 1, -1):
        if num_samples % candidate == 0:
            return candidate
    return num_samples


def load_project_images(
    data_source: str,
    n: int,
    size: int,
    seed: int,
) -> list[tuple[str, torch.Tensor]]:
    """Adapt a project ImageNet root or JSON manifest to Task 5/6's image API."""
    source = Path(data_source).expanduser()
    if source.is_file():
        samples = _manifest_samples(source.resolve(), n)
    else:
        samples = load_imagenet_subset(
            source,
            num_samples=n,
            num_classes=_balanced_class_count(n),
            seed=seed,
        )

    images: list[tuple[str, torch.Tensor]] = []
    for index, sample in enumerate(samples):
        path = Path(str(sample["path"]))
        name = str(sample.get("name") or f"{index:04d}_{path.name}")
        images.append((name, load_imagenet_image(path, image_size=size).contiguous()))
    return images
