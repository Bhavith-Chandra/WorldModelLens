"""Export a small, reproducible Torchvision dataset slice for Task 8 smoke tests.

This is deliberately *not* an ImageNet substitute.  It makes ordinary image
files so ``task8_ijepa_ema_ambiguity.py`` can validate its portable execution
path before licensed ImageNet-1K validation data is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/task8_cifar10_smoke"))
    parser.add_argument("--download-root", type=Path, default=Path("data/torchvision"))
    parser.add_argument("--n-images", type=int, default=32)
    args = parser.parse_args()
    if args.n_images < 2:
        raise ValueError("--n-images must be at least 2 for CKA.")

    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise ImportError("Install torchvision to prepare the Task 8 smoke image set.") from exc

    print("[Task 8 smoke] Downloading/loading Torchvision CIFAR-10 test data...", flush=True)
    dataset = CIFAR10(root=args.download_root, train=False, download=True)
    args.dest.mkdir(parents=True, exist_ok=True)
    for index in range(args.n_images):
        image, label = dataset[index % len(dataset)]
        category_dir = args.dest / f"class_{label:02d}"
        category_dir.mkdir(exist_ok=True)
        image.save(category_dir / f"cifar10_{index:04d}.png")
    print(
        f"[Task 8 smoke] Exported {args.n_images} images to {args.dest}. "
        "Use only for pipeline validation, never as ImageNet-1K evidence.",
        flush=True,
    )


if __name__ == "__main__":
    main()
