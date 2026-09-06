"""Build a synthetic Task 8 bank solely to exercise the evaluator.

It creates the required source/candidate/metadata layout from ordinary images.
The central target region is replaced with three deliberately simple visual
variants. This is not Stable Diffusion inpainting and its output must never be
used as a scientific ambiguity result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GRID_SIZE = 16
PATCH_SIZE = 14


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Folder of smoke-test images.")
    parser.add_argument("--dest", type=Path, default=Path("data/task8_synthetic_smoke_bank"))
    parser.add_argument("--n-cases", type=int, default=3)
    args = parser.parse_args()
    source_paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(source_paths) < args.n_cases:
        raise ValueError(f"Need {args.n_cases} images below {args.images}; found {len(source_paths)}.")

    # A 6x6 central patch block: patch IDs are row-major in the ViT-H/14 16x16 grid.
    rows = range(5, 11)
    cols = range(5, 11)
    target_ids = [row * GRID_SIZE + col for row in rows for col in cols]
    box = (min(cols) * PATCH_SIZE, min(rows) * PATCH_SIZE, (max(cols) + 1) * PATCH_SIZE, (max(rows) + 1) * PATCH_SIZE)

    for index, source_path in enumerate(source_paths[: args.n_cases], start=1):
        case_dir = args.dest / f"synthetic_case_{index:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        source = Image.open(source_path).convert("RGB").resize((224, 224))
        source.save(case_dir / "source.png")
        crop = source.crop(box)

        candidate_0 = source.copy()
        candidate_0.save(case_dir / "candidate_0.png")

        candidate_1 = source.copy()
        candidate_1.paste(ImageEnhance.Color(crop).enhance(2.0), box)
        candidate_1.save(case_dir / "candidate_1.png")

        candidate_2 = source.copy()
        variant = ImageOps.mirror(crop).filter(ImageFilter.GaussianBlur(radius=1.2))
        candidate_2.paste(variant, box)
        candidate_2.save(case_dir / "candidate_2.png")

        (case_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "target_patch_ids": target_ids,
                    "protocol": "synthetic_smoke_only_not_stable_diffusion",
                    "source": str(source_path),
                },
                indent=2,
            )
        )
    print(
        f"[Task 8 smoke] Built {args.n_cases} synthetic cases at {args.dest}. "
        "Do not use this bank for Task 8 findings.",
        flush=True,
    )


if __name__ == "__main__":
    main()
