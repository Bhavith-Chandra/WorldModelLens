"""Task 6 with the full-resident canonical ModelHub I-JEPA adapter.

All latent-lens tracing, statistics, caching, plotting, and mini-model behavior
remain in latent_lens_trajectory.py. Only the official ViT-H model factory and
image-source adapter are replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import latent_lens_trajectory as task6
from ijepa_modelhub_port import load_modelhub_vith, load_project_images


def build_vith_modelhub(
    checkpoint: str,
    device: torch.device,
    slim_dir: Optional[str],
) -> Any:
    """Build Task 6's existing LensModel around a canonical resident adapter."""
    loaded = load_modelhub_vith(
        checkpoint,
        device=device,
        dtype=torch.float32,
        slim_dir=slim_dir,
    )
    adapter = loaded.adapter
    return task6.LensModel(
        predictor=adapter.predictor,
        arch=loaded.arch,
        name="official ViT-H/14 (ModelHub)",
        meta=loaded.checkpoint_meta | {"key_coverage": loaded.coverage},
        _encode_context=lambda image, ids: adapter.context_encoder(
            image.to(device), patch_ids=ids
        ),
        _target_encode=lambda image: adapter.target_encoder(image.to(device)),
        # Both encoders are resident, so the preserved phase calls are no-ops.
        _use_target_role=lambda: None,
        _use_context_role=lambda: None,
    )


def main() -> None:
    task6.build_vith = build_vith_modelhub
    task6.load_images = load_project_images
    task6.main()


if __name__ == "__main__":
    main()
