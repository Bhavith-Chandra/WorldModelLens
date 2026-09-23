"""Task 5 with canonical, strict ModelHub checkpoint and project data loading.

The attribution implementation remains in attribution_patching_vith14.py. This
entry point replaces only model loading, encoder-role selection, and image
source adaptation so results can be compared directly with the original script.

The existing --data_dir argument accepts either an ImageNet root or a JSON
manifest containing image paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import attribution_patching_vith14 as task5
from ijepa_modelhub_port import load_modelhub_vith, load_project_images


class ModelHubAttributionPatchingViTH(task5.AttributionPatchingViTH):
    """Preserve Task 5 while selecting resident context/target encoders."""

    def __init__(self, loaded: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(loaded, *args, **kwargs)
        self.context_encoder = loaded.context_encoder
        self.target_encoder = loaded.target_encoder

    def load_role(self, role: str) -> None:
        if role == "target":
            self.encoder = self.target_encoder
        elif role == "context":
            self.encoder = self.context_encoder
        else:
            raise ValueError(f"Unknown I-JEPA encoder role: {role}")


def main() -> None:
    task5.load_official_vith = load_modelhub_vith
    task5.load_images = load_project_images
    task5.AttributionPatchingViTH = ModelHubAttributionPatchingViTH
    task5.main()


if __name__ == "__main__":
    main()
