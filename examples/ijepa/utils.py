from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter

try:
    from torchvision import transforms
except ImportError:
    transforms = None


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
OUTPUT_DIR = EXAMPLE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CHECKPOINT = EXAMPLE_DIR / "ijepa_mini.pth"


@dataclass
class PatchComponent:
    name: str
    activations: torch.Tensor
    patch_ids: list[int]
    description: str


@dataclass
class IJEPARunArtifacts:
    raw_image: Image.Image
    image_tensor: torch.Tensor
    cache: Any
    context_ids: list[int]
    target_ids: list[int]
    grid_size: int
    patch_size: int
    components: dict[str, PatchComponent]
    metadata: dict[str, Any]


def build_ijepa_config(
    *,
    d_embed: int = 192,
    n_layers: int = 6,
    n_heads: int = 3,
    predictor_embed_dim: int = 384,
    predictor_depth: int = 4,
    predictor_heads: int = 6,
    img_size: int = 224,
    patch_size: int = 16,
) -> WorldModelConfig:
    config = WorldModelConfig(
        backend="ijepa",
        d_embed=d_embed,
        n_layers=n_layers,
        n_heads=n_heads,
        predictor_embed_dim=predictor_embed_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
    )
    config.img_size = img_size
    config.patch_size = patch_size
    return config


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
    return payload


def load_ijepa_world_model(
    checkpoint_path: Optional[str | Path] = None,
    config: Optional[WorldModelConfig] = None,
    device: Optional[str | torch.device] = None,
) -> tuple[HookedWorldModel, IJEPAAdapter, WorldModelConfig, Path]:
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"I-JEPA checkpoint not found: {checkpoint}")

    config = config or build_ijepa_config()
    adapter = IJEPAAdapter(config)

    state_dict = _load_checkpoint_state_dict(checkpoint)
    adapter.load_state_dict(state_dict, strict=False)

    if device is not None:
        adapter = adapter.to(torch.device(device))

    wm = HookedWorldModel(adapter=adapter, config=config, name="ijepa_checkpoint")
    adapter.eval()
    return wm, adapter, config, checkpoint


def _find_first_image(root: Path) -> Optional[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    for pattern in patterns:
        hits = sorted(root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def get_sample_image(
    image_path: Optional[str | Path] = None,
    imagenet_root: Optional[str | Path] = None,
    image_size: int = 224,
) -> tuple[Image.Image, dict[str, Any]]:
    metadata: dict[str, Any] = {"source": "unknown"}

    if image_path is not None:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        image = Image.open(path).convert("RGB")
        metadata.update({"source": "image_path", "path": str(path)})
        return image.resize((image_size, image_size)), metadata

    if imagenet_root is not None:
        root = Path(imagenet_root)
        if not root.exists():
            raise FileNotFoundError(f"ImageNet root does not exist: {root}")

        candidate_dirs = [root / "val", root / "train", root]
        for candidate in candidate_dirs:
            if candidate.exists():
                first_image = _find_first_image(candidate)
                if first_image is not None:
                    image = Image.open(first_image).convert("RGB")
                    metadata.update(
                        {
                            "source": "imagenet",
                            "path": str(first_image),
                            "label_name": first_image.parent.name,
                        }
                    )
                    return image.resize((image_size, image_size)), metadata

        raise FileNotFoundError(f"No image files found under ImageNet root: {root}")

    grid = np.linspace(0, 255, image_size, dtype=np.uint8)
    xv, yv = np.meshgrid(grid, grid)
    array = np.stack(
        [
            xv,
            yv,
            ((xv.astype(np.int32) + yv.astype(np.int32)) // 2).astype(np.uint8),
        ],
        axis=-1,
    )
    image = Image.fromarray(array, mode="RGB")
    metadata.update({"source": "synthetic_gradient"})
    return image, metadata


def preprocess_image(raw_image: Image.Image, image_size: int = 224) -> torch.Tensor:
    image = raw_image.convert("RGB").resize((image_size, image_size))
    if transforms is not None:
        pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return pipeline(image).unsqueeze(0)

    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def get_patch_rect(patch_id: int, grid_size: int, image_size: int = 224) -> tuple[int, int, int, int]:
    row, col = divmod(int(patch_id), grid_size)
    patch_extent = image_size // grid_size
    return col * patch_extent, row * patch_extent, patch_extent, patch_extent


def build_patch_labels(patch_ids: list[int], grid_size: int) -> dict[str, np.ndarray]:
    patch_ids_np = np.asarray(patch_ids, dtype=np.int64)
    rows = patch_ids_np // grid_size
    cols = patch_ids_np % grid_size

    center_low = grid_size // 4
    center_high = grid_size - center_low

    return {
        "top_half": (rows < grid_size // 2).astype(np.int64),
        "left_half": (cols < grid_size // 2).astype(np.int64),
        "center_band": ((rows >= center_low) & (rows < center_high)).astype(np.int64),
        "patch_index_norm": patch_ids_np.astype(np.float32) / max(grid_size * grid_size - 1, 1),
    }


class PatchLabelBuilder:
    """Build per-patch labels from grid geometry and raw pixel content.

    Geometric labels require only patch_ids + grid_size.
    Content labels additionally require the raw PIL image.
    """

    def __init__(self, patch_ids: list[int], grid_size: int, image_size: int = 224) -> None:
        self.patch_ids = patch_ids
        self.grid_size = grid_size
        self.image_size = image_size
        self.patch_extent = image_size // grid_size

    def build_geometric(self) -> dict[str, np.ndarray]:
        """Position-based labels — extends the original four with richer splits."""
        ids = np.asarray(self.patch_ids, dtype=np.int64)
        rows = ids // self.grid_size
        cols = ids % self.grid_size
        g = self.grid_size
        c_lo, c_hi = g // 4, g - g // 4

        return {
            "top_half":         (rows < g // 2).astype(np.int64),
            "left_half":        (cols < g // 2).astype(np.int64),
            "center_band":      ((rows >= c_lo) & (rows < c_hi)).astype(np.int64),
            "patch_index_norm": ids.astype(np.float32) / max(g * g - 1, 1),
            "row_norm":         rows.astype(np.float32) / max(g - 1, 1),
            "col_norm":         cols.astype(np.float32) / max(g - 1, 1),
            "quadrant":         (rows >= g // 2).astype(np.int64) * 2
                                + (cols >= g // 2).astype(np.int64),
            "is_boundary":      (
                (rows == 0) | (rows == g - 1) | (cols == 0) | (cols == g - 1)
            ).astype(np.int64),
        }

    def build_content(self, raw_image) -> dict[str, np.ndarray]:
        """Pixel-derived labels computed by cropping each patch from raw_image.

        No external deps — uses only numpy arithmetic.
        Labels:
          brightness      — mean luminance [0, 1]
          edge_density    — mean gradient magnitude (finite diff) [0, 1]
          texture_variance— luminance variance [0, 1]
          saturation      — mean HSV-style saturation [0, 1]
        """
        img = np.array(
            raw_image.resize((self.image_size, self.image_size))
        ).astype(np.float32)   # [H, W, 3], values in [0, 255]

        p = self.patch_extent
        brightness_vals, edge_vals, texture_vals, sat_vals = [], [], [], []

        for patch_id in self.patch_ids:
            row, col = divmod(int(patch_id), self.grid_size)
            y0, x0 = row * p, col * p
            patch = img[y0:y0 + p, x0:x0 + p]   # [p, p, 3]

            lum = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]

            brightness_vals.append(float(lum.mean() / 255.0))
            texture_vals.append(float(lum.var() / (255.0 ** 2)))

            gy = float(np.abs(np.diff(lum, axis=0)).mean())
            gx = float(np.abs(np.diff(lum, axis=1)).mean())
            edge_vals.append((gx + gy) / 255.0)

            cmax = np.maximum(np.maximum(patch[:, :, 0], patch[:, :, 1]), patch[:, :, 2])
            cmin = np.minimum(np.minimum(patch[:, :, 0], patch[:, :, 1]), patch[:, :, 2])
            sat = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0.0)
            sat_vals.append(float(sat.mean()))

        return {
            "brightness":       np.array(brightness_vals, dtype=np.float32),
            "edge_density":     np.array(edge_vals,       dtype=np.float32),
            "texture_variance": np.array(texture_vals,    dtype=np.float32),
            "saturation":       np.array(sat_vals,        dtype=np.float32),
        }

    def build_all(self, raw_image) -> dict[str, np.ndarray]:
        """All labels — geometric + pixel content."""
        return {**self.build_geometric(), **self.build_content(raw_image)}


def build_patch_factors(patch_ids: list[int], grid_size: int) -> dict[str, torch.Tensor]:
    labels = build_patch_labels(patch_ids, grid_size)
    return {name: torch.from_numpy(values.astype(np.float32)) for name, values in labels.items()}


def build_disentanglement_factors(
    patch_ids: list[int],
    grid_size: int,
    image_size: int = 224,
) -> dict[str, torch.Tensor]:
    """Build a richer factor set for disentanglement analysis."""
    labels = PatchLabelBuilder(
        patch_ids=patch_ids,
        grid_size=grid_size,
        image_size=image_size,
    ).build_geometric()
    return {name: torch.from_numpy(values.astype(np.float32)) for name, values in labels.items()}


def build_patch_heatmap(
    patch_ids: list[int],
    values: np.ndarray | torch.Tensor,
    grid_size: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    heatmap = np.full((grid_size, grid_size), fill_value, dtype=np.float32)
    for patch_id, value in zip(patch_ids, values, strict=False):
        row, col = divmod(int(patch_id), grid_size)
        heatmap[row, col] = float(value)
    return heatmap


def collect_ijepa_artifacts(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    raw_image: Image.Image,
    image_tensor: torch.Tensor,
) -> IJEPARunArtifacts:
    # Some adapters in this repo override ``named_parameters`` with a non-standard
    # signature, which makes ``adapter.parameters()`` unsafe. Read the device from
    # a concrete submodule instead.
    model_device = next(adapter.context_encoder.parameters()).device
    image_tensor = image_tensor.to(model_device)
    _, cache = wm.run_with_cache(image_tensor)

    grid_size = adapter.context_encoder.patch_embed.grid_size
    patch_size = adapter.context_encoder.patch_embed.patch_size
    num_patches = adapter.context_encoder.patch_embed.n_patches
    context_ids = list(adapter.last_context_ids or [])
    target_ids = list(adapter.last_target_ids or [])

    components: dict[str, PatchComponent] = {}
    context_latents_device = None

    if "z_posterior" in cache.component_names:
        context_latents_device = cache["z_posterior", 0]
        if context_latents_device.dim() != 2:
            context_latents_device = context_latents_device.reshape(context_latents_device.shape[0], -1)
        components["z_posterior"] = PatchComponent(
            name="z_posterior",
            activations=context_latents_device.detach().cpu(),
            patch_ids=context_ids,
            description="Context-patch latents returned by HookedWorldModel.encode().",
        )

    if "target_encoding" in cache.component_names:
        target_encoding = cache["target_encoding", 0]
        if target_encoding.dim() != 2:
            target_encoding = target_encoding.reshape(target_encoding.shape[0], -1)
        components["target_encoding"] = PatchComponent(
            name="target_encoding",
            activations=target_encoding.detach().cpu(),
            patch_ids=list(range(num_patches)),
            description="Full target-encoder patch embeddings.",
        )

    if context_latents_device is not None and context_ids and target_ids:
        with torch.no_grad():
            predictor_targets = adapter.predictor(
                context_latents_device.unsqueeze(0),
                context_ids,
                target_ids,
            ).squeeze(0)
        components["predictor_targets"] = PatchComponent(
            name="predictor_targets",
            activations=predictor_targets.detach().cpu(),
            patch_ids=target_ids,
            description="Predictor outputs for the currently masked target patches.",
        )

    return IJEPARunArtifacts(
        raw_image=raw_image,
        image_tensor=image_tensor.detach().cpu(),
        cache=cache,
        context_ids=context_ids,
        target_ids=target_ids,
        grid_size=grid_size,
        patch_size=patch_size,
        components=components,
        metadata={},
    )
