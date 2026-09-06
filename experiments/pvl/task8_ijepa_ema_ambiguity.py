"""Task 8: I-JEPA EMA structure and ambiguity evaluation.

This is an evaluation-only script for an official I-JEPA ViT-H/14 checkpoint.
It intentionally uses linear CKA, rather than a pointwise distance, to compare
the deterministic context and target representations.  For the inpainting
experiment it fits regularised Gaussians to *matched triplets* of candidate
inpaintings and reports both differential entropy and KL(I-JEPA || baseline).

Expected inpainting-bank layout::

    inpainting_bank/
      case_001/
        source.png
        candidate_0.png
        candidate_1.png
        candidate_2.png
        metadata.json

``metadata.json`` must contain ``{"target_patch_ids": [..]}`` using the
row-major patch IDs in the ViT-H/14 grid; it may also specify
``"context_patch_ids"``.  The three candidates in each case must use the
same source image and masked region.  The baseline is torchvision's supervised
ViT-H/14, supplied as a local state dict; no checkpoint is downloaded by this
script.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from world_model_lens.hub.model_hub import ModelHub


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LAYER_DEPTHS = (16, 24, 32)  # one-based, matching the manuscript notation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Context encoder + EMA target encoder + predictor + ViT-H/14 baseline are
# resident together in the standard evaluator.  4 GB GPUs cannot fit them.
MIN_CUDA_VRAM_GB = 12


def image_to_tensor(path: Path, image_size: int = 224) -> Tensor:
    """Load an RGB image with the ImageNet normalization used by this repo."""
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return torch.from_numpy(array).permute(2, 0, 1)


class ImageNetSlice(Dataset[Tensor]):
    """Deterministic lexical ImageNet validation set or explicit smoke-test slice."""

    def __init__(self, root: Path, n_samples: int | None, image_size: int = 224):
        paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise FileNotFoundError(f"No images found below {root}")
        self.paths = paths if n_samples is None else paths[:n_samples]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        return image_to_tensor(self.paths[index], self.image_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
class LinearCKAAccumulator:
    """Streaming, globally centered linear CKA accumulator.

    Treats each patch embedding as one observation, avoiding a large
    ``[n_images * n_patches, d]`` allocation for the validation slice.
    """

    def __init__(self, dimension: int, device: torch.device):
        self.n = 0
        self.sum_x = torch.zeros(dimension, device=device, dtype=torch.float64)
        self.sum_y = torch.zeros(dimension, device=device, dtype=torch.float64)
        self.xtx = torch.zeros((dimension, dimension), device=device, dtype=torch.float64)
        self.yty = torch.zeros((dimension, dimension), device=device, dtype=torch.float64)
        self.xty = torch.zeros((dimension, dimension), device=device, dtype=torch.float64)

    def update(self, x: Tensor, y: Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(dtype=torch.float64)
        y = y.reshape(-1, y.shape[-1]).to(dtype=torch.float64)
        if x.shape != y.shape:
            raise ValueError(f"CKA representations must have equal shapes; got {x.shape} and {y.shape}.")
        self.n += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.xtx += x.T @ x
        self.yty += y.T @ y
        self.xty += x.T @ y

    def value(self) -> float:
        if self.n < 2:
            raise ValueError("At least two embedding observations are required for CKA.")
        n = float(self.n)
        xtx = self.xtx - torch.outer(self.sum_x, self.sum_x) / n
        yty = self.yty - torch.outer(self.sum_y, self.sum_y) / n
        xty = self.xty - torch.outer(self.sum_x, self.sum_y) / n
        numerator = xty.square().sum()
        denominator = torch.sqrt(xtx.square().sum() * yty.square().sum()).clamp_min(1e-18)
        return (numerator / denominator).item()


def capture_layers(encoder: nn.Module, images: Tensor, depths: Sequence[int]) -> Dict[int, Tensor]:
    """Return post-block activations for one-based ViT depths."""
    n_blocks = len(encoder.blocks)
    missing = [depth for depth in depths if depth < 1 or depth > n_blocks]
    if missing:
        raise ValueError(f"Requested layers {missing}, but encoder has {n_blocks} blocks.")
    outputs: Dict[int, Tensor] = {}
    handles = []

    def capture(depth: int):
        def hook(_: nn.Module, __: Tuple[object, ...], output: Tensor) -> None:
            outputs[depth] = output.detach()

        return hook

    for depth in depths:
        handles.append(encoder.blocks[depth - 1].register_forward_hook(capture(depth)))
    try:
        with torch.no_grad():
            encoder(images)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def portable_capture_layers(
    encoder: nn.Module, images: Tensor, depths: Sequence[int], device: torch.device
) -> Dict[int, Tensor]:
    """Capture one encoder on GPU, immediately returning its activations to CPU.

    This permits a 4 GB GPU to evaluate a single ViT-H/14 at a time.  It is
    deliberately an exploratory protocol: PCIe transfers make it unsuitable
    for the full ImageNet-scale result.
    """
    encoder.to(device)
    try:
        outputs = capture_layers(encoder, images.to(device), depths)
        return {depth: output.cpu() for depth, output in outputs.items()}
    finally:
        encoder.to("cpu")
        torch.cuda.empty_cache()


def portable_encoder_embeddings(encoder: nn.Module, images: Tensor, device: torch.device) -> Tensor:
    """Run an encoder one image at a time and retain only CPU embeddings."""
    outputs: List[Tensor] = []
    encoder.to(device)
    try:
        with torch.no_grad():
            for image in images:
                outputs.append(encoder(image.unsqueeze(0).to(device)).cpu())
    finally:
        encoder.to("cpu")
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def portable_predictor_embeddings(
    adapter: nn.Module, source: Tensor, context_ids: Sequence[int], target_ids: Sequence[int], device: torch.device
) -> Tensor:
    """Predict target tokens with only one I-JEPA submodule resident on GPU."""
    # Match ``predictor_target_embeddings`` exactly: the context encoder must
    # emit only the visible patch tokens, in the same order as context_ids.
    # Encoding all 256 patches here makes positional embeddings incompatible
    # with the predictor's [n_context, d] input.
    adapter.context_encoder.to(device)
    try:
        with torch.no_grad():
            context = adapter.context_encoder(source.to(device), patch_ids=list(context_ids)).cpu()
    finally:
        adapter.context_encoder.to("cpu")
        torch.cuda.empty_cache()
    adapter.predictor.to(device)
    try:
        with torch.no_grad():
            prediction = adapter.predictor(context.to(device), list(context_ids), list(target_ids)).cpu()
    finally:
        adapter.predictor.to("cpu")
        torch.cuda.empty_cache()
    return prediction


def target_patch_embeddings(adapter: nn.Module, images: Tensor) -> Tensor:
    """Full-image, frozen target-encoder patch embeddings [B, patches, d]."""
    with torch.no_grad():
        return adapter.target_encoder(images)


def predictor_target_embeddings(
    adapter: nn.Module, source: Tensor, context_ids: Sequence[int], target_ids: Sequence[int]
) -> Tensor:
    """Predict target-token embeddings from the fixed visible source context."""
    with torch.no_grad():
        context = adapter.context_encoder(source, patch_ids=list(context_ids))
        return adapter.predictor(context, list(context_ids), list(target_ids))


class TorchvisionViTH14:
    """Frozen supervised ViT-H/14 baseline with patch-token extraction."""

    def __init__(
        self,
        checkpoint: Path | None,
        weights_name: str | None,
        device: torch.device,
    ):
        try:
            from torchvision.models import ViT_H_14_Weights, vit_h_14
        except ImportError as exc:
            raise ImportError(
                "Task 8's supervised ViT-H/14 baseline requires torchvision. "
                "Install it with `pip install torchvision`."
            ) from exc
        if checkpoint is None:
            try:
                weights = ViT_H_14_Weights[weights_name or "IMAGENET1K_SWAG_LINEAR_V1"]
            except KeyError as exc:
                valid = ", ".join(weight.name for weight in ViT_H_14_Weights)
                raise ValueError(f"Unknown torchvision ViT-H/14 weights '{weights_name}'. Choose one of: {valid}") from exc
            self.model = vit_h_14(weights=weights, progress=True).to(device)
        else:
            self.model = vit_h_14(weights=None).to(device)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError("Baseline checkpoint must be a ViT-H/14 state-dict or contain 'state_dict'.")
            state = {key.removeprefix("module."): value for key, value in state.items()}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    "Baseline checkpoint is not compatible with torchvision ViT-H/14; "
                    f"missing={len(missing)}, unexpected={len(unexpected)}."
                )
        self.model.eval()

    def patch_embeddings(self, images: Tensor) -> Tensor:
        """Return final patch tokens; the class token is excluded."""
        with torch.no_grad():
            tokens = self.model._process_input(images)
            cls = self.model.class_token.expand(tokens.shape[0], -1, -1)
            encoded = self.model.encoder(torch.cat([cls, tokens], dim=1))
        return encoded[:, 1:, :]

    def portable_patch_embeddings(self, images: Tensor, device: torch.device) -> Tensor:
        """Run one baseline image at a time for the portable 4 GB protocol."""
        outputs: List[Tensor] = []
        self.model.to(device)
        try:
            for image in images:
                outputs.append(self.patch_embeddings(image.unsqueeze(0).to(device)).cpu())
        finally:
            self.model.to("cpu")
            torch.cuda.empty_cache()
        return torch.cat(outputs, dim=0)


def fit_ridge_alignment(baseline: Tensor, ijepa: Tensor, ridge: float = 1e-4) -> Tensor:
    """Fit B @ W ~= I on validation embeddings to place Gaussian means in one space."""
    if baseline.shape != ijepa.shape:
        raise ValueError(f"Paired baseline/I-JEPA shapes differ: {baseline.shape} vs {ijepa.shape}.")
    baseline = baseline.double()
    ijepa = ijepa.double()
    scale = baseline.square().mean().clamp_min(1e-12)
    eye = torch.eye(baseline.shape[1], dtype=torch.float64)
    return torch.linalg.solve(baseline.T @ baseline + ridge * scale * eye, baseline.T @ ijepa)


@dataclass
class GaussianMetrics:
    entropy: float
    mean: Tensor
    covariance: Tensor


def fit_gaussian(features: Tensor, covariance_ridge: float = 1e-4) -> GaussianMetrics:
    """Fit a stable Gaussian in an already-selected common subspace.

    Three variations cannot identify a 1280-D full covariance.  We therefore
    use the maximum estimable shared rank (K-1) and add a scale-aware ridge.
    This is required for finite entropy and KL values rather than an optional
    numerical convenience.
    """
    if features.ndim != 2 or features.shape[0] < 3:
        raise ValueError("Gaussian fitting requires [K, d] features with K >= 3 variations.")
    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / (features.shape[0] - 1)
    scale = covariance.diagonal().mean().clamp_min(1e-12)
    dimension = features.shape[1]
    covariance = covariance + covariance_ridge * scale * torch.eye(dimension, dtype=torch.float64)
    _, logdet = torch.linalg.slogdet(covariance)
    entropy = 0.5 * (dimension * (1.0 + math.log(2.0 * math.pi)) + logdet.item())
    return GaussianMetrics(entropy=entropy, mean=mean, covariance=covariance)


def shared_gaussian_metrics(ijepa_features: Tensor, baseline_features: Tensor) -> Tuple[GaussianMetrics, GaussianMetrics, float]:
    """Fit both distributions in a shared PCA coordinate system and return KL.

    The baseline features must already be aligned into I-JEPA's coordinate
    system.  KL would otherwise be undefined up to an arbitrary basis change.
    """
    combined = torch.cat([ijepa_features, baseline_features], dim=0).double()
    center = combined.mean(dim=0, keepdim=True)
    _, _, right = torch.linalg.svd(combined - center, full_matrices=False)
    rank = min(ijepa_features.shape[0] - 1, right.shape[0])
    basis = right[:rank].T
    ijepa = fit_gaussian((ijepa_features.double() - center) @ basis)
    baseline = fit_gaussian((baseline_features.double() - center) @ basis)
    inv_baseline = torch.linalg.inv(baseline.covariance)
    delta = baseline.mean - ijepa.mean
    dim = ijepa.mean.numel()
    _, logdet_i = torch.linalg.slogdet(ijepa.covariance)
    _, logdet_b = torch.linalg.slogdet(baseline.covariance)
    kl = 0.5 * (
        torch.trace(inv_baseline @ ijepa.covariance)
        + delta @ inv_baseline @ delta
        - dim
        + logdet_b
        - logdet_i
    )
    return ijepa, baseline, kl.item()


@dataclass
class InpaintingCase:
    identifier: str
    source: Tensor
    candidates: Tensor
    context_ids: List[int]
    target_ids: List[int]


def inpainting_cases(bank: Path, num_patches: int, image_size: int) -> Iterable[InpaintingCase]:
    """Yield validated, predictor-ready ambiguity cases from the inpainting bank."""
    for case_dir in sorted(path for path in bank.iterdir() if path.is_dir()):
        source_paths = sorted(path for path in case_dir.iterdir() if path.stem == "source" and path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(source_paths) != 1:
            raise ValueError(f"{case_dir} must contain exactly one source image named source.<extension>.")
        candidates = sorted(
            path for path in case_dir.iterdir() if path.stem != "source" and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) != 3:
            raise ValueError(f"{case_dir} must contain exactly three candidates (found {len(candidates)}).")
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"{case_dir} requires metadata.json with target_patch_ids.")
        metadata = json.loads(metadata_path.read_text())
        target_ids = sorted({int(index) for index in metadata.get("target_patch_ids", [])})
        if not target_ids or min(target_ids) < 0 or max(target_ids) >= num_patches:
            raise ValueError(f"{case_dir} has invalid target_patch_ids for a {num_patches}-patch ViT grid.")
        context_ids = sorted({int(index) for index in metadata.get("context_patch_ids", [])})
        if not context_ids:
            context_ids = sorted(set(range(num_patches)) - set(target_ids))
        if (
            not context_ids
            or set(context_ids) & set(target_ids)
            or min(context_ids) < 0
            or max(context_ids) >= num_patches
        ):
            raise ValueError(f"{case_dir} has invalid or overlapping context_patch_ids.")
        yield InpaintingCase(
            identifier=case_dir.name,
            source=image_to_tensor(source_paths[0], image_size).unsqueeze(0),
            candidates=torch.stack([image_to_tensor(path, image_size) for path in candidates]),
            context_ids=context_ids,
            target_ids=target_ids,
        )


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def plot_cka(cka: Dict[int, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    layers = list(cka)
    ax.plot(layers, [cka[layer] for layer in layers], marker="o", linewidth=2.4, color="#1f5a94")
    ax.set(xlabel="ViT-H/14 encoder depth", ylabel="Linear CKA", xticks=layers, ylim=(0, 1.02))
    ax.set_title("Context vs. EMA Target Representation Similarity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ambiguity(rows: List[Dict[str, Any]], path: Path) -> None:
    labels = [row["case"] for row in rows]
    x = np.arange(len(rows))
    fig, (entropy_ax, kl_ax, alignment_ax) = plt.subplots(
        1, 3, figsize=(15.6, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]}
    )
    width = 0.36
    entropy_ax.bar(x - width / 2, [row["ijepa_entropy"] for row in rows], width, label="I-JEPA", color="#1f5a94")
    entropy_ax.bar(x + width / 2, [row["baseline_entropy"] for row in rows], width, label="ViT-H/14 baseline", color="#d98942")
    entropy_ax.set(title="Candidate embedding differential entropy", xlabel="Ambiguity case", ylabel="nats", xticks=x, xticklabels=labels)
    entropy_ax.legend(frameon=False)
    entropy_ax.grid(axis="y", alpha=0.25)
    kl_ax.bar(x, [row["kl_ijepa_to_baseline"] for row in rows], color="#7d4e9f")
    kl_ax.axhline(0, color="black", linewidth=0.8)
    kl_ax.set(title="KL(I-JEPA || baseline)", xlabel="Ambiguity case", ylabel="nats", xticks=x, xticklabels=labels)
    kl_ax.grid(axis="y", alpha=0.25)
    alignment_ax.bar(x, [row["mean_predictor_cosine_distance"] for row in rows], color="#3c8d69")
    alignment_ax.set(
        title="I-JEPA predictor-to-candidate alignment",
        xlabel="Ambiguity case",
        ylabel="mean cosine distance",
        xticks=x,
        xticklabels=labels,
    )
    alignment_ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def entropy_bootstrap_summary(rows: List[Dict[str, Any]], n_resamples: int = 1_000) -> Dict[str, object]:
    """Paired bootstrap summary of I-JEPA-minus-baseline entropy.

    Each ambiguity case is the independent unit; the three candidate images
    within a case estimate that case's Gaussian.  A single triplet cannot
    establish statistical significance, so the function reports that fact
    instead of overstating evidence.
    """
    deltas = np.asarray([row["ijepa_entropy"] - row["baseline_entropy"] for row in rows])
    summary: Dict[str, object] = {"n_cases": int(deltas.size), "mean_entropy_delta": float(deltas.mean())}
    if deltas.size < 2:
        summary["note"] = "At least two independent ambiguity cases are required for a bootstrap interval."
        return summary
    rng = np.random.default_rng(0)
    resamples = rng.choice(deltas, size=(n_resamples, deltas.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(resamples, [0.025, 0.975])
    summary["bootstrap_95_ci"] = [float(lower), float(upper)]
    summary["tighter_than_baseline_supported"] = bool(upper < 0.0)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ijepa-checkpoint", type=Path, required=True)
    baseline_source = parser.add_mutually_exclusive_group()
    baseline_source.add_argument("--baseline-checkpoint", type=Path, help="Local torchvision ViT-H/14 state dict.")
    baseline_source.add_argument(
        "--baseline-weights",
        default="IMAGENET1K_SWAG_LINEAR_V1",
        help="Named torchvision ViT-H/14 weights to download/cache (default: IMAGENET1K_SWAG_LINEAR_V1).",
    )
    parser.add_argument("--imagenet-val", type=Path, required=True, help="ImageNet validation ImageFolder root.")
    parser.add_argument("--inpainting-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/pvl/reports/task8"))
    parser.add_argument(
        "--n-validation",
        type=int,
        default=None,
        help="Validation images for CKA/alignment; omit to use the full ImageNet-1K validation set.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50, help="Print validation progress every N batches.")
    parser.add_argument(
        "--portable-4gb",
        action="store_true",
        help=(
            "Exploratory 4 GB-GPU protocol: keep only one ViT-H component on GPU at a time, "
            "force batch size 1, and use at most 64 validation images. Not for reported full-scale results."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.n_validation is not None and args.n_validation < 2:
        raise ValueError("--n-validation must be at least 2.")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive.")
    if not args.ijepa_checkpoint.is_file():
        raise FileNotFoundError(
            f"I-JEPA checkpoint not found: {args.ijepa_checkpoint}\n"
            "Download the official ImageNet-1K ViT-H/14 checkpoint with:\n"
            "  python scripts/download_ijepa_weights.py --dest weights --verify\n"
            "Then pass --ijepa-checkpoint weights/vith14_in1k_ep300.pth.tar"
        )
    if args.baseline_checkpoint is not None and not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"ViT-H/14 baseline checkpoint not found: {args.baseline_checkpoint}")
    validation_images = [
        path for path in args.imagenet_val.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    ] if args.imagenet_val.is_dir() else []
    if not validation_images:
        raise FileNotFoundError(
            f"No supported images found below {args.imagenet_val}.\n"
            "For a portable smoke test, run:\n"
            "  python scripts/download_task8_smoke_images.py --n-images 32\n"
            "and pass --imagenet-val data/task8_cifar10_smoke."
        )
    if not args.inpainting_bank.is_dir():
        raise FileNotFoundError(
            f"Inpainting bank directory not found: {args.inpainting_bank}. "
            "Task 8 requires one source and three structural candidates per case."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but this PyTorch installation has no usable CUDA device.\n"
            "For a smoke test, rerun with --device cpu. A full ViT-H/14 evaluation over "
            "ImageNet-1K requires a CUDA-enabled PyTorch build and an NVIDIA GPU.\n"
            f"torch.version.cuda={torch.version.cuda!r}; torch.cuda.is_available()={torch.cuda.is_available()}"
        )
    if args.portable_4gb:
        if device.type != "cuda":
            raise ValueError("--portable-4gb requires --device cuda.")
        if args.n_validation is None:
            args.n_validation = 32
        if args.n_validation > 64:
            raise ValueError("--portable-4gb supports at most 64 validation images; use the cluster protocol for more.")
        if args.batch_size != 1:
            print("[Task 8] Portable mode forces --batch-size 1.", flush=True)
            args.batch_size = 1
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_vram_gb < MIN_CUDA_VRAM_GB and not args.portable_4gb:
            raise RuntimeError(
                f"Task 8 requires at least {MIN_CUDA_VRAM_GB} GB of CUDA VRAM in its standard "
                f"configuration, but {torch.cuda.get_device_name(device)!r} provides {total_vram_gb:.1f} GB.\n"
                "Use --device cpu only for a tiny smoke test, or run the full ViT-H/14/ImageNet "
                "evaluation on a higher-memory GPU (16 GB+ recommended; 24 GB+ preferred)."
            )
        if total_vram_gb < MIN_CUDA_VRAM_GB and args.portable_4gb:
            print(
                f"[Task 8] Portable mode on {total_vram_gb:.1f} GB VRAM: components will be offloaded "
                "to CPU between forwards. Outputs are exploratory only.",
                flush=True,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    print(f"[Task 8] Loading I-JEPA checkpoint: {args.ijepa_checkpoint}", flush=True)
    model_device = "cpu" if args.portable_4gb else args.device
    adapter = ModelHub._load_ijepa(str(args.ijepa_checkpoint), device=model_device)
    adapter.eval()
    if len(adapter.context_encoder.blocks) < max(LAYER_DEPTHS):
        raise ValueError("Task 8 requires a 32-layer ViT-H/14 I-JEPA checkpoint.")
    baseline_label = str(args.baseline_checkpoint) if args.baseline_checkpoint else args.baseline_weights
    print(f"[Task 8] Loading frozen ViT-H/14 baseline: {baseline_label}", flush=True)
    baseline = TorchvisionViTH14(
        args.baseline_checkpoint, args.baseline_weights, torch.device("cpu") if args.portable_4gb else device
    )
    dataset = ImageNetSlice(args.imagenet_val, args.n_validation, image_size=adapter.config.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(
        f"[Task 8] CKA and alignment calibration on {len(dataset):,} ImageNet validation images "
        f"({len(loader):,} batches; device={device}; portable_4gb={args.portable_4gb}).",
        flush=True,
    )

    cka_accumulators: Dict[int, LinearCKAAccumulator] = {}
    paired_ijepa: List[Tensor] = []
    paired_baseline: List[Tensor] = []
    processed = 0
    accumulator_device = torch.device("cpu") if args.portable_4gb else device
    for batch_index, images in enumerate(loader, start=1):
        if args.portable_4gb:
            context = portable_capture_layers(adapter.context_encoder, images, LAYER_DEPTHS, device)
            target = portable_capture_layers(adapter.target_encoder, images, LAYER_DEPTHS, device)
            target_embeddings = portable_encoder_embeddings(adapter.target_encoder, images, device)
            baseline_embeddings = baseline.portable_patch_embeddings(images, device)
        else:
            images = images.to(device)
            context = capture_layers(adapter.context_encoder, images, LAYER_DEPTHS)
            target = capture_layers(adapter.target_encoder, images, LAYER_DEPTHS)
            target_embeddings = target_patch_embeddings(adapter, images)
            baseline_embeddings = baseline.patch_embeddings(images)
        for depth in LAYER_DEPTHS:
            if depth not in cka_accumulators:
                cka_accumulators[depth] = LinearCKAAccumulator(context[depth].shape[-1], accumulator_device)
            cka_accumulators[depth].update(context[depth], target[depth])
        paired_ijepa.append(target_embeddings.mean(dim=1).cpu())
        paired_baseline.append(baseline_embeddings.mean(dim=1).cpu())
        processed += images.shape[0]
        if batch_index % args.log_every == 0 or batch_index == len(loader):
            print(f"[Task 8] Validation progress: {processed:,}/{len(dataset):,} images.", flush=True)

    cka = {depth: accumulator.value() for depth, accumulator in cka_accumulators.items()}
    print(
        "[Task 8] CKA complete: " + ", ".join(f"L{depth}={score:.5f}" for depth, score in cka.items()),
        flush=True,
    )
    plot_cka(cka, args.output_dir / "cka_depth.pdf")
    print("[Task 8] Fitting baseline-to-I-JEPA representation alignment.", flush=True)
    alignment = fit_ridge_alignment(torch.cat(paired_baseline), torch.cat(paired_ijepa))

    rows: List[Dict[str, Any]] = []
    print("[Task 8] Evaluating predictor-conditioned ambiguity cases.", flush=True)
    num_patches = adapter.context_encoder.patch_embed.n_patches
    for case_index, case in enumerate(
        inpainting_cases(args.inpainting_bank, num_patches, adapter.config.img_size), start=1
    ):
        if args.portable_4gb:
            source = case.source
            candidates = case.candidates
            predicted_targets = portable_predictor_embeddings(
                adapter, source, case.context_ids, case.target_ids, device
            )
            candidate_targets = portable_encoder_embeddings(adapter.target_encoder, candidates, device)[
                :, case.target_ids, :
            ]
            baseline_tokens = baseline.portable_patch_embeddings(candidates, device)
        else:
            source = case.source.to(device)
            candidates = case.candidates.to(device)
            predicted_targets = predictor_target_embeddings(adapter, source, case.context_ids, case.target_ids)
            candidate_targets = target_patch_embeddings(adapter, candidates)[:, case.target_ids, :]
            baseline_tokens = baseline.patch_embeddings(candidates)
        predicted_targets = predicted_targets.expand_as(candidate_targets)
        prediction_mse = (predicted_targets - candidate_targets).square().mean(dim=(1, 2))
        prediction_cosine = 1.0 - F.cosine_similarity(predicted_targets, candidate_targets, dim=-1).mean(dim=1)

        # Gaussian comparison uses only the masked target positions.  Pooling
        # preserves one representation per structural candidate/variation.
        ijepa_features = candidate_targets.mean(dim=1).cpu()
        if baseline_tokens.shape[1] != num_patches:
            raise ValueError("Baseline and I-JEPA must expose the same ViT-H/14 patch grid.")
        baseline_features = baseline_tokens[:, case.target_ids, :].mean(dim=1).cpu().double() @ alignment
        ijepa_gaussian, baseline_gaussian, kl = shared_gaussian_metrics(ijepa_features, baseline_features)
        rows.append(
            {
                "case": case.identifier,
                "ijepa_entropy": ijepa_gaussian.entropy,
                "baseline_entropy": baseline_gaussian.entropy,
                "kl_ijepa_to_baseline": kl,
                "mean_predictor_mse": prediction_mse.mean().item(),
                "mean_predictor_cosine_distance": prediction_cosine.mean().item(),
                "best_candidate_index": int(prediction_cosine.argmin().item()),
                "candidate_predictor_alignment": [
                    {
                        "candidate_index": int(index),
                        "mse": prediction_mse[index].item(),
                        "cosine_distance": prediction_cosine[index].item(),
                    }
                    for index in range(candidates.shape[0])
                ],
            }
        )
        print(
            f"[Task 8] Case {case_index}: {case.identifier}; "
            f"entropy(I-JEPA)={ijepa_gaussian.entropy:.4f}, "
            f"entropy(baseline)={baseline_gaussian.entropy:.4f}, KL={kl:.4f}, "
            f"best candidate={int(prediction_cosine.argmin().item())}",
            flush=True,
        )

    if not rows:
        raise ValueError("The inpainting bank must contain at least one ambiguity case.")
    plot_ambiguity(rows, args.output_dir / "kl_ambiguity_variance.pdf")
    summary = entropy_bootstrap_summary(rows)
    (args.output_dir / "task8_metrics.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "portable_4gb": args.portable_4gb,
                    "n_validation": len(dataset),
                    "batch_size": args.batch_size,
                    "interpretation": (
                        "exploratory pipeline validation; rerun without --portable-4gb on the full "
                        "ImageNet validation set for reportable results"
                        if args.portable_4gb
                        else "full-scale protocol when n_validation is the complete ImageNet validation set",
                    ),
                },
                "cka": cka,
                "ambiguity": rows,
                "entropy_inference": summary,
            },
            indent=2,
        )
    )
    print(f"Saved {args.output_dir / 'cka_depth.pdf'}")
    print(f"Saved {args.output_dir / 'kl_ambiguity_variance.pdf'}")
    print(f"Saved {args.output_dir / 'task8_metrics.json'}")


if __name__ == "__main__":
    main()
