"""Attribution Patching on the pretrained I-JEPA ViT-H/14 (Task 5).

Question
--------
Reviewer Questions 3 & 4 ask whether the causal claims in this repo scale from
the 18 MB mini I-JEPA checkpoints to the *official* 10.36 GB Meta ViT-H/14
release (632 M-parameter encoder, 32 layers x 16 heads). Exhaustive activation
patching does not scale: restoring one component at a time costs one full
forward pass per site, and the ViT-H encoder alone has 32 x 16 = 512 attention
heads plus 96 block-level sites.

Attribution patching (Nanda 2023; Syed et al. 2023) replaces that sweep with a
first-order Taylor expansion of the patching effect. For a site with clean
activation ``a_clean`` and corrupted activation ``a_corr``:

    L(a_clean) - L(a_corr)  ~=  (a_clean - a_corr) . dL/da |_{a_corr}

so *every* site in the network is scored from **two forward passes and one
backward pass**, independent of how many sites there are.

Design
------
Task metric. We use I-JEPA's own objective, so the causal quantity is the
model's real predictive competence rather than a proxy:

    L(run) = - mean_j || predictor(context)_j - target_encoder(x_clean)_j ||^2 / d

evaluated on one target block j of a structured I-JEPA multi-block mask.
Higher L = the predictor still reconstructs the *clean* image's target
representations.

Clean / corrupted runs. The clean run encodes the clean image's context
patches; the corrupted run encodes a *different* image's context patches under
the identical mask, so tokens align 1:1 between runs. L is high on the clean
run and low on the corrupted run; patching asks which components, when restored
from the clean run, recover the clean prediction (the "denoising" direction).

Sites swept (one backward pass covers all of them):
  * ``enc.resid_pre.{l}``   residual stream entering encoder block l
  * ``enc.attn_out.{l}``    attention block output (post out-projection)
  * ``enc.mlp_out.{l}``     MLP block output
  * ``enc.head.{l}.{h}``    per-head slice of z (pre out-projection). Because the
                            out-projection is linear, patching a head's z-slice
                            is exactly patching that head's additive
                            contribution to the residual stream.
  * the same four families on the 12-layer predictor (``pred.*``).

Validation (the point of the experiment). Attribution patching is an
*approximation*. For a subset of sites we also run true activation patching -
substituting the cached clean activation into the corrupted forward pass and
re-evaluating L - and report Pearson / Spearman / sign-agreement and top-k
recovery between the two. That is what licenses using the cheap estimator for
the full ViT-H sweep.

Weight loading. The official checkpoint's key names do not match this repo's
``IJEPAAdapter`` (``mlp.fc1`` vs ``mlp.0``, ``predictor_blocks`` vs ``blocks``,
``predictor_proj`` vs ``predictor_project_back``), and the Meta ViT uses
``qkv_bias=True`` while ``Block`` hardcodes ``qkv_bias=False``. A silent
``strict=False`` load therefore leaves every MLP and every qkv bias at random
initialisation. ``load_official_vith`` remaps the keys, adds the qkv biases and
*asserts* 100% key coverage in both directions; the coverage report is written
into the results JSON.

The 10.36 GB file is read with ``torch.load(..., mmap=True)`` and only the
``encoder`` / ``target_encoder`` / ``predictor`` sub-dicts are materialised, so
peak host RAM stays near one encoder (~2.5 GB) instead of ~11 GB.

Usage
-----
    python scripts/download_ijepa_weights.py --dest checkpoints/
    python experiments/attribution_patching_vith14.py \
        --checkpoint checkpoints/vith14_in1k_ep300.pth.tar \
        --n_pairs 8 --validate \
        --save experiments/results_attribution_patching_vith14.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# A 632 M-parameter encoder leaves little headroom on a small GPU; expandable
# segments keep the backward pass from failing on fragmentation. Not supported
# on Windows, where setting it only produces a warning per allocation.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_model_lens.backends.ijepa_adapter import (  # noqa: E402
    IJEPAAdapter,
    IJEPAPredictor,
    VisionTransformer,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Official checkpoint loading
# ---------------------------------------------------------------------------


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd):
        return dict(sd)
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def _remap_encoder_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Meta I-JEPA ViT key names -> this repo's ``VisionTransformer`` names.

    Only the MLP differs: Meta uses a named ``Mlp`` module (``fc1``/``fc2``),
    this repo uses ``nn.Sequential(Linear, GELU, Linear, Dropout)`` -> ``0``/``2``.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        k = k.replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        out[k] = v
    return out


def _remap_predictor_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Meta ``VisionTransformerPredictor`` names -> this repo's ``IJEPAPredictor``."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("predictor_blocks."):
            k = "blocks." + k[len("predictor_blocks."):]
        elif k == "predictor_pos_embed":
            k = "pos_embed"
        elif k.startswith("predictor_norm."):
            k = "norm." + k[len("predictor_norm."):]
        elif k.startswith("predictor_proj."):
            k = "predictor_project_back." + k[len("predictor_proj."):]
        k = k.replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        out[k] = v
    return out


def _add_qkv_bias(module: nn.Module) -> int:
    """Give every ``attn.qkv`` a zero bias parameter (Meta ViTs use qkv_bias=True).

    ``nn.Linear`` skips the bias term when ``self.bias is None``; assigning a
    Parameter switches it on without reallocating the weight matrix.
    """
    n = 0
    for m in module.modules():
        qkv = getattr(m, "qkv", None)
        if isinstance(qkv, nn.Linear) and qkv.bias is None:
            qkv.bias = nn.Parameter(
                torch.zeros(qkv.out_features, device=qkv.weight.device, dtype=qkv.weight.dtype)
            )
            n += 1
    return n


@dataclass
class LoadedIJEPA:
    encoder: VisionTransformer
    predictor: IJEPAPredictor
    slim: Dict[str, str]
    arch: Dict[str, int]
    coverage: List[Dict[str, Any]]
    checkpoint_meta: Dict[str, Any]


def slim_paths(checkpoint: str, out_dir: Optional[str] = None) -> Dict[str, str]:
    """Where the per-component extracts of ``checkpoint`` live."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(checkpoint))
    stem = os.path.basename(checkpoint).replace(".pth.tar", "").replace(".pth", "")
    return {
        role: os.path.join(out_dir, f"{stem}.{role}.pt")
        for role in ("encoder", "target_encoder", "predictor")
    }


def ensure_slim_checkpoints(
    checkpoint: str, out_dir: Optional[str] = None, verbose: bool = True
) -> Dict[str, str]:
    """Split the official checkpoint into one already-remapped file per component.

    Why this exists: half of the 10.36 GB file is optimiser state we never use,
    and holding a mapping that large is actively harmful on Windows. A live
    private mapping charges against the system commit limit that the WDDM driver
    draws on, which caps CUDA allocations at ~1.7 GB no matter what
    ``mem_get_info`` reports, and copying out of it faults once host RAM runs
    low. Mapping a 2.5 GB single-component file instead avoids both.

    The extracts also carry this repo's key names, so the remapping is done once
    rather than on every run.
    """
    paths = slim_paths(checkpoint, out_dir)
    if all(os.path.exists(p) for p in paths.values()):
        return paths

    if verbose:
        print(f"[slim] first run: extracting components from {os.path.basename(checkpoint)}")
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    except Exception as exc:  # legacy (non-zipfile) pickles cannot be mmapped
        print(f"[slim] mmap unavailable ({exc}); falling back to a full read")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    remap = {
        "encoder": _remap_encoder_keys,
        "target_encoder": _remap_encoder_keys,
        "predictor": _remap_predictor_keys,
    }
    for role, path in paths.items():
        if os.path.exists(path):
            continue
        state = remap[role](_strip_prefix(ckpt[role]))
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        del state
        gc.collect()
        if verbose:
            print(f"[slim]   {os.path.basename(path)}  {os.path.getsize(path)/1e9:.2f} GB")

    meta_path = os.path.join(os.path.dirname(paths["encoder"]), "checkpoint_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(
            {
                "source": os.path.abspath(checkpoint),
                "source_bytes": os.path.getsize(checkpoint),
                "epoch": int(ckpt.get("epoch", -1)),
                "loss": float(ckpt.get("loss", float("nan"))),
                "batch_size": int(ckpt.get("batch_size", -1)),
                "world_size": int(ckpt.get("world_size", -1)),
            },
            fh,
            indent=2,
        )
    del ckpt
    gc.collect()
    return paths


def open_slim_state(path: str) -> Dict[str, torch.Tensor]:
    """Memory-map one component extract. Drop the result before allocating."""
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def copy_state_into(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Copy ``state`` into ``model``'s existing tensors, one at a time.

    ``nn.Module.load_state_dict`` does the same ``copy_`` but materialises its
    bookkeeping over the whole dict first; on Windows that faulted when copying
    2.5 GB out of a mapping under memory pressure. Copying explicitly, with a
    collection every so often, keeps the resident set flat.
    """
    own = dict(model.state_dict())
    missing = set(own) - set(state)
    unexpected = set(state) - set(own)
    if missing or unexpected:
        raise RuntimeError(
            f"state dict does not match the module: missing={sorted(missing)[:6]} "
            f"unexpected={sorted(unexpected)[:6]}"
        )
    with torch.no_grad():
        for i, (k, v) in enumerate(state.items()):
            if tuple(own[k].shape) != tuple(v.shape):
                raise RuntimeError(f"shape mismatch at {k}: {own[k].shape} vs {v.shape}")
            own[k].copy_(v)
            if i % 64 == 63:
                gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_official_vith(
    checkpoint: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    slim_dir: Optional[str] = None,
    verbose: bool = True,
) -> LoadedIJEPA:
    """Load the official Meta I-JEPA ViT-H/14 checkpoint into repo modules.

    Ordering matters on a small GPU: the architecture is read from a mapping,
    the mapping is dropped, the 2.5 GB encoder is allocated on the device, and
    only then is a (much smaller) mapping reopened to copy weights in place. No
    mapping is live while the sweep runs, so the backward pass gets the whole
    remaining device budget.

    A single encoder instance serves both roles: the EMA target-encoder weights
    are loaded to compute prediction targets, then the context-encoder weights
    are loaded over them (see :meth:`AttributionPatchingViTH.load_role`).
    """
    t0 = time.time()
    slim = ensure_slim_checkpoints(checkpoint, slim_dir, verbose=verbose)

    # ---- architecture, inferred from the tensors themselves ----------------
    enc_state = open_slim_state(slim["encoder"])
    prd_state = open_slim_state(slim["predictor"])

    embed_dim, _, patch_size, _ = enc_state["patch_embed.proj.weight"].shape
    depth = sum(1 for k in enc_state if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    num_patches = enc_state["pos_embed"].shape[1]
    grid = int(round(math.sqrt(num_patches)))
    img_size = grid * patch_size
    qkv_out = enc_state["blocks.0.attn.qkv.weight"].shape[0]
    assert qkv_out == 3 * embed_dim, f"unexpected qkv shape {qkv_out} for embed_dim {embed_dim}"
    num_heads = embed_dim // 80 if embed_dim % 80 == 0 else 16  # ViT-H: 1280/16 = 80

    pred_dim = prd_state["predictor_embed.weight"].shape[0]
    pred_depth = sum(1 for k in prd_state if k.startswith("blocks.") and k.endswith(".norm1.weight"))

    arch = dict(
        embed_dim=int(embed_dim),
        depth=int(depth),
        num_heads=int(num_heads),
        patch_size=int(patch_size),
        img_size=int(img_size),
        num_patches=int(num_patches),
        grid_size=int(grid),
        predictor_embed_dim=int(pred_dim),
        predictor_depth=int(pred_depth),
        predictor_heads=int(num_heads),
    )
    enc_keys, prd_keys = set(enc_state), set(prd_state)
    tgt_keys = set(open_slim_state(slim["target_encoder"]))
    del enc_state, prd_state
    gc.collect()
    if verbose:
        print(f"[load] inferred architecture: {arch}")

    # ---- allocate on the device with no mapping live ------------------------
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with torch.device(device):
        encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        predictor = IJEPAPredictor(
            encoder_embed_dim=embed_dim,
            predictor_embed_dim=pred_dim,
            depth=pred_depth,
            num_heads=num_heads,
            num_patches=num_patches,
        )
    torch.set_default_dtype(prev_dtype)
    n_enc_bias = _add_qkv_bias(encoder)
    n_prd_bias = _add_qkv_bias(predictor)
    if verbose:
        print(f"[load] enabled qkv_bias on {n_enc_bias} encoder / {n_prd_bias} predictor blocks")

    # ---- key coverage: every official tensor must land somewhere ------------
    model_enc_keys = set(encoder.state_dict())
    model_prd_keys = set(predictor.state_dict())
    coverage = []
    for what, ckpt_keys, model_keys in (
        ("context_encoder", enc_keys, model_enc_keys),
        ("target_encoder", tgt_keys, model_enc_keys),
        ("predictor", prd_keys, model_prd_keys),
    ):
        diff = sorted(ckpt_keys.symmetric_difference(model_keys))
        if diff:
            raise RuntimeError(
                f"[{what}] official checkpoint did not map cleanly onto the module: "
                f"{diff[:8]} - refusing to run a causal experiment on "
                "partially-initialised weights."
            )
        coverage.append(
            {
                "component": what,
                "checkpoint_tensors": len(ckpt_keys),
                "model_tensors": len(model_keys),
                "missing": [],
                "unexpected": [],
            }
        )

    # ---- copy weights in place (no new device allocations here) -------------
    state = open_slim_state(slim["encoder"])
    copy_state_into(encoder, state)
    del state
    gc.collect()
    state = open_slim_state(slim["predictor"])
    copy_state_into(predictor, state)
    del state
    gc.collect()

    meta_path = os.path.join(os.path.dirname(slim["encoder"]), "checkpoint_meta.json")
    raw_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta = {
        "path": os.path.abspath(checkpoint),
        "file_size_bytes": os.path.getsize(checkpoint),
        "epoch": raw_meta.get("epoch", -1),
        "loss": raw_meta.get("loss", float("nan")),
        "batch_size": raw_meta.get("batch_size", -1),
        "world_size": raw_meta.get("world_size", -1),
        "encoder_params": int(sum(p.numel() for p in encoder.parameters())),
        "predictor_params": int(sum(p.numel() for p in predictor.parameters())),
        "load_seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(
            f"[load] encoder {meta['encoder_params']/1e6:.1f} M params, predictor "
            f"{meta['predictor_params']/1e6:.1f} M params, epoch={meta['epoch']} "
            f"({meta['load_seconds']:.1f}s)"
        )

    encoder.eval()
    predictor.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)

    return LoadedIJEPA(
        encoder=encoder,
        predictor=predictor,
        slim=slim,
        arch=arch,
        coverage=coverage,
        checkpoint_meta=meta,
    )



# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_images(data_dir: str, n: int, size: int, seed: int) -> List[Tuple[str, torch.Tensor]]:
    """Load and ImageNet-normalise ``n`` images from ``data_dir`` (sorted, seeded)."""
    from PIL import Image

    paths = []
    for root, _dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                paths.append(os.path.join(root, f))
    if not paths:
        raise RuntimeError(f"no images found under {data_dir}")
    paths.sort()
    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:n]

    out = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        out.append((os.path.basename(p), t))
    return out


def sample_masks(num_patches: int, grid_size: int, seed: int) -> Tuple[List[int], List[int]]:
    """Sample I-JEPA structured multi-block masks, reusing the library sampler.

    ``IJEPAAdapter._get_structured_masks`` touches no instance state, so we call
    it on an uninitialised instance rather than paying for a second ViT-H.
    """
    random.seed(seed)
    dummy = IJEPAAdapter.__new__(IJEPAAdapter)
    context_ids, target_blocks = dummy._get_structured_masks(num_patches, grid_size)
    target_blocks = sorted(target_blocks, key=len, reverse=True)
    target_ids = sorted(target_blocks[0])  # single largest block: one predictor pass
    return sorted(context_ids), target_ids


# ---------------------------------------------------------------------------
# Hook plumbing: capture / patch every site in one pass
# ---------------------------------------------------------------------------


class SiteHooks:
    """Registers capture-and-patch hooks over an encoder / predictor stack.

    Site names are ``{prefix}.{family}.{layer}`` (and ``{prefix}.head.{l}.{h}``).
    A single object serves three modes:
      * ``mode="capture"``   store activations (used for the clean run)
      * ``mode="grad"``      store activations *and* register grad hooks
      * ``mode="patch"``     substitute ``patch_value`` at ``patch_site``
    """

    def __init__(self, blocks: nn.ModuleList, prefix: str, num_heads: int):
        self.blocks = blocks
        self.prefix = prefix
        self.num_heads = num_heads
        self.handles: List[Any] = []
        self.mode = "off"
        self.acts: Dict[str, torch.Tensor] = {}
        self.to_cpu = False  # clean-run caches live on host RAM, not the 4 GB GPU
        self.on_grad: Optional[Callable[[str, torch.Tensor], None]] = None
        self.patch_site: Optional[str] = None
        self.patch_value: Optional[torch.Tensor] = None

    # -- names -----------------------------------------------------------
    def resid_pre(self, l: int) -> str:
        return f"{self.prefix}.resid_pre.{l}"

    def attn_out(self, l: int) -> str:
        return f"{self.prefix}.attn_out.{l}"

    def mlp_out(self, l: int) -> str:
        return f"{self.prefix}.mlp_out.{l}"

    def z(self, l: int) -> str:
        return f"{self.prefix}.z.{l}"

    def head(self, l: int, h: int) -> str:
        return f"{self.prefix}.head.{l}.{h}"

    def block_sites(self) -> List[str]:
        names = []
        for l in range(len(self.blocks)):
            names += [self.resid_pre(l), self.attn_out(l), self.mlp_out(l)]
        return names

    def head_sites(self) -> List[str]:
        return [self.head(l, h) for l in range(len(self.blocks)) for h in range(self.num_heads)]

    # -- hook bodies -----------------------------------------------------
    def _store(self, name: str, tensor: torch.Tensor) -> None:
        act = tensor.detach()
        self.acts[name] = act.to("cpu", copy=True) if self.to_cpu else act
        if self.mode == "grad" and tensor.requires_grad and self.on_grad is not None:
            tensor.register_hook(lambda g, n=name: self.on_grad(n, g))

    def _handle(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode == "patch":
            if name == self.patch_site and self.patch_value is not None:
                return self.patch_value.to(device=tensor.device, dtype=tensor.dtype)
            return tensor
        if self.mode in ("capture", "grad"):
            self._store(name, tensor)
        return tensor

    def _handle_z(self, layer: int, tensor: torch.Tensor) -> torch.Tensor:
        """z is [B, N, C]; head h owns channels [h*hd:(h+1)*hd]."""
        if self.mode == "patch" and self.patch_site is not None:
            parts = self.patch_site.split(".")
            if parts[-3] == "head" and int(parts[-2]) == layer:
                h = int(parts[-1])
                hd = tensor.shape[-1] // self.num_heads
                out = tensor.clone()
                out[..., h * hd:(h + 1) * hd] = self.patch_value.to(
                    device=tensor.device, dtype=tensor.dtype
                )
                return out
            return tensor
        if self.mode in ("capture", "grad"):
            self._store(self.z(layer), tensor)
        return tensor

    def register(self) -> None:
        self.remove()
        for l, blk in enumerate(self.blocks):
            self.handles.append(
                blk.register_forward_pre_hook(
                    lambda m, inp, l=l: (self._handle(self.resid_pre(l), inp[0]),) + tuple(inp[1:])
                )
            )
            self.handles.append(
                blk.attn.register_forward_hook(
                    lambda m, inp, out, l=l: self._handle(self.attn_out(l), out)
                )
            )
            self.handles.append(
                blk.mlp.register_forward_hook(
                    lambda m, inp, out, l=l: self._handle(self.mlp_out(l), out)
                )
            )
            self.handles.append(
                blk.attn.hook_z.register_forward_hook(
                    lambda m, inp, out, l=l: self._handle_z(l, out)
                )
            )

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self) -> None:
        self.acts = {}

    # -- modes -----------------------------------------------------------
    def set_capture(self, to_cpu: bool = False) -> None:
        self.mode = "capture"
        self.to_cpu = to_cpu
        self.patch_site = None

    def set_grad(self, on_grad: Callable[[str, torch.Tensor], None]) -> None:
        self.mode = "grad"
        self.to_cpu = False
        self.on_grad = on_grad
        self.patch_site = None

    def set_patch(self, site: str, value: torch.Tensor) -> None:
        self.mode = "patch"
        self.patch_site = site
        self.patch_value = value

    def set_off(self) -> None:
        self.mode = "off"
        self.patch_site = None
        self.patch_value = None


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    clean_name: str
    corrupt_name: str
    n_context: int
    n_target: int
    L_clean: float
    L_corrupt: float
    ap_scores: Dict[str, Dict[str, float]]   # estimator -> site -> score
    true_scores: Dict[str, float]
    ap_seconds: Dict[str, float]
    true_seconds: float
    true_forwards: int


class AttributionPatchingViTH:
    def __init__(
        self,
        loaded: LoadedIJEPA,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.loaded = loaded
        self.device = device
        self.dtype = dtype
        self.arch = loaded.arch
        self.encoder = loaded.encoder
        self.predictor = loaded.predictor

        self.enc_hooks = SiteHooks(self.encoder.blocks, "enc", self.arch["num_heads"])
        self.pred_hooks = SiteHooks(self.predictor.blocks, "pred", self.arch["predictor_heads"])
        self.enc_hooks.register()
        self.pred_hooks.register()

    # -- weight swapping so one encoder instance serves both roles ---------
    def load_role(self, role: str) -> None:
        """Load ``'target'`` (EMA) or ``'context'`` weights into the encoder.

        ``load_state_dict`` copies tensor-by-tensor into the already-resident
        parameters, so this never moves the 2.5 GB encoder off the device.
        """
        key = "target_encoder" if role == "target" else "encoder"
        state = open_slim_state(self.loaded.slim[key])
        copy_state_into(self.encoder, state)
        del state
        gc.collect()

    # -- forward halves ----------------------------------------------------
    def encode_context(self, img: torch.Tensor, context_ids: List[int]) -> torch.Tensor:
        return self.encoder(img.to(self.device, dtype=self.dtype), patch_ids=context_ids)

    def predict(
        self, latents: torch.Tensor, context_ids: List[int], target_ids: List[int]
    ) -> torch.Tensor:
        return self.predictor(latents, context_ids, target_ids)

    @staticmethod
    def metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative per-dimension MSE: the I-JEPA objective, sign-flipped."""
        return -((pred - target) ** 2).mean()

    # -- one clean/corrupt pair -------------------------------------------
    def prepare_pair(
        self,
        img_clean: torch.Tensor,
        img_corrupt: torch.Tensor,
        context_ids: List[int],
        target_ids: List[int],
        target_reps: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, float]:
        """Cache the two endpoint runs.

        Returns ``(clean_acts, deltas, L_clean, L_corrupt)`` with every tensor on
        the host, so the device keeps its whole budget for the backward pass.
        ``deltas[site] = a_clean - a_corrupt`` is the displacement that patching
        would apply, and is fixed for every estimator.
        """
        acts: Dict[str, Dict[str, torch.Tensor]] = {}
        losses: Dict[str, float] = {}
        for label, img in (("clean", img_clean), ("corrupt", img_corrupt)):
            self.enc_hooks.clear()
            self.pred_hooks.clear()
            self.enc_hooks.set_capture(to_cpu=True)
            self.pred_hooks.set_capture(to_cpu=True)
            with torch.no_grad():
                lat = self.encode_context(img, context_ids)
                pred = self.predict(lat, context_ids, target_ids)
                L = float(self.metric(pred, target_reps).item())
            acts[label] = {**self.enc_hooks.acts, **self.pred_hooks.acts}
            losses[label] = L
            del lat, pred
            self.enc_hooks.clear()
            self.pred_hooks.clear()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        clean_acts, corrupt_acts = acts["clean"], acts["corrupt"]
        deltas = {k: (clean_acts[k].float() - corrupt_acts[k].float()) for k in clean_acts}
        del corrupt_acts, acts
        gc.collect()
        return clean_acts, deltas, losses["clean"], losses["corrupt"]

    def attribution_scores(
        self,
        deltas: Dict[str, torch.Tensor],
        img_clean: torch.Tensor,
        img_corrupt: torch.Tensor,
        context_ids: List[int],
        target_ids: List[int],
        target_reps: torch.Tensor,
        ig_steps: int = 1,
    ) -> Dict[str, float]:
        """Score every site at once from ``ig_steps`` backward passes.

        ``ig_steps == 1`` is plain attribution patching: the gradient is taken at
        the corrupted run and dotted with the displacement, i.e. a first-order
        Taylor expansion around a single point.

        ``ig_steps > 1`` averages the gradient along the straight line from the
        corrupted input to the clean one (integrated-gradient attribution
        patching, Hanna et al. 2024). The displacement is unchanged; only the
        gradient it multiplies becomes a path average, which is what makes the
        estimate survive the large perturbation a cross-image corruption applies.
        """
        scores: Dict[str, float] = {}
        alphas = [0.0] if ig_steps <= 1 else [(i + 0.5) / ig_steps for i in range(ig_steps)]
        weight = 1.0 / len(alphas)

        def on_grad(name: str, grad: torch.Tensor) -> None:
            hooks = self.enc_hooks if name.startswith("enc.") else self.pred_hooks
            delta = deltas[name].to(grad.device, torch.float32)
            prod = delta * grad.float()
            if ".z." in name:
                layer = int(name.split(".")[-1])
                per_head = prod.reshape(
                    prod.shape[0], prod.shape[1], hooks.num_heads, -1
                ).sum(dim=(0, 1, 3))
                for h in range(hooks.num_heads):
                    key = hooks.head(layer, h)
                    scores[key] = scores.get(key, 0.0) + weight * float(per_head[h].item())
            else:
                scores[name] = scores.get(name, 0.0) + weight * float(prod.sum().item())

        self.enc_hooks.set_grad(on_grad)
        self.pred_hooks.set_grad(on_grad)
        x_clean = img_clean.to(self.device, dtype=self.dtype)
        x_corrupt = img_corrupt.to(self.device, dtype=self.dtype)
        with torch.enable_grad():
            for alpha in alphas:
                x = (x_corrupt + alpha * (x_clean - x_corrupt)).detach().requires_grad_(True)
                lat = self.encoder(x, patch_ids=context_ids)
                pred = self.predict(lat, context_ids, target_ids)
                L = self.metric(pred, target_reps)
                L.backward()
                del x, lat, pred, L
                self.enc_hooks.clear()
                self.pred_hooks.clear()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        self.enc_hooks.set_off()
        self.pred_hooks.set_off()
        self.enc_hooks.clear()
        self.pred_hooks.clear()
        return scores

    # -- ground-truth activation patching ----------------------------------
    def patch_site(
        self,
        site: str,
        clean_acts: Dict[str, torch.Tensor],
        img_corrupt: torch.Tensor,
        context_ids: List[int],
        target_ids: List[int],
        target_reps: torch.Tensor,
    ) -> float:
        """L after substituting the clean activation at ``site`` into the corrupt run."""
        prefix = site.split(".")[0]
        hooks = self.enc_hooks if prefix == "enc" else self.pred_hooks
        other = self.pred_hooks if prefix == "enc" else self.enc_hooks

        parts = site.split(".")
        if parts[1] == "head":
            l, h = int(parts[2]), int(parts[3])
            z = clean_acts[hooks.z(l)]
            hd = z.shape[-1] // hooks.num_heads
            value = z[..., h * hd:(h + 1) * hd]
        else:
            value = clean_acts[site]

        hooks.set_patch(site, value)
        other.set_off()
        with torch.no_grad():
            lat = self.encode_context(img_corrupt, context_ids)
            pred = self.predict(lat, context_ids, target_ids)
            L = float(self.metric(pred, target_reps).item())
        hooks.set_off()
        return L


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        return r

    return pearson(rank(a), rank(b))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a))
    if k == 0:
        return float("nan")
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--checkpoint", default="checkpoints/vith14_in1k_ep300.pth.tar")
    ap_.add_argument("--data_dir", default="data/mini_train")
    ap_.add_argument("--n_pairs", type=int, default=8)
    ap_.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap_.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap_.add_argument("--slim_dir", default=None,
                     help="where per-component checkpoint extracts are cached")
    ap_.add_argument("--strict_device", action="store_true", help="fail instead of CPU fallback")
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--validate", action="store_true", help="run ground-truth activation patching")
    ap_.add_argument("--validate_pairs", type=int, default=3)
    ap_.add_argument("--validate_heads", type=int, default=32, help="heads per pair (top + random)")
    ap_.add_argument("--ig_steps", type=int, default=8,
                     help="path steps for integrated-gradient attribution patching (1 disables it)")
    ap_.add_argument("--save", default="experiments/results_attribution_patching_vith14.json")
    ap_.add_argument("--plots", default="experiments/plots")
    args = ap_.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[args.dtype]

    print("=" * 78)
    print("Task 5 - Attribution Patching on pretrained I-JEPA ViT-H/14")
    print("=" * 78)

    # A ViT-H encoder needs ~2.5 GB (fp32) of weights plus room for the
    # backward graph. Falling back beats dying half-way through a sweep.
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        itemsize = torch.empty(0, dtype=dtype).element_size()
        # ~655 M parameters, plus the corrupted run's backward graph. The clean
        # activation cache lives on the host, so it is not counted here.
        needed = 655e6 * itemsize * 1.02 + 0.6e9
        print(f"[gpu] {torch.cuda.get_device_name(0)}: {free/1e9:.2f} GB free of "
              f"{total/1e9:.2f} GB, need ~{needed/1e9:.2f} GB")
        if free < needed:
            if args.strict_device:
                raise RuntimeError("not enough free VRAM; pass --device cpu or free the GPU")
            print("[gpu] insufficient free VRAM (another process is holding it) "
                  "-> falling back to CPU")
            device = torch.device("cpu")

    loaded = load_official_vith(
        args.checkpoint, device=device, dtype=dtype, slim_dir=args.slim_dir
    )
    arch = loaded.arch
    runner = AttributionPatchingViTH(loaded, device, dtype)

    images = load_images(args.data_dir, args.n_pairs + 1, arch["img_size"], args.seed)
    if len(images) < 2:
        raise RuntimeError("need at least two images")
    print(f"[data] {len(images)} images from {args.data_dir} @ {arch['img_size']}px")

    # Pairs: image i is clean, image i+1 is the corrupting source.
    pairs = [(images[i], images[(i + 1) % len(images)]) for i in range(min(args.n_pairs, len(images)))]

    # ---- targets first, using the EMA target encoder ----------------------
    print("[phase 1/3] computing target-encoder representations (EMA weights)")
    runner.load_role("target")
    runner.enc_hooks.set_off()
    runner.pred_hooks.set_off()
    masks = {}
    targets = {}
    with torch.no_grad():
        for idx, (name, img) in enumerate(images):
            ctx, tgt = sample_masks(arch["num_patches"], arch["grid_size"], args.seed * 1000 + idx)
            masks[name] = (ctx, tgt)
            reps = runner.encoder(img.to(device, dtype=dtype))  # full image, no mask
            targets[name] = reps[:, tgt, :].detach().clone()
            del reps
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[phase 2/3] swapping in context-encoder weights")
    runner.load_role("context")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- attribution patching sweep ---------------------------------------
    n_block_sites = len(runner.enc_hooks.block_sites()) + len(runner.pred_hooks.block_sites())
    n_head_sites = len(runner.enc_hooks.head_sites()) + len(runner.pred_hooks.head_sites())
    print(
        f"[phase 3/3] attribution patching over {n_block_sites + n_head_sites} sites "
        f"({n_block_sites} block-level + {n_head_sites} heads) x {len(pairs)} pairs"
    )

    # Two estimators from the same cached displacements: the plain first-order
    # expansion, and the same expansion with the gradient averaged along the
    # corrupt -> clean path.
    estimators: Dict[str, int] = {"ap": 1}
    if args.ig_steps > 1:
        estimators[f"ap_ig{args.ig_steps}"] = args.ig_steps
    primary_est = list(estimators)[-1]
    print(f"[phase 3/3] estimators: {', '.join(f'{k} ({v} backward pass(es))' for k, v in estimators.items())}")

    pair_results: List[PairResult] = []
    for i, ((cname, cimg), (xname, ximg)) in enumerate(pairs):
        ctx, tgt = masks[cname]
        treps = targets[cname]
        clean_acts, deltas, L_clean, L_corrupt = runner.prepare_pair(
            cimg, ximg, ctx, tgt, treps
        )

        ap_scores: Dict[str, Dict[str, float]] = {}
        ap_secs: Dict[str, float] = {}
        for est, k_steps in estimators.items():
            t0 = time.time()
            ap_scores[est] = runner.attribution_scores(
                deltas, cimg, ximg, ctx, tgt, treps, ig_steps=k_steps
            )
            ap_secs[est] = time.time() - t0
        del deltas
        gc.collect()

        true_scores: Dict[str, float] = {}
        true_secs = 0.0
        n_fwd = 0
        if args.validate and i < args.validate_pairs:
            ranking = ap_scores[primary_est]
            sites = runner.enc_hooks.block_sites() + runner.pred_hooks.block_sites()
            head_names = np.array(runner.enc_hooks.head_sites())
            head_vals = np.array([abs(ranking[h]) for h in head_names])
            k = max(1, args.validate_heads // 2)
            top = head_names[np.argsort(-head_vals)[:k]].tolist()
            rng = np.random.RandomState(args.seed + i)
            rest = [h for h in head_names.tolist() if h not in set(top)]
            rnd = rng.choice(rest, size=min(k, len(rest)), replace=False).tolist()
            sites = sites + top + rnd

            t1 = time.time()
            for s in sites:
                L_patched = runner.patch_site(s, clean_acts, ximg, ctx, tgt, treps)
                true_scores[s] = L_patched - L_corrupt
                n_fwd += 1
            true_secs = time.time() - t1

        del clean_acts
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pair_results.append(
            PairResult(
                clean_name=cname,
                corrupt_name=xname,
                n_context=len(ctx),
                n_target=len(tgt),
                L_clean=L_clean,
                L_corrupt=L_corrupt,
                ap_scores=ap_scores,
                true_scores=true_scores,
                ap_seconds=ap_secs,
                true_seconds=true_secs,
                true_forwards=n_fwd,
            )
        )
        msg = (
            f"  pair {i+1}/{len(pairs)}  {cname[:28]:<28} <- {xname[:24]:<24} "
            f"L_clean={L_clean:+.5f} L_corrupt={L_corrupt:+.5f} "
            f"gap={L_clean - L_corrupt:+.5f}  "
            + " ".join(f"{k} {v:.1f}s" for k, v in ap_secs.items())
        )
        if n_fwd:
            msg += f"  +true patching {n_fwd} sites {true_secs:.1f}s"
        print(msg)

    results = summarise(pair_results, arch, loaded, args, device, dtype, runner)
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[save] {args.save}")

    try:
        make_plots(results, pair_results, runner, args.plots)
        print(f"[save] plots -> {args.plots}")
    except Exception as exc:  # plotting must never sink a completed run
        print(f"[warn] plotting failed: {exc}")

    print_summary(results)


def summarise(
    pair_results: List[PairResult],
    arch: Dict[str, int],
    loaded: LoadedIJEPA,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    runner: "AttributionPatchingViTH",
) -> Dict[str, Any]:
    estimators = list(pair_results[0].ap_scores)
    all_sites = sorted(pair_results[0].ap_scores[estimators[0]])
    n_sites = len(all_sites)
    vpairs = [p for p in pair_results if p.true_scores]

    def family(site: str) -> str:
        parts = site.split(".")
        return f"{parts[0]}.{parts[1]}"

    scores_block: Dict[str, Any] = {}
    validation: Dict[str, Any] = {}
    for est in estimators:
        mean_ap = {s: float(np.mean([p.ap_scores[est][s] for p in pair_results])) for s in all_sites}
        std_ap = {s: float(np.std([p.ap_scores[est][s] for p in pair_results])) for s in all_sites}
        # normalise each pair by its own clean-corrupt gap before averaging:
        # 1.0 = "restoring this site alone recovers the whole clean-run advantage"
        frac_ap = {
            s: float(
                np.mean(
                    [
                        p.ap_scores[est][s] / (p.L_clean - p.L_corrupt)
                        for p in pair_results
                        if p.L_clean != p.L_corrupt
                    ]
                )
            )
            for s in all_sites
        }
        scores_block[est] = {
            "backward_passes": 1 if est == "ap" else int(est.replace("ap_ig", "")),
            "seconds_per_pair": float(np.mean([p.ap_seconds[est] for p in pair_results])),
            "mean": mean_ap,
            "std": std_ap,
            "fraction_of_gap": frac_ap,
        }

        if not vpairs:
            continue
        per_pair, pooled_ap, pooled_true = [], [], []
        for p in vpairs:
            sites = sorted(p.true_scores)
            a = np.array([p.ap_scores[est][s] for s in sites])
            b = np.array([p.true_scores[s] for s in sites])
            row = {
                "clean": p.clean_name,
                "corrupt": p.corrupt_name,
                "n_sites": len(sites),
                "pearson": pearson(a, b),
                "spearman": spearman(a, b),
                "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                "top5_overlap": topk_overlap(a, b, 5),
                "top10_overlap": topk_overlap(a, b, 10),
                "seconds_true_patching": p.true_seconds,
                "seconds_attribution_patching": p.ap_seconds[est],
                "forwards_true_patching": p.true_forwards,
            }
            per_pair.append(row)
            pooled_ap += a.tolist()
            pooled_true += b.tolist()

        pa, pb = np.array(pooled_ap), np.array(pooled_true)
        pooled_sites = [s for p in vpairs for s in sorted(p.true_scores)]
        by_family = {}
        for fam in sorted({family(s) for s in pooled_sites}):
            idx = [i for i, s in enumerate(pooled_sites) if family(s) == fam]
            if len(idx) < 3:
                continue
            by_family[fam] = {
                "n": len(idx),
                "pearson": pearson(pa[idx], pb[idx]),
                "spearman": spearman(pa[idx], pb[idx]),
                "sign_agreement": float(np.mean(np.sign(pa[idx]) == np.sign(pb[idx]))),
            }
        # residual-stream sites are where a first-order step is most obviously
        # wrong (patching them restores everything upstream at once), so report
        # the sweep-relevant subset - individual heads and block outputs - too
        local = [i for i, s in enumerate(pooled_sites) if ".resid_pre." not in s]
        heads = [i for i, s in enumerate(pooled_sites) if ".head." in s]
        validation[est] = {
            "per_pair": per_pair,
            "pooled": {
                "n": len(pa),
                "pearson": pearson(pa, pb),
                "spearman": spearman(pa, pb),
                "sign_agreement": float(np.mean(np.sign(pa) == np.sign(pb))),
                "top10_overlap": topk_overlap(pa, pb, 10),
                "top20_overlap": topk_overlap(pa, pb, 20),
            },
            "pooled_excluding_resid": {
                "n": len(local),
                "pearson": pearson(pa[local], pb[local]),
                "spearman": spearman(pa[local], pb[local]),
                "sign_agreement": float(np.mean(np.sign(pa[local]) == np.sign(pb[local]))),
                "top10_overlap": topk_overlap(pa[local], pb[local], 10),
            },
            "pooled_heads_only": {
                "n": len(heads),
                "pearson": pearson(pa[heads], pb[heads]),
                "spearman": spearman(pa[heads], pb[heads]),
                "sign_agreement": float(np.mean(np.sign(pa[heads]) == np.sign(pb[heads]))),
                "top10_overlap": topk_overlap(pa[heads], pb[heads], 10),
            },
            "by_family": by_family,
        }

    cost = {
        est: {
            "backward_passes": scores_block[est]["backward_passes"],
            "seconds_all_sites": scores_block[est]["seconds_per_pair"],
        }
        for est in estimators
    }
    if vpairs:
        per_site_fwd = float(np.mean([p.true_seconds / max(1, p.true_forwards) for p in vpairs]))
        cost["measured_seconds_per_patched_site"] = per_site_fwd
        cost["extrapolated_exhaustive_patching_seconds"] = per_site_fwd * n_sites
        for est in estimators:
            secs = scores_block[est]["seconds_per_pair"]
            cost[est]["speedup_vs_exhaustive"] = (per_site_fwd * n_sites) / secs if secs else None

    return {
        "experiment": "task5_attribution_patching_vith14",
        "checkpoint": loaded.checkpoint_meta,
        "key_coverage": loaded.coverage,
        "architecture": arch,
        "config": {
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "n_pairs": len(pair_results),
            "seed": args.seed,
            "data_dir": args.data_dir,
            "validate": args.validate,
            "validate_pairs": args.validate_pairs,
            "validate_heads": args.validate_heads,
            "ig_steps": args.ig_steps,
            "estimators": estimators,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "sites": {
            "total": n_sites,
            "encoder_blocks": len(runner.enc_hooks.block_sites()),
            "encoder_heads": len(runner.enc_hooks.head_sites()),
            "predictor_blocks": len(runner.pred_hooks.block_sites()),
            "predictor_heads": len(runner.pred_hooks.head_sites()),
        },
        "metric": {
            "definition": "L = -mean_j ||predictor(context)_j - target_encoder(x_clean)_j||^2 / d",
            "L_clean_mean": float(np.mean([p.L_clean for p in pair_results])),
            "L_corrupt_mean": float(np.mean([p.L_corrupt for p in pair_results])),
            "gap_mean": float(np.mean([p.L_clean - p.L_corrupt for p in pair_results])),
        },
        "cost": cost,
        "attribution_patching": scores_block,
        "per_pair": [
            {
                "clean": p.clean_name,
                "corrupt": p.corrupt_name,
                "n_context": p.n_context,
                "n_target": p.n_target,
                "L_clean": p.L_clean,
                "L_corrupt": p.L_corrupt,
                "ap_seconds": p.ap_seconds,
                "true_scores": p.true_scores,
                # estimates restricted to the validated sites, so the fidelity
                # scatter can be redrawn without re-running the sweep
                "ap_scores_validated": {
                    est: {s: p.ap_scores[est][s] for s in p.true_scores}
                    for est in p.ap_scores
                } if p.true_scores else {},
            }
            for p in pair_results
        ],
        "validation": validation,
    }


def print_summary(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    arch = res["architecture"]
    ck = res["checkpoint"]
    print(
        f"checkpoint : {os.path.basename(ck['path'])} "
        f"({ck['file_size_bytes']/1e9:.2f} GB, epoch {ck['epoch']}, "
        f"{ck['encoder_params']/1e6:.0f} M encoder params)"
    )
    print(
        f"arch       : ViT-H/{arch['patch_size']} d={arch['embed_dim']} "
        f"L={arch['depth']} H={arch['num_heads']}, predictor d={arch['predictor_embed_dim']} "
        f"L={arch['predictor_depth']}"
    )
    print("key cover  : " + ", ".join(
        f"{c['component']} {c['checkpoint_tensors']}/{c['model_tensors']} tensors, "
        f"{len(c['missing'])} missing" for c in res["key_coverage"]
    ))
    m = res["metric"]
    print(
        f"metric     : L_clean={m['L_clean_mean']:+.5f} L_corrupt={m['L_corrupt_mean']:+.5f} "
        f"gap={m['gap_mean']:+.5f}"
    )

    cost = res["cost"]
    n_sites = res["sites"]["total"]
    print(f"\ncost for all {n_sites} sites (per image pair):")
    for est in res["attribution_patching"]:
        c = cost[est]
        line = (f"  {est:<8} {c['backward_passes']:>2} backward pass(es)  "
                f"{c['seconds_all_sites']:.1f}s")
        if c.get("speedup_vs_exhaustive"):
            line += f"   {c['speedup_vs_exhaustive']:.0f}x faster than exhaustive patching"
        print(line)
    if "extrapolated_exhaustive_patching_seconds" in cost:
        print(f"  exhaustive activation patching (measured "
              f"{cost['measured_seconds_per_patched_site']*1000:.0f} ms/site): "
              f"{cost['extrapolated_exhaustive_patching_seconds']:.0f}s")

    val = res.get("validation") or {}
    if val:
        print("\nfidelity vs true activation patching:")
        print(f"  {'estimator':<9} {'subset':<22} {'n':>5} {'pearson':>8} {'spearman':>9} "
              f"{'sign':>6} {'top10':>6}")
        for est, v in val.items():
            for label, key in (
                ("all sites", "pooled"),
                ("heads + block outs", "pooled_excluding_resid"),
                ("attention heads", "pooled_heads_only"),
            ):
                d = v[key]
                print(f"  {est:<9} {label:<22} {d['n']:>5} {d['pearson']:>8.3f} "
                      f"{d['spearman']:>9.3f} {d['sign_agreement']:>6.2f} "
                      f"{d.get('top10_overlap', float('nan')):>6.2f}")
        primary = list(val)[-1]
        print(f"\n  by site family ({primary}):")
        for fam, d in sorted(val[primary]["by_family"].items()):
            print(f"    {fam:<18} n={d['n']:<4} r={d['pearson']:+.3f} "
                  f"rho={d['spearman']:+.3f} sign={d['sign_agreement']:.2f}")

    primary = list(res["attribution_patching"])[-1]
    fr = res["attribution_patching"][primary]["fraction_of_gap"]
    heads = {s: v for s, v in fr.items() if s.startswith("enc.head.")}
    order = sorted(heads, key=lambda s: -abs(heads[s]))
    print(f"\ntop 15 ViT-H encoder heads by |{primary}| "
          "(fraction of the clean-corrupt gap):")
    for s in order[:15]:
        print(f"  {s:<24} {heads[s]:+.4f}")

    print("\nencoder layer profile (fraction of gap, "
          f"{primary}):")
    print(f"  {'layer':>5} {'attn_out':>10} {'mlp_out':>10} {'resid_pre':>11} {'heads_abs':>10}")
    for l in range(arch["depth"]):
        h_abs = sum(abs(fr.get(f"enc.head.{l}.{h}", 0.0)) for h in range(arch["num_heads"]))
        print(f"  {l:>5} {fr.get(f'enc.attn_out.{l}', float('nan')):>10.4f} "
              f"{fr.get(f'enc.mlp_out.{l}', float('nan')):>10.4f} "
              f"{fr.get(f'enc.resid_pre.{l}', float('nan')):>11.4f} {h_abs:>10.4f}")


def make_plots(
    res: Dict[str, Any],
    pair_results: List[PairResult],
    runner: "AttributionPatchingViTH",
    outdir: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    estimators = list(res["attribution_patching"])
    primary = estimators[-1]
    # The validation in this experiment shows the two estimators are faithful on
    # different site families, so each plot uses the one that applies: plain "ap"
    # for individual heads (small perturbations), the path-averaged estimator for
    # block outputs and the residual stream (large ones).
    head_est = "ap" if "ap" in estimators else primary
    fr = res["attribution_patching"][primary]["fraction_of_gap"]
    frh = res["attribution_patching"][head_est]["fraction_of_gap"]
    arch = res["architecture"]
    L, H = arch["depth"], arch["num_heads"]

    # -- 1. encoder head heatmap ------------------------------------------
    grid = np.array([[frh.get(f"enc.head.{l}.{h}", 0.0) for h in range(H)] for l in range(L)])
    vmax = float(np.abs(grid).max()) or 1.0
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("head")
    ax.set_ylabel("encoder layer")
    ax.set_title("I-JEPA ViT-H/14: per-head causal effect on prediction\n"
                 f"attribution patching ({head_est}), fraction of clean-corrupt gap")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ap_vith14_head_heatmap.png"), dpi=140)
    plt.close(fig)

    # -- 2. per-layer families ---------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for fam, style in (("attn_out", "-o"), ("mlp_out", "-s"), ("resid_pre", "--")):
        axes[0].plot(range(L), [fr.get(f"enc.{fam}.{l}", np.nan) for l in range(L)],
                     style, ms=3, label=f"enc.{fam}")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("fraction of gap")
    axes[0].set_title(f"Attribution patching by encoder layer, ViT-H/14 ({primary})")
    axes[0].legend()
    axes[1].bar(range(L),
                [sum(abs(frh.get(f"enc.head.{l}.{h}", 0.0)) for h in range(H)) for l in range(L)])
    axes[1].set_xlabel("encoder layer")
    axes[1].set_ylabel("sum |head effect|")
    axes[1].set_title(f"Total per-head attribution mass per layer ({head_est})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ap_vith14_layer_profile.png"), dpi=140)
    plt.close(fig)

    # -- 3. estimator vs ground truth --------------------------------------
    vpairs = [p for p in pair_results if p.true_scores]
    if vpairs and res.get("validation"):
        fig, axes = plt.subplots(1, len(estimators), figsize=(5.6 * len(estimators), 5.4),
                                 squeeze=False)
        for ax, est in zip(axes[0], estimators):
            for p in vpairs:
                sites = sorted(p.true_scores)
                a = np.array([p.ap_scores[est][s] for s in sites])
                b = np.array([p.true_scores[s] for s in sites])
                is_head = np.array([".head." in s for s in sites])
                is_resid = np.array([".resid_pre." in s for s in sites])
                other = ~is_head & ~is_resid
                ax.scatter(a[other], b[other], s=16, alpha=0.7, c="tab:blue")
                ax.scatter(a[is_head], b[is_head], s=16, alpha=0.7, c="tab:orange")
                ax.scatter(a[is_resid], b[is_resid], s=16, alpha=0.7, c="tab:red", marker="x")
            lim = max(abs(np.array(ax.get_xlim())).max(), abs(np.array(ax.get_ylim())).max())
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
            ax.scatter([], [], c="tab:blue", label="attn/mlp outputs")
            ax.scatter([], [], c="tab:orange", label="attention heads")
            ax.scatter([], [], c="tab:red", marker="x", label="residual stream")
            v = res["validation"][est]
            ax.set_xlabel(f"{est} (estimated dL)")
            ax.set_ylabel("true activation patching (measured dL)")
            ax.set_title(
                f"{est}: r={v['pooled']['pearson']:.3f}, "
                f"rho={v['pooled']['spearman']:.3f}\n"
                f"excluding residual stream: r={v['pooled_excluding_resid']['pearson']:.3f}, "
                f"heads only: r={v['pooled_heads_only']['pearson']:.3f}"
            )
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ap_vith14_validation_scatter.png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
