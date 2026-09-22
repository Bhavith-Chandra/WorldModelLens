"""Latent Lens: layer-wise identity emergence in the I-JEPA predictor (Task 6).

Question
--------
Reviewer Question 4 asks how to tell an *architectural* property apart from a
*representational* one. Positional swaps and attention blockades confirm that
the predictor depends on the structures it is built out of, but they cannot say
when, along the predictor's depth, a target token stops being "a position" and
starts being "the content at that position".

The Latent Lens answers that observationally. I-JEPA's predictor ends with

    prediction_j = predictor_project_back(norm(x_j))

mapping the predictor's residual width back into the encoder's representation
space. That head is applied here to the residual stream at *every* predictor
layer, not just the last, giving a trajectory of intermediate "predictions" in
the target-representation space - the JEPA analogue of the logit lens.

The control that makes this answer Q4 is built into the architecture. At layer 0
a target token is exactly ``mask_token + pos_embed[p]``: identical for every
image, carrying position and nothing else. So layer 0 *is* the purely
architectural prior, measured rather than assumed, and everything above it is
representation dynamics.

Design
------
For each image i and each target patch p we project the residual state at every
predictor layer through the output head and score it against the ground-truth
target-encoder representation ``t[i, p]``:

The measurement has two halves, reported separately because a predictor can pass
one and fail the other:

* **6a - absolute.** Does the projection get *near* the true target embedding at
  all? Cosine against t[i, p], no gallery. Raw cosine cannot be read alone: at
  layer 0 the target token carries zero image information yet already scores
  ~0.09 on the ViT-H, and the position-only prototype - which never sees the
  image - reaches ~0.35, because target space has a large shared mean
  component. The reportable 6a quantity is therefore the *margin over that
  prototype* and the depth at which it first becomes positive.
* **6b - relative.** Is t[i, p] ranked above a gallery of wrong candidates? The
  two galleries below.

On the official ViT-H the two disagree in an informative way: Gallery B is at 29x
chance by layer 3, while absolute alignment does not clear the positional
prototype until layer 5. The predictor learns to discriminate before it learns to
arrive, which only an absolute test can see.

* ``cos_target``   cosine similarity to t[i, p].
* ``cos_centered`` cosine after subtracting the positional prototype
  ``t_bar[p] = mean_i t[i, p]`` from both sides. This removes everything
  predictable from position alone, so it isolates the *content-specific*
  component - the part a positional prior cannot produce.
* **Gallery A - "which patch of this image?"** rank t[i, p] against all patches
  of the same image, {t[i, q] : q}. Within one image, position and content are
  bound, so this measures localisation. Chance = 1/n_patches.
* **Gallery B - "which image?"** rank t[i, p] against the same patch index in
  every other image, {t[i', p] : i'}. Position is held fixed, so this is pure
  content identity and cannot be solved by any positional prior.
  Chance = 1/n_images.

Gallery B at layer 0 must come out at chance by construction; the script asserts
it, which doubles as a correctness check on the whole trace.

Uncertainty
-----------
95% CIs come from a paired cluster bootstrap (``--bootstrap``). Images are the
primary resampling unit because one image contributes n_masks * n_target_patches
rows from a single encoder pass; resampling tokens would treat those correlated
rows as independent and give intervals that are far too tight. The same
resampled indices are reused at every layer, so layer-to-layer increments and
margins over the prototype are paired differences rather than gaps between
independent intervals. Where the run has enough masks
(``--min_masks_two_way``), masks are resampled as a second cluster axis as well,
which is the interval a claim needs if it is meant to generalise past the
particular masks that were drawn.

The trajectory asks the same question of every layer, so its tests arrive as
*families* - 12 consecutive-layer increments on a 12-layer predictor, 13
per-layer margins over the prototype. Uncorrected, a family of 12 at alpha=0.05
has a 46% chance of at least one false positive, so "every step is significant"
is not a claim the pointwise intervals can carry. Each family is reported with
Holm-Bonferroni adjusted p-values (familywise control under arbitrary
dependence), Benjamini-Hochberg adjusted p-values (false discovery), and a
simultaneous sup-t band, alongside the uncorrected interval.

Where a curve crosses zero is reported as a depth with its own interval, by
interpolating inside each replicate, rather than as "the first layer whose
interval clears zero" - which localises the crossing only to whichever block
happened to be sampled.

The trace is expensive and the statistics are not, so a completed run's
sufficient statistics are cached (``--cache_dir``); ``--reanalyze`` rebuilds
every number and figure from them without touching a model.

The pre-registered criteria below are evaluated on point estimates and left
exactly as registered. The bootstrap is reported alongside as
``verdict_robust_to_ci``: whether the registered call survives its own
uncertainty. It is a separate field and never overwrites ``verdict``.

Pre-registered verdict (the task's watch item)
----------------------------------------------
The lens is recorded as INFORMATIVE only if all three hold:

1. final-layer Gallery-B top-1 >= 2x chance (content identity is really recovered);
2. the largest single-layer increment in Gallery-B top-1 is < 80% of the total
   rise (the identity emerges *across* layers rather than appearing in one jump
   at the output head, which would mean the lens sees nothing intermediate);
3. layer-0 Gallery-B top-1 is within 2x chance (the architectural floor is where
   the architecture says it should be).

If the verdict is AMBIGUOUS the task calls for escalating to a QK-routing vs
OV-content decomposition and path patching across the predictor, which is a
causal experiment rather than this observational one.

Usage
-----
    python experiments/latent_lens_trajectory.py --model mini
    python experiments/latent_lens_trajectory.py --model vith
    python experiments/latent_lens_trajectory.py --reanalyze   # cached, no GPU
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from attribution_patching_vith14 import (  # noqa: E402
    load_images,
    load_official_vith,
    open_slim_state,
    copy_state_into,
    sample_masks,
)
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter  # noqa: E402
from world_model_lens.core.config import WorldModelConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Model wrappers: mini checkpoint and official ViT-H behind one interface
# ---------------------------------------------------------------------------


@dataclass
class LensModel:
    """Everything the lens needs, with the two encoder roles kept separate."""

    predictor: nn.Module
    arch: Dict[str, int]
    name: str
    meta: Dict[str, Any]
    _encode_context: Any
    _target_encode: Any
    _use_target_role: Any = None
    _use_context_role: Any = None

    def encode_context(self, img: torch.Tensor, context_ids: List[int]) -> torch.Tensor:
        return self._encode_context(img, context_ids)

    def target_encode(self, img: torch.Tensor) -> torch.Tensor:
        return self._target_encode(img)

    def use_target_role(self) -> None:
        if self._use_target_role is not None:
            self._use_target_role()

    def use_context_role(self) -> None:
        if self._use_context_role is not None:
            self._use_context_role()


def build_mini(checkpoint: str, device: torch.device) -> LensModel:
    """Load a mini I-JEPA checkpoint (6-layer encoder, 4-layer predictor)."""
    config = WorldModelConfig(
        backend="ijepa",
        d_embed=192,
        n_layers=6,
        n_heads=3,
        predictor_embed_dim=384,
        predictor_depth=4,
        predictor_heads=6,
        img_size=224,
        patch_size=16,
    )
    adapter = IJEPAAdapter(config)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = adapter.load_state_dict(sd, strict=False)
    hard = [k for k in missing if not k.startswith("hook")]
    if hard or unexpected:
        raise RuntimeError(
            f"mini checkpoint did not load cleanly: missing={hard[:6]} "
            f"unexpected={list(unexpected)[:6]}"
        )
    # BaseModelAdapter.to() takes `device` keyword-only, unlike nn.Module.to()
    adapter.eval()
    adapter.to(device=device)
    for p in adapter.parameters():
        p.requires_grad_(False)

    grid = adapter.context_encoder.patch_embed.grid_size
    arch = dict(
        embed_dim=192,
        depth=6,
        num_heads=3,
        patch_size=16,
        img_size=224,
        num_patches=grid * grid,
        grid_size=grid,
        predictor_embed_dim=384,
        predictor_depth=4,
        predictor_heads=6,
    )
    return LensModel(
        predictor=adapter.predictor,
        arch=arch,
        name=f"mini ({os.path.basename(checkpoint)})",
        meta={
            "checkpoint": os.path.abspath(checkpoint),
            "file_size_bytes": os.path.getsize(checkpoint),
            "encoder_params": int(sum(p.numel() for p in adapter.context_encoder.parameters())),
            "predictor_params": int(sum(p.numel() for p in adapter.predictor.parameters())),
        },
        _encode_context=lambda img, ids: adapter.context_encoder(img.to(device), patch_ids=ids),
        _target_encode=lambda img: adapter.target_encoder(img.to(device)),
    )


def build_vith(checkpoint: str, device: torch.device, slim_dir: Optional[str]) -> LensModel:
    """Load the official 10.36 GB ViT-H/14 (32-layer encoder, 12-layer predictor).

    Reuses the Task 5 loader, which remaps the official key names, restores the
    qkv biases this repo's ``Block`` omits, and asserts 100% key coverage. One
    encoder instance serves both roles, so the two are swapped rather than held
    at once.
    """
    loaded = load_official_vith(checkpoint, device=device, dtype=torch.float32,
                                slim_dir=slim_dir)
    enc = loaded.encoder

    def use_role(role: str) -> None:
        state = open_slim_state(loaded.slim[role])
        copy_state_into(enc, state)
        del state
        gc.collect()

    return LensModel(
        predictor=loaded.predictor,
        arch=loaded.arch,
        name="official ViT-H/14",
        meta=loaded.checkpoint_meta | {"key_coverage": loaded.coverage},
        _encode_context=lambda img, ids: enc(img.to(device), patch_ids=ids),
        _target_encode=lambda img: enc(img.to(device)),
        _use_target_role=lambda: use_role("target_encoder"),
        _use_context_role=lambda: use_role("encoder"),
    )


# ---------------------------------------------------------------------------
# The lens
# ---------------------------------------------------------------------------


class PredictorTrace:
    """Capture the residual stream entering every predictor block."""

    def __init__(self, predictor: nn.Module):
        self.predictor = predictor
        self.states: Dict[int, torch.Tensor] = {}
        self.handles: List[Any] = []

    def __enter__(self) -> "PredictorTrace":
        for i, blk in enumerate(self.predictor.blocks):
            self.handles.append(
                blk.register_forward_pre_hook(
                    lambda m, inp, i=i: self.states.__setitem__(i, inp[0].detach())
                )
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def latent_lens(predictor: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Project a predictor residual state into the target representation space.

    This is the predictor's own output head - ``norm`` then
    ``predictor_project_back`` - applied at an arbitrary depth. Using the
    model's real head rather than a fitted probe is what keeps the trajectory
    interpretable: at the last layer it reproduces the model's actual output
    exactly, so intermediate layers are read off the same ruler.
    """
    return predictor.predictor_project_back(predictor.norm(x))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model == "mini":
        model = build_mini(args.mini_checkpoint, device)
    else:
        model = build_vith(args.checkpoint, device, args.slim_dir)
    arch = model.arch
    n_layers = arch["predictor_depth"]
    print(f"[model] {model.name}: encoder d={arch['embed_dim']} L={arch['depth']}, "
          f"predictor d={arch['predictor_embed_dim']} L={n_layers}, "
          f"{arch['num_patches']} patches")

    images = load_images(args.data_dir, args.n_images, arch["img_size"], args.seed)
    n_img = len(images)
    print(f"[data] {n_img} images from {args.data_dir} @ {arch['img_size']}px")

    # Masks are shared across images. Gallery B holds the patch index fixed and
    # varies the image, so every image must be masked identically for the
    # comparison to be like-for-like - and it is what makes the layer-0 target
    # tokens literally identical across images, which the check below asserts.
    masks = [sample_masks(arch["num_patches"], arch["grid_size"], args.seed * 1000 + m)
             for m in range(args.n_masks)]
    print(f"[masks] {len(masks)} shared masks, "
          f"{[len(t) for _, t in masks]} target patches each")

    # ---- ground-truth target representations, from the EMA target encoder ---
    model.use_target_role()
    targets = torch.zeros(n_img, arch["num_patches"], arch["embed_dim"])
    with torch.no_grad():
        for i, (_name, img) in enumerate(images):
            targets[i] = model.target_encode(img)[0].detach().float().cpu()
    print(f"[targets] {tuple(targets.shape)} from the target encoder")

    # positional prototype: what position alone predicts, averaged over images
    proto = targets.mean(dim=0)  # [n_patches, d]

    model.use_context_role()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- trace every (mask, image) through the predictor -------------------
    lens_points: List[Any] = list(range(n_layers)) + ["out"]
    keys_all = [str(lp) for lp in lens_points] + ["proto"]
    acc = LensStats(keys_all, n_img, len(masks))
    rows: List[Dict[str, Any]] = []
    l0_spreads = []

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for mi, (ctx, tgt) in enumerate(masks):
            n_ctx = len(ctx)
            patch_idx = torch.as_tensor(tgt, dtype=torch.long)
            image_idx = torch.zeros(len(tgt), dtype=torch.long)
            layer0 = []
            # Gallery B and the positional prototype depend on the mask alone, so
            # they are built once per mask rather than once per (mask, image).
            # Rebuilding them inside the image loop is not just wasted work: on
            # the ViT-H each one is a ~50 MB fancy-index copy, and churning
            # 100 MB per image for 240 images x 32 masks fragments host memory
            # badly enough to abort the run outright.
            t_proto = proto[tgt]                             # [n_tgt, d]
            gb = torch.nn.functional.normalize(targets[:, tgt, :], dim=-1)
            for i, (_name, img) in enumerate(images):
                with PredictorTrace(model.predictor) as trace:
                    latents = model.encode_context(img, ctx)
                    out = model.predictor(latents, ctx, tgt)

                per_layer: Dict[Any, torch.Tensor] = {}
                for l in range(n_layers):
                    state = trace.states[l][:, n_ctx:, :]   # target tokens only
                    per_layer[l] = latent_lens(model.predictor, state)[0].float().cpu()
                per_layer["out"] = out[0].float().cpu()
                layer0.append(trace.states[0][:, n_ctx:, :][0].float().cpu())

                t_true = targets[i, tgt]              # [n_tgt, d]
                nt_all = torch.nn.functional.normalize(targets[i], dim=-1)
                image_idx.fill_(i)

                # the positional prototype is a baseline "predictor" that uses
                # position and nothing else - the ceiling for a purely
                # architectural account of the trajectory
                for lp in lens_points + ["proto"]:
                    z = t_proto if lp == "proto" else per_layer[lp]
                    cos_t = cosine(z, t_true)
                    cos_c = cosine(z - t_proto, t_true - t_proto)
                    mse = ((z - t_true) ** 2).mean(dim=-1)
                    zn = torch.nn.functional.normalize(z, dim=-1)
                    sims_a = zn @ nt_all.T                          # [n_tgt, n_patches]
                    sims_b = torch.einsum("td,itd->ti", zn, gb)     # [n_tgt, n_img]
                    rank_a = ranks_against(sims_a, patch_idx)
                    rank_b = ranks_against(sims_b, image_idx)

                    acc.add(lp, i, mi, cos_t.double().numpy(), cos_c.double().numpy(),
                            mse.double().numpy(), rank_a, rank_b)
                    if args.save_rows:
                        rows.extend(
                            {
                                "mask": mi,
                                "image": i,
                                "layer": str(lp),
                                "patch": int(patch),
                                "cos_target": float(cos_t[k]),
                                "cos_centered": float(cos_c[k]),
                                "mse": float(mse[k]),
                                "rank_a": int(rank_a[k]),
                                "rank_b": int(rank_b[k]),
                            }
                            for k, patch in enumerate(tgt)
                        )
                done += 1
                if done % 32 == 0:
                    print(f"  traced {done}/{len(masks)*n_img} (mask, image) pairs "
                          f"({time.time()-t0:.0f}s)")

            # Architectural check: at layer 0 a target token is exactly
            # mask_token + pos_embed[p], so with a shared mask it must be
            # byte-for-byte identical across images. If this drifts, the trace
            # is not reading the residual stream it claims to.
            st = torch.stack(layer0)
            l0_spreads.append(float((st - st[0]).abs().max()))

    l0_spread = max(l0_spreads)
    ok = l0_spread < 1e-4
    print(f"[check] layer-0 target-token spread across images: {l0_spread:.2e} "
          f"({'identical, as the architecture requires' if ok else 'NOT identical - trace is wrong'})")
    if not ok:
        raise RuntimeError(
            f"layer-0 target tokens differ across images by {l0_spread:.2e}; they are "
            "mask_token + pos_embed by construction and must be identical"
        )

    if args.cache_dir:
        acc.save(cache_path(args, model, n_img, len(masks)),
                 trace_meta(arch, model, args, n_img, len(masks), l0_spread, lens_points))

    return summarise(acc, lens_points, arch, model, args, n_img, len(masks),
                     l0_spread, rows)

# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------

BOOT_METRICS = ("cos_target", "cos_centered", "gallery_a_top1", "gallery_b_top1")

# pre-registered C2: the largest single-layer increment must be under this
# fraction of the total rise for identity to count as emerging *across* layers
C2_THRESHOLD = 0.8

# per-lens-point running sums, enough to rebuild every reported scalar
STAT_FIELDS = ("cos_target", "cos_target_sq", "cos_centered", "cos_centered_sq",
               "mse", "a_top1", "a_top5", "a_mrr", "b_top1", "b_mrr", "n")


class LensStats:
    """Running sums over (lens point, image, mask), instead of per-token rows.

    The trajectory needs three things from the trace: per-layer means, the
    per-(image, mask) sums the cluster bootstrap contracts against, and nothing
    else. Materialising one dict per (image, mask, lens point, patch) costs
    hundreds of megabytes as soon as the mask count goes up - and mask count is
    exactly the axis this study needs to push on - so the trace folds each
    (image, mask) straight into these accumulators. Everything downstream reads
    them, which also makes a completed run cacheable in a few megabytes.
    """

    def __init__(self, keys: List[str], n_img: int, n_masks: int):
        self.keys = list(keys)                       # lens points + "proto"
        self.n_img, self.n_masks = n_img, n_masks
        self.sums = np.zeros((len(BOOT_METRICS), len(keys), n_img, n_masks),
                             dtype=np.float64)
        self.counts = np.zeros((n_img, n_masks), dtype=np.float64)
        self.stats = np.zeros((len(keys), len(STAT_FIELDS)), dtype=np.float64)
        self._j = {k: j for j, k in enumerate(self.keys)}

    def add(self, key: Any, i: int, k: int, cos_t: np.ndarray, cos_c: np.ndarray,
            mse: np.ndarray, rank_a: np.ndarray, rank_b: np.ndarray) -> None:
        j = self._j[str(key)]
        a1 = (rank_a == 1).astype(np.float64)
        b1 = (rank_b == 1).astype(np.float64)
        self.sums[0, j, i, k] += cos_t.sum()
        self.sums[1, j, i, k] += cos_c.sum()
        self.sums[2, j, i, k] += a1.sum()
        self.sums[3, j, i, k] += b1.sum()
        self.stats[j] += (
            cos_t.sum(), (cos_t ** 2).sum(), cos_c.sum(), (cos_c ** 2).sum(),
            mse.sum(), a1.sum(), (rank_a <= 5).sum(), (1.0 / rank_a).sum(),
            b1.sum(), (1.0 / rank_b).sum(), len(cos_t),
        )
        if j == 0:
            self.counts[i, k] += len(cos_t)

    def mean(self, key: Any, field: str) -> float:
        j, f = self._j[str(key)], STAT_FIELDS.index(field)
        n = self.stats[j, STAT_FIELDS.index("n")]
        return float(self.stats[j, f] / n) if n else float("nan")

    def std(self, key: Any, field: str) -> float:
        m = self.mean(key, field)
        return float(np.sqrt(max(0.0, self.mean(key, field + "_sq") - m * m)))

    def n(self, key: Any) -> int:
        return int(self.stats[self._j[str(key)], STAT_FIELDS.index("n")])

    # -- cache ----------------------------------------------------------
    def save(self, path: str, meta: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, sums=self.sums, counts=self.counts,
                            stats=self.stats,
                            keys=np.array(self.keys, dtype=object),
                            meta=np.array(json.dumps(meta)))
        print(f"[cache] {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> Tuple["LensStats", Dict[str, Any]]:
        z = np.load(path, allow_pickle=True)
        keys = [str(k) for k in z["keys"]]
        counts = z["counts"]
        obj = cls(keys, int(counts.shape[0]), int(counts.shape[1]))
        obj.sums, obj.counts, obj.stats = z["sums"], counts, z["stats"]
        return obj, json.loads(str(z["meta"]))


def ranks_against(sims: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """1-based rank of each row's true entry: 1 + how many candidates score higher."""
    true_score = sims.gather(1, truth.view(-1, 1))
    return (1 + (sims > true_score).sum(dim=1)).cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Cluster bootstrap, and the corrections that go with a family of tests
# ---------------------------------------------------------------------------


def bootstrap_draws(
    sums: np.ndarray,
    counts: np.ndarray,
    n_boot: int,
    seed: int,
    unit: str = "image",
    chunk: int = 2048,
) -> np.ndarray:
    """Cluster bootstrap over ``sums``; returns draws of shape [metric, layer, B].

    Two choices matter here and both are deliberate.

    **The resampling unit is a cluster, never a token.** One image contributes
    ``n_masks * n_target_patches`` rows that all come from the same encoder pass
    over the same picture, so they are anything but independent. Bootstrapping
    tokens would treat ~11k correlated rows as 11k samples and produce intervals
    that are far too tight - a CI that says the trend is real when it has only
    measured how many patches a mask happens to contain.

    ``unit="image"`` resamples images and holds the mask set fixed. It answers
    "would another 240 images say the same?" and is the headline interval.
    ``unit="image_mask"`` also resamples masks - a two-way cluster bootstrap
    over a crossed design - and answers "would another draw of images *and*
    masks say the same?". The second is the honest interval whenever a claim is
    meant to generalise past the particular masks that were drawn, but it needs
    enough masks to estimate that component, so the caller asks for it only once
    the mask count is large enough to support it.

    **The same resampled indices are reused for every layer and metric.** That
    makes every layer-to-layer comparison paired: the interval on
    ``layer 5 - layer 4`` is an interval on that increment, computed from
    replicates in which both layers saw the same images, rather than the much
    wider and much less meaningful gap between two independently drawn
    intervals. It is what lets the trajectory claim "the rise between these
    layers is real" instead of only "these two layers differ somewhere".

    A resample of *n* clusters drawn with replacement is exactly a
    ``Multinomial(n, 1/n)`` vector of multiplicities, so each replicate is a
    contraction of the precomputed sums rather than a re-scan of anything.
    """
    n_img, n_masks = counts.shape
    rng = np.random.default_rng(seed)
    n_metric, n_key = sums.shape[:2]
    out = np.empty((n_metric, n_key, n_boot), dtype=np.float64)
    p_img = np.full(n_img, 1.0 / n_img)
    p_msk = np.full(n_masks, 1.0 / n_masks)
    if unit not in ("image", "image_mask"):
        raise ValueError(f"unknown resampling unit {unit!r}")
    s_img = sums.sum(axis=3)                                       # [M, L, I]
    c_img = counts.sum(axis=1)                                     # [I]

    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        a = rng.multinomial(n_img, p_img, size=stop - start).astype(np.float64)
        if unit == "image":
            num = np.tensordot(s_img, a, axes=([2], [1]))          # [M, L, b]
            den = a @ c_img                                        # [b]
        else:
            g = rng.multinomial(n_masks, p_msk, size=stop - start).astype(np.float64)
            t = np.tensordot(sums, a, axes=([2], [1]))             # [M, L, K, b]
            num = np.einsum("mlkb,bk->mlb", t, g)
            den = np.einsum("ik,bi,bk->b", counts, a, g)
        out[:, :, start:stop] = num / den
    return out


def ci_of(draws: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "se": float(np.std(draws, ddof=1))}


def boot_pvalue(d: np.ndarray, n_boot: int) -> float:
    """Two-sided bootstrap p-value for ``mean(d) == 0``.

    ``(1 + count) / (B + 1)`` rather than ``count / B`` so a p-value can never
    come out exactly zero: with B replicates the smallest reportable value is
    ``2 / (B + 1)``, and no test can resolve past its own resolution. That floor
    is what decides how many replicates a *corrected* analysis needs - Bonferroni
    over 12 tests compares against 0.05/12 = 0.0042, which B = 1000 (floor
    0.002) can only just distinguish and B = 10000 (floor 0.0002) clears
    comfortably.
    """
    le = int(np.count_nonzero(d <= 0))
    ge = int(np.count_nonzero(d >= 0))
    return float(min(1.0, 2.0 * min((1 + le) / (n_boot + 1), (1 + ge) / (n_boot + 1))))


def normal_pvalue(delta: float, se: float) -> float:
    """Companion p-value from the bootstrap SE, unaffected by the replicate floor.

    Reported next to ``p_boot`` only so that "far past the resolution of the
    bootstrap" can be checked rather than asserted. It buys that by assuming
    normality, which the percentile p-value does not.
    """
    if se <= 0.0:
        return 0.0 if delta != 0.0 else 1.0
    return float(math.erfc(abs(delta) / (se * math.sqrt(2.0))))


def holm_adjust(p: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Controls the familywise error rate under arbitrary dependence between the
    tests - which paired layer-to-layer increments certainly have - and is
    uniformly more powerful than plain Bonferroni.
    """
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh_adjust(p: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m * p[i] / (rank + 1))
        adj[i] = min(1.0, running)
    return adj


def paired_tests(
    d: np.ndarray,
    labels: List[Dict[str, Any]],
    family: str,
    alpha: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One *family* of paired bootstrap tests, reported with and without correction.

    ``d`` is [n_test, n_boot] of paired differences sharing bootstrap indices.

    Twelve increments each tested at alpha = 0.05 carry a familywise error rate
    of 1 - 0.95^12 = 46%, so "every step is significant" is not a claim the
    uncorrected intervals can support on their own. Three corrections are
    recorded and they answer different questions:

    * **Holm-Bonferroni** adjusted p-values - familywise control, valid under
      arbitrary dependence. This is the one a claim about *all* the steps should
      be read off.
    * **Benjamini-Hochberg** adjusted p-values - false-discovery control, the
      more forgiving standard, reported so the gap between the two is visible.
    * **A simultaneous sup-t band** - the smallest constant ``c`` for which
      ``delta +/- c*se`` covers every test at once in 95% of replicates. Unlike
      Bonferroni's alpha/m per-test intervals it uses the correlation between
      layers rather than assuming the worst case, so it is the tightest interval
      that still supports a statement about the whole trajectory.

    The uncorrected ``excludes_zero`` is kept so earlier numbers stay
    comparable, but it is no longer the field a family-level claim reads off.
    """
    n_test, n_boot = d.shape
    deltas = d.mean(axis=1)
    se = d.std(axis=1, ddof=1)
    p_raw = [boot_pvalue(d[s], n_boot) for s in range(n_test)]
    p_holm, p_bh = holm_adjust(p_raw), bh_adjust(p_raw)

    # sup-t: the largest standardised deviation anywhere in the family, per replicate
    se_safe = np.where(se > 0, se, np.inf)
    t = np.abs(d - deltas[:, None]) / se_safe[:, None]
    c = float(np.percentile(t.max(axis=0), 100 * (1 - alpha)))
    bonf_pct = 100 * (alpha / n_test) / 2

    tests = []
    for s in range(n_test):
        lo_s, hi_s = float(deltas[s] - c * se[s]), float(deltas[s] + c * se[s])
        blo, bhi = np.percentile(d[s], [bonf_pct, 100 - bonf_pct])
        ci = ci_of(d[s], alpha)
        tests.append({
            **labels[s],
            "delta": float(deltas[s]),
            "ci": ci,
            "excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0),
            "p_boot": p_raw[s],
            "p_normal": normal_pvalue(float(deltas[s]), float(se[s])),
            "p_holm": p_holm[s],
            "p_bh": p_bh[s],
            "significant_uncorrected": bool(p_raw[s] < alpha),
            "significant_holm": bool(p_holm[s] < alpha),
            "significant_bh": bool(p_bh[s] < alpha),
            "ci_bonferroni": {"lo": float(blo), "hi": float(bhi)},
            "ci_simultaneous": {"lo": lo_s, "hi": hi_s},
            "excludes_zero_bonferroni": bool(blo > 0 or bhi < 0),
            "excludes_zero_simultaneous": bool(lo_s > 0 or hi_s < 0),
        })

    meta = {
        "family": family,
        "n_tests": n_test,
        "alpha": alpha,
        "n_boot": n_boot,
        "p_value_resolution": 2.0 / (n_boot + 1),
        "uncorrected_familywise_error_rate": float(1.0 - (1.0 - alpha) ** n_test),
        "bonferroni_per_test_alpha": alpha / n_test,
        "supt_critical_value": c,
        "n_significant_uncorrected": int(sum(x["significant_uncorrected"] for x in tests)),
        "n_significant_holm": int(sum(x["significant_holm"] for x in tests)),
        "n_significant_bh": int(sum(x["significant_bh"] for x in tests)),
        "n_excludes_zero_simultaneous": int(sum(x["excludes_zero_simultaneous"] for x in tests)),
        "all_significant_holm": bool(all(x["significant_holm"] for x in tests)),
        "max_p_holm": float(max(x["p_holm"] for x in tests)),
    }
    return tests, meta


def zero_crossings(d: np.ndarray) -> np.ndarray:
    """Depth at which each column of ``d`` [lens point, B] first turns positive.

    Linear interpolation between the two lens points that bracket the sign
    change, in lens-point units (0 = mask+position, ``n_layers`` = the model's
    real output). ``nan`` where a column never turns positive.
    """
    pos = d > 0
    any_pos = pos.any(axis=0)
    idx = np.argmax(pos, axis=0)                 # first positive lens point
    cols = np.arange(d.shape[1])
    prev = np.maximum(idx - 1, 0)
    y1, y2 = d[prev, cols], d[idx, cols]
    gap = y2 - y1
    frac = np.where((idx > 0) & (gap != 0), -y1 / np.where(gap == 0, 1.0, gap), 0.0)
    return np.where(any_pos, prev + frac, np.nan)


def crossing_summary(
    d: np.ndarray,
    keys_traj: List[str],
    n_layers: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Where a paired margin curve crosses zero, with an interval on the crossing.

    "Negative at layer 4, positive at layer 5" localises the crossing only to a
    whole block, and stating it that way makes the answer depend on which depths
    happen to have been sampled. Interpolating the curve inside *each* bootstrap
    replicate and taking percentiles of the resulting distribution gives an
    interval on the crossing depth itself, which is both sharper and the
    quantity the claim is actually about. It is well defined precisely because
    the replicates are paired across layers: within one replicate the whole
    curve moves together.
    """
    point = float(zero_crossings(d.mean(axis=1)[:, None])[0])
    per_boot = zero_crossings(d)
    ok = ~np.isnan(per_boot)
    n_ok = int(ok.sum())
    res: Dict[str, Any] = {
        "lens_point": None if math.isnan(point) else point,
        "relative_depth": None if math.isnan(point) else point / n_layers,
        "n_boot": int(d.shape[1]),
        "replicates_with_a_crossing": n_ok,
        "fraction_with_a_crossing": float(n_ok / d.shape[1]),
    }
    if not math.isnan(point):
        j = int(math.floor(point))
        res["brackets"] = [keys_traj[j], keys_traj[min(j + 1, len(keys_traj) - 1)]]
    # An interval needs replicates to be an interval. When almost none of them
    # cross, the crossing depth is not estimated - it is an artefact of the one
    # replicate that happened to - and a percentile range over a single draw
    # would report 0.0 with a nan spread as though it meant something.
    if n_ok >= 2:
        lo, hi = np.percentile(per_boot[ok], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        res["ci"] = {"lo": float(lo), "hi": float(hi),
                     "se": float(np.std(per_boot[ok], ddof=1))}
        res["ci_relative_depth"] = {"lo": float(lo / n_layers),
                                    "hi": float(hi / n_layers)}
    return res


def bootstrap_report(
    acc: LensStats,
    keys_traj: List[str],
    keys_all: List[str],
    n_layers: int,
    n_boot: int,
    seed: int,
    unit: str,
    alpha: float = 0.05,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Everything the bootstrap has to say, under one resampling unit.

    Returns the report block and the raw draws, so the caller can also hang
    per-layer intervals off the headline unit without drawing twice.
    """
    draws = bootstrap_draws(acc.sums, acc.counts, n_boot, seed, unit)
    m_of = {name: m for m, name in enumerate(BOOT_METRICS)}
    p_j = keys_all.index("proto")
    n_traj = len(keys_traj)

    steps: Dict[str, Any] = {}
    margin: Dict[str, Any] = {}
    crossing: Dict[str, Any] = {}
    families: Dict[str, Any] = {}
    for name, m in m_of.items():
        # Paired increments between consecutive lens points, and the paired
        # margin over the position-only prototype. Both are differences taken
        # *within* each replicate, which is the whole point of sharing indices.
        inc = draws[m, 1:n_traj, :] - draws[m, 0:n_traj - 1, :]
        steps[name], families["steps." + name] = paired_tests(
            inc,
            [{"from": keys_traj[s], "to": keys_traj[s + 1]} for s in range(n_traj - 1)],
            f"consecutive-layer increments in {name}", alpha,
        )

        d = draws[m, 0:n_traj, :] - draws[m, p_j, :][None, :]
        margin[name], families["margin_vs_prototype." + name] = paired_tests(
            d, [{"layer": keys_traj[s]} for s in range(n_traj)],
            f"per-layer margin over the positional prototype in {name}", alpha,
        )
        for e in margin[name]:
            e["beats_prototype"] = bool(e["ci"]["lo"] > 0)
            e["beats_prototype_holm"] = bool(e["significant_holm"] and e["delta"] > 0)
        crossing[name] = crossing_summary(d, keys_traj, n_layers, alpha)

    gb = draws[m_of["gallery_b_top1"], 0:n_traj, :]
    rise = gb[-1] - gb[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(rise > 1e-9, np.diff(gb, axis=0).max(axis=0) / rise, np.nan)

    block = {
        "n_boot": int(n_boot),
        "resample_unit": unit,
        "paired_across_layers": True,
        "alpha": alpha,
        "steps": steps,
        "margin_vs_prototype": margin,
        "margin_zero_crossing": crossing,
        "multiple_comparisons": {
            "why": (
                "the trajectory asks one question of every layer at once, so the "
                "tests come in families; at alpha=0.05 uncorrected, a family of 12 "
                "has a 46% chance of at least one false positive"
            ),
            "reported": [
                "p_boot: two-sided percentile bootstrap p-value, floored at 2/(B+1)",
                "p_normal: the same test through the bootstrap SE, no floor",
                "p_holm: Holm-Bonferroni adjusted, familywise error control",
                "p_bh: Benjamini-Hochberg adjusted, false discovery control",
                "ci_simultaneous: sup-t band covering the whole family at 95%",
                "ci_bonferroni: per-test percentile interval at alpha/m",
            ],
            "families": families,
        },
        "final_gallery_b_top1_ci": ci_of(gb[-1], alpha),
        "total_gallery_b_rise_ci": ci_of(rise, alpha),
        "largest_increment_fraction_ci": (
            ci_of(frac[~np.isnan(frac)], alpha) if np.any(~np.isnan(frac)) else None
        ),
        # A criterion whose threshold sits inside its own interval cannot be
        # settled by a pass/fail reading, and no feasible amount of extra data
        # will settle it when the quantity genuinely lives that close to the
        # line. What the bootstrap *can* say exactly is how much of its mass
        # falls on each side. Reporting that fraction turns "AMBIGUOUS" into a
        # number, which is a strictly better answer than a shrug: it says which
        # way the evidence points and how strongly.
        "largest_increment_fraction_below_threshold_probability": (
            float(np.mean(frac[~np.isnan(frac)] < C2_THRESHOLD))
            if np.any(~np.isnan(frac)) else None
        ),
        "largest_increment_fraction_threshold": C2_THRESHOLD,
    }
    return block, draws


def summarise(
    acc: LensStats,
    lens_points: List[Any],
    arch: Dict[str, int],
    model: Any,                       # LensModel, or a name/meta stub from cache
    args: argparse.Namespace,
    n_img: int,
    n_masks: int,
    l0_spread: float,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    chance_a = 1.0 / arch["num_patches"]
    chance_b = 1.0 / n_img

    by_layer: Dict[str, Any] = {}
    for lp in list(lens_points) + ["proto"]:
        k = str(lp)
        by_layer[k] = {
            "n": acc.n(k),
            "cos_target": acc.mean(k, "cos_target"),
            "cos_target_std": acc.std(k, "cos_target"),
            "cos_centered": acc.mean(k, "cos_centered"),
            "cos_centered_std": acc.std(k, "cos_centered"),
            "mse": acc.mean(k, "mse"),
            "gallery_a_top1": acc.mean(k, "a_top1"),
            "gallery_a_top5": acc.mean(k, "a_top5"),
            "gallery_a_mrr": acc.mean(k, "a_mrr"),
            "gallery_b_top1": acc.mean(k, "b_top1"),
            "gallery_b_mrr": acc.mean(k, "b_mrr"),
        }

    # ---- cluster bootstrap --------------------------------------------------
    keys_traj = [str(lp) for lp in lens_points]
    keys_all = keys_traj + ["proto"]
    boot: Dict[str, Any] = {"n_boot": 0}
    if args.bootstrap > 0:
        t_boot = time.time()
        boot, draws = bootstrap_report(acc, keys_traj, keys_all,
                                       arch["predictor_depth"], args.bootstrap,
                                       args.seed * 7919 + 1, "image")
        m_of = {name: m for m, name in enumerate(BOOT_METRICS)}
        for j, k in enumerate(keys_all):
            for name, m in m_of.items():
                by_layer[k][name + "_ci"] = ci_of(draws[m, j, :])
        del draws

        # Masks are the second sampling axis, and the headline interval holds
        # them fixed. With enough of them the two-way cluster bootstrap can say
        # whether a claim survives another *draw of masks* as well as another
        # draw of images. With four it cannot: four clusters do not carry a
        # usable variance component, and pretending otherwise would produce a
        # confident-looking interval built on three degrees of freedom.
        if n_masks >= args.min_masks_two_way:
            two_way, _ = bootstrap_report(acc, keys_traj, keys_all,
                                          arch["predictor_depth"], args.bootstrap,
                                          args.seed * 7919 + 2, "image_mask")
            two_way["note"] = (
                f"images and masks both resampled ({n_masks} masks); generalises "
                "past the particular masks drawn, and is therefore wider than the "
                "image-only interval by construction"
            )
            boot["cluster_image_mask"] = two_way
        else:
            boot["cluster_image_mask"] = None
            boot["cluster_image_mask_skipped"] = (
                f"{n_masks} masks is too few clusters to estimate a mask variance "
                f"component (need >= {args.min_masks_two_way}); the reported "
                "intervals are conditional on these masks"
            )
        boot["seconds"] = round(time.time() - t_boot, 1)
        fam = boot["multiple_comparisons"]["families"]["steps.gallery_b_top1"]
        print(f"[bootstrap] {args.bootstrap} image-level replicates, paired across "
              f"{len(keys_traj)} lens points ({boot['seconds']}s); "
              f"{fam['n_significant_holm']}/{fam['n_tests']} Gallery-B increments "
              f"survive Holm correction")

    # ---- is the model itself worth looking through a lens? ----------------
    # If the predictor's real output is no better than the positional prototype,
    # the trajectory is measuring an untrained model, not a representation.
    out_q, proto_q = by_layer["out"], by_layer["proto"]
    quality = {
        "output_cos_target": out_q["cos_target"],
        "prototype_cos_target": proto_q["cos_target"],
        "output_gallery_b_top1": out_q["gallery_b_top1"],
        "prototype_gallery_b_top1": proto_q["gallery_b_top1"],
        "output_beats_positional_prototype": bool(
            out_q["cos_target"] > proto_q["cos_target"]
            and out_q["gallery_b_top1"] > proto_q["gallery_b_top1"]
        ),
    }

    # ---- pre-registered verdict -------------------------------------------
    keys = [str(lp) for lp in lens_points]
    b = np.array([by_layer[k]["gallery_b_top1"] for k in keys])
    final, first = b[-1], b[0]
    increments = np.diff(b)
    total_rise = final - first
    largest_frac = float(increments.max() / total_rise) if total_rise > 1e-9 else float("nan")
    crit = {
        "final_gallery_b_top1": float(final),
        "chance_gallery_b": chance_b,
        "c1_final_at_least_2x_chance": bool(final >= 2 * chance_b),
        "c2_largest_single_layer_increment_fraction": largest_frac,
        "c2_emerges_across_layers": bool(largest_frac < C2_THRESHOLD) if total_rise > 1e-9 else False,
        "c3_layer0_within_2x_chance": bool(first <= 2 * chance_b),
        "layer0_gallery_b_top1": float(first),
    }
    if boot["n_boot"]:
        crit["c1_final_ci"] = boot["final_gallery_b_top1_ci"]
        crit["c1_holds_at_ci_lower_bound"] = bool(
            boot["final_gallery_b_top1_ci"]["lo"] >= 2 * chance_b)
        if boot["largest_increment_fraction_ci"]:
            crit["c2_largest_increment_fraction_ci"] = boot["largest_increment_fraction_ci"]
            crit["c2_holds_at_ci_upper_bound"] = bool(
                boot["largest_increment_fraction_ci"]["hi"] < C2_THRESHOLD)
            crit["c2_probability_below_threshold"] =                 boot["largest_increment_fraction_below_threshold_probability"]
        tw = boot.get("cluster_image_mask")
        if tw:
            crit["c1_final_ci_image_mask"] = tw["final_gallery_b_top1_ci"]
            crit["c1_holds_at_ci_lower_bound_image_mask"] = bool(
                tw["final_gallery_b_top1_ci"]["lo"] >= 2 * chance_b)
            if tw["largest_increment_fraction_ci"]:
                crit["c2_largest_increment_fraction_ci_image_mask"] = \
                    tw["largest_increment_fraction_ci"]
                crit["c2_holds_at_ci_upper_bound_image_mask"] = bool(
                    tw["largest_increment_fraction_ci"]["hi"] < C2_THRESHOLD)
                crit["c2_probability_below_threshold_image_mask"] =                     tw["largest_increment_fraction_below_threshold_probability"]
    crit["model_predicts_at_all"] = quality["output_beats_positional_prototype"]
    if not quality["output_beats_positional_prototype"]:
        # Nothing to see: the failure is upstream of the lens.
        crit["verdict"] = "MODEL TOO WEAK"
    elif (crit["c1_final_at_least_2x_chance"]
          and crit["c2_emerges_across_layers"]
          and crit["c3_layer0_within_2x_chance"]):
        crit["verdict"] = "INFORMATIVE"
    else:
        crit["verdict"] = "AMBIGUOUS"

    # Robustness of the verdict under the bootstrap. The pre-registered rule is
    # evaluated on point estimates and is deliberately left exactly as
    # registered - rewriting a criterion after seeing the interval is the thing
    # pre-registration exists to prevent. What the CI can honestly add is
    # whether the registered call survives its own uncertainty: a criterion that
    # passes at 79% against an 80% threshold while its interval reaches 85% has
    # not really been met, and the writeup should say so rather than bank the
    # PASS. Recorded separately, never overwriting `verdict`.
    for suffix in ("", "_image_mask"):
        flags = [crit[k] for k in ("c1_holds_at_ci_lower_bound" + suffix,
                                   "c2_holds_at_ci_upper_bound" + suffix) if k in crit]
        if not flags:
            continue
        if crit["verdict"] == "INFORMATIVE":
            crit["verdict_robust_to_ci" + suffix] = bool(all(flags))
            crit["verdict_if_ci_bounds_used" + suffix] = (
                "INFORMATIVE" if all(flags) else "AMBIGUOUS")
        else:
            crit["verdict_robust_to_ci" + suffix] = None
            crit["verdict_if_ci_bounds_used" + suffix] = crit["verdict"]

    return {
        "experiment": "task6_latent_lens_trajectory",
        "model": model.name,
        "model_meta": model.meta,
        "architecture": arch,
        "config": {
            "device": args.device,
            "n_images": n_img,
            "n_masks": n_masks,
            "seed": args.seed,
            "bootstrap": int(args.bootstrap),
            "data_dir": args.data_dir,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "chance": {"gallery_a": chance_a, "gallery_b": chance_b},
        "layer0_target_token_spread_across_images": l0_spread,
        "by_layer": by_layer,
        "bootstrap": boot,
        "model_quality": quality,
        "verdict": crit,
        "rows": rows if (args.save_rows and rows) else [],
    }


# ---------------------------------------------------------------------------
# Trace cache: the model pass and the statistics, decoupled
# ---------------------------------------------------------------------------


def cache_path(args: argparse.Namespace, model: Any, n_img: int, n_masks: int) -> str:
    """Where a completed trace's sufficient statistics live.

    The ViT-H trace is hours of GPU time and the statistics that come out of it
    are a few megabytes, so every question asked *after* the trace - a different
    correction, more bootstrap replicates, a new derived quantity - should cost
    seconds rather than a rerun. Keyed on everything that changes the trace.
    """
    ckpt = args.mini_checkpoint if args.model == "mini" else args.checkpoint
    tag = os.path.basename(str(ckpt)).split(".")[0]
    d = hashlib.sha1(os.path.abspath(args.data_dir).encode()).hexdigest()[:8]
    return os.path.join(args.cache_dir,
                        f"{tag}__{n_img}img_{n_masks}masks_seed{args.seed}_{d}.npz")


def trace_meta(arch: Dict[str, int], model: Any, args: argparse.Namespace,
               n_img: int, n_masks: int, l0_spread: float,
               lens_points: List[Any]) -> Dict[str, Any]:
    return {
        "model": model.name,
        "model_meta": model.meta,
        "architecture": arch,
        "lens_points": [str(lp) for lp in lens_points],
        "n_images": n_img,
        "n_masks": n_masks,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "device": args.device,
        "l0_spread": l0_spread,
        "torch": torch.__version__,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def summarise_cached(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Rebuild a full run report from cached sufficient statistics, no model."""
    acc, meta = LensStats.load(path)
    stub = SimpleNamespace(name=meta["model"], meta=meta["model_meta"])
    sub = argparse.Namespace(**vars(args))
    sub.data_dir, sub.seed, sub.device = meta["data_dir"], meta["seed"], meta["device"]
    sub.save_rows = False
    print(f"[cache] {os.path.basename(path)}: {meta['model']}, "
          f"{meta['n_images']} images x {meta['n_masks']} masks")
    return summarise(acc, list(meta["lens_points"]), meta["architecture"], stub, sub,
                     int(meta["n_images"]), int(meta["n_masks"]),
                     float(meta["l0_spread"]), None)


def print_report(res: Dict[str, Any]) -> None:
    arch = res["architecture"]
    print("\n" + "=" * 86)
    print(f"LATENT LENS TRAJECTORY - {res['model']}")
    print("=" * 86)
    print(f"predictor: d={arch['predictor_embed_dim']} L={arch['predictor_depth']} "
          f"-> target space d={arch['embed_dim']};  "
          f"{res['config']['n_images']} images, {arch['num_patches']} patches")
    ca, cb = res["chance"]["gallery_a"], res["chance"]["gallery_b"]
    print(f"chance: gallery A (which patch) {ca*100:.2f}%   "
          f"gallery B (which image) {cb*100:.2f}%\n")

    boot = res.get("bootstrap", {"n_boot": 0})
    nb = boot["n_boot"]
    if nb:
        print(f"CIs: 95%, {nb} bootstrap replicates resampling IMAGES "
              f"(paired across layers)\n")

    def band(d: Dict[str, Any], key: str, scale: float = 1.0, w: int = 16) -> str:
        c = d.get(key + "_ci")
        return (f"[{c['lo']*scale:+.3f},{c['hi']*scale:+.3f}]".rjust(w)) if c else "".rjust(w)

    print("6a - ABSOLUTE: how close is the projection to the one true target?")
    print(f"  {'layer':<6} {'cos_target':>11} {'95% CI':>16} "
          f"{'cos_centered':>13} {'95% CI':>16} {'mse':>9}")
    for k, d in res["by_layer"].items():
        print(f"  {k:<6} {d['cos_target']:>11.4f} {band(d, 'cos_target')} "
              f"{d['cos_centered']:>13.4f} {band(d, 'cos_centered')} {d['mse']:>9.4f}")

    print("\n6b - RELATIVE: is the true target ranked above a gallery of wrong ones?")
    print(f"  {'layer':<6} {'A@1':>7} {'A@5':>7} {'B@1':>7} {'95% CI (B@1)':>18} "
          f"{'B_mrr':>7}")
    for k, d in res["by_layer"].items():
        print(f"  {k:<6} {d['gallery_a_top1']*100:>6.1f}% "
              f"{d['gallery_a_top5']*100:>6.1f}% {d['gallery_b_top1']*100:>6.1f}% "
              f"{band(d, 'gallery_b_top1', 100.0, 18)} {d['gallery_b_mrr']:>7.3f}")

    if nb:
        print("\n6a paired margin over the position-only prototype (cos_target):")
        print(f"  {'layer':<6} {'delta':>8} {'95% CI':>19} {'p (Holm)':>10}  clears prototype")
        for m in boot["margin_vs_prototype"]["cos_target"]:
            print(f"  {m['layer']:<6} {m['delta']:>+8.4f} "
                  + f"[{m['ci']['lo']:+.4f},{m['ci']['hi']:+.4f}]".rjust(19)
                  + f" {m['p_holm']:>10.2e}  {m['beats_prototype_holm']}")
        first = next((m["layer"] for m in boot["margin_vs_prototype"]["cos_target"]
                      if m["beats_prototype_holm"]), None)
        print(f"  -> first layer clearing the prototype (Holm-corrected): {first}")
        cross = boot.get("margin_zero_crossing", {}).get("cos_target", {})
        if cross.get("lens_point") is not None:
            c_ci = cross.get("ci", {})
            print(f"  -> interpolated zero crossing: layer {cross['lens_point']:.2f}"
                  + (f"  95% CI [{c_ci['lo']:.2f}, {c_ci['hi']:.2f}]" if c_ci else "")
                  + f"  (relative depth {cross['relative_depth']:.3f}; "
                  f"{cross['fraction_with_a_crossing']*100:.1f}% of replicates cross)")

        fam = boot["multiple_comparisons"]["families"]
        f_gb = fam["steps.gallery_b_top1"]
        print(f"\npaired layer-to-layer increments, {f_gb['n_tests']} per metric. "
              f"Uncorrected, a family that size\n"
              f"carries a {f_gb['uncorrected_familywise_error_rate']*100:.0f}% chance of "
              f"at least one false positive, so the\n"
              f"column that decides the claim is Holm, not the raw interval.")
        print(f"  {'step':<10} {'d cos_centered':>15} {'p Holm':>9} "
              f"{'d B@1':>8} {'95% CI':>18} {'p Holm':>9} {'p BH':>9} {'sup-t':>7}")
        cc = boot["steps"]["cos_centered"]
        gb = boot["steps"]["gallery_b_top1"]
        for a, b in zip(cc, gb):
            print(f"  {a['from']+'->'+a['to']:<10} {a['delta']:>+15.4f} "
                  f"{a['p_holm']:>9.2e} "
                  + f"{b['delta']*100:>+7.1f}% "
                  + f"[{b['ci']['lo']*100:+.1f},{b['ci']['hi']*100:+.1f}]".rjust(18)
                  + f" {b['p_holm']:>9.2e} {b['p_bh']:>9.2e}"
                  + ("     yes" if b["excludes_zero_simultaneous"] else "      no"))
        for key in ("steps.cos_centered", "steps.gallery_b_top1"):
            f = fam[key]
            print(f"  [{key}] {f['n_significant_uncorrected']}/{f['n_tests']} uncorrected, "
                  f"{f['n_significant_holm']}/{f['n_tests']} Holm, "
                  f"{f['n_significant_bh']}/{f['n_tests']} BH, "
                  f"{f['n_excludes_zero_simultaneous']}/{f['n_tests']} sup-t; "
                  f"largest adjusted p = {f['max_p_holm']:.2e} "
                  f"(bootstrap resolution {f['p_value_resolution']:.1e})")

        tw = boot.get("cluster_image_mask")
        if tw:
            f = tw["multiple_comparisons"]["families"]["steps.gallery_b_top1"]
            c2 = tw.get("largest_increment_fraction_ci")
            print(f"\nalso resampling MASKS ({res['config']['n_masks']} of them):")
            print(f"  Gallery-B increments surviving Holm: "
                  f"{f['n_significant_holm']}/{f['n_tests']}")
            print(f"  final B@1 CI [{tw['final_gallery_b_top1_ci']['lo']*100:.1f}, "
                  f"{tw['final_gallery_b_top1_ci']['hi']*100:.1f}]%"
                  + (f"   largest-step fraction CI [{c2['lo']*100:.0f}, {c2['hi']*100:.0f}]%"
                     if c2 else ""))
            cr = tw.get("margin_zero_crossing", {}).get("cos_target", {})
            if cr.get("lens_point") is not None and cr.get("ci"):
                print(f"  zero crossing layer {cr['lens_point']:.2f} "
                      f"95% CI [{cr['ci']['lo']:.2f}, {cr['ci']['hi']:.2f}]")
        elif boot.get("cluster_image_mask_skipped"):
            print(f"\n[note] {boot['cluster_image_mask_skipped']}")

    q = res["model_quality"]
    print(f"\nmodel quality gate: predictor output vs positional-prototype baseline")
    print(f"  cos_target  output {q['output_cos_target']:+.4f}  vs  "
          f"prototype {q['prototype_cos_target']:+.4f}")
    print(f"  gallery B@1 output {q['output_gallery_b_top1']*100:.1f}%  vs  "
          f"prototype {q['prototype_gallery_b_top1']*100:.1f}%")
    print(f"  output beats a position-only predictor: "
          f"{q['output_beats_positional_prototype']}")

    v = res["verdict"]
    print(f"\npre-registered verdict: {v['verdict']}")
    print(f"  C1 final gallery-B top1 >= 2x chance : {v['c1_final_at_least_2x_chance']} "
          f"({v['final_gallery_b_top1']*100:.1f}% vs {v['chance_gallery_b']*200:.1f}%)")
    print(f"  C2 emerges across layers (max step < 80% of rise) : "
          f"{v['c2_emerges_across_layers']} (largest step = "
          f"{v['c2_largest_single_layer_increment_fraction']*100:.0f}% of the rise)")
    print(f"  C3 layer-0 at the architectural floor : {v['c3_layer0_within_2x_chance']} "
          f"({v['layer0_gallery_b_top1']*100:.1f}%)")
    if "c1_holds_at_ci_lower_bound" in v:
        print(f"  C1 still holds at the CI lower bound : "
              f"{v['c1_holds_at_ci_lower_bound']} "
              f"(lo {v['c1_final_ci']['lo']*100:.1f}%)")
    if "c2_holds_at_ci_upper_bound" in v:
        print(f"  C2 still holds at the CI upper bound : "
              f"{v['c2_holds_at_ci_upper_bound']} "
              f"(hi {v['c2_largest_increment_fraction_ci']['hi']*100:.0f}% of the rise)")
    if "c2_holds_at_ci_upper_bound_image_mask" in v:
        print(f"  C2 still holds once MASKS are resampled too : "
              f"{v['c2_holds_at_ci_upper_bound_image_mask']} "
              f"(hi {v['c2_largest_increment_fraction_ci_image_mask']['hi']*100:.0f}% "
              f"of the rise)")
    # Only meaningful when there is a rise to take a fraction of; on a model that
    # never leaves chance the ratio is noise over noise.
    if (v.get("c2_probability_below_threshold") is not None
            and v.get("model_predicts_at_all")):
        pm = v.get("c2_probability_below_threshold_image_mask")
        print(f"  C2 bootstrap mass below the {C2_THRESHOLD:.0%} threshold : "
              f"{v['c2_probability_below_threshold']*100:.1f}% of replicates"
              + (f" ({pm*100:.1f}% resampling masks too)" if pm is not None else ""))
    if v.get("verdict_robust_to_ci") is False:
        print(f"  !! the registered PASS does not survive its own CI: on the CI "
              f"bounds this reads {v['verdict_if_ci_bounds_used']}")
    if v.get("verdict_robust_to_ci_image_mask") is False:
        print(f"  !! nor does it survive resampling the masks: "
              f"{v['verdict_if_ci_bounds_used_image_mask']}")
    if v["verdict"] == "MODEL TOO WEAK":
        print("\n  -> the predictor does not beat a position-only baseline, so its\n"
              "     residual trajectory carries no representation to observe. This is a\n"
              "     statement about the checkpoint, not about the Latent Lens.")
    if v["verdict"] == "AMBIGUOUS":
        print("\n  -> watch item triggered: escalate to QK-routing vs OV-content "
              "decomposition\n     and path patching across the predictor.")


def make_plots(results: List[Dict[str, Any]], outdir: str) -> None:
    """Draw the trajectories against *relative* predictor depth.

    Predictors of different depth (4 layers vs 12) cannot share an integer layer
    axis - doing so puts one model's output where another's middle is. Relative
    depth (layer / n_layers, with the model's real output at 1.0) is also the
    comparison the experiment is actually about: whether identity emerges
    gradually along the stack or all at once.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for ci, res in enumerate(results):
        col = colours[ci % len(colours)]
        d = res["by_layer"]
        keys = [k for k in d if k != "proto"]
        n_l = res["architecture"]["predictor_depth"]
        xs = np.array([(int(k) if k != "out" else n_l) / n_l for k in keys])
        short = res["model"].replace("official ", "").replace("mini (", "").replace(".pth)", "")
        weak = res["verdict"]["verdict"] == "MODEL TOO WEAK"
        lbl = short + (" [untrained]" if weak else "")
        style = dict(color=col, alpha=0.35 if weak else 1.0)

        def ci_band(ax: Any, key: str, scale: float = 1.0) -> None:
            if not all(key + "_ci" in d[k] for k in keys):
                return
            lo = np.array([d[k][key + "_ci"]["lo"] for k in keys]) * scale
            hi = np.array([d[k][key + "_ci"]["hi"] for k in keys]) * scale
            ax.fill_between(xs, lo, hi, color=col, alpha=0.18 if not weak else 0.08, lw=0)

        ci_band(axes[0], "cos_target")
        ci_band(axes[0], "cos_centered")
        axes[0].plot(xs, [d[k]["cos_target"] for k in keys], "-o", ms=4, label=lbl, **style)
        axes[0].plot(xs, [d[k]["cos_centered"] for k in keys], "--s", ms=3,
                     label=f"{short}: position removed", **style)
        axes[0].axhline(d["proto"]["cos_target"], color=col, ls=":", lw=1, alpha=0.6)
        ci_band(axes[1], "gallery_a_top1", 100.0)
        axes[1].plot(xs, [d[k]["gallery_a_top1"] * 100 for k in keys], "-o", ms=4,
                     label=lbl, **style)
        axes[1].axhline(d["proto"]["gallery_a_top1"] * 100, color=col, ls=":", lw=1, alpha=0.6)
        ci_band(axes[2], "gallery_b_top1", 100.0)
        axes[2].plot(xs, [d[k]["gallery_b_top1"] * 100 for k in keys], "-o", ms=4,
                     label=lbl, **style)

    for ax, chance in zip(axes, [None,
                                 results[0]["chance"]["gallery_a"] * 100,
                                 results[0]["chance"]["gallery_b"] * 100]):
        ax.set_xlabel("relative predictor depth  (0 = mask+position, 1 = model output)")
        ax.set_xlim(-0.03, 1.03)
        if chance is not None:
            ax.axhline(chance, color="k", ls="--", lw=0.8)
            ax.text(0.02, chance, " chance", fontsize=7, va="bottom")
        ax.legend(fontsize=7)

    axes[0].set_title("Alignment with the true target representation\n"
                      "(dotted = position-only prototype baseline)")
    axes[0].set_ylabel("cosine similarity")
    axes[1].set_title("Gallery A: which patch of this image?\n(localisation)")
    axes[1].set_ylabel("top-1 %")
    axes[2].set_title("Gallery B: which image, same patch?\n(pure content identity)")
    axes[2].set_ylabel("top-1 %")
    fig.tight_layout()
    path = os.path.join(outdir, "latent_lens_trajectory.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[save] {path}")
    make_absolute_plot(results, outdir)
    make_margin_plot(results, outdir)


def make_absolute_plot(results: List[Dict[str, Any]], outdir: str) -> None:
    """Task 6a on its own axes: alignment with the one true target, no gallery.

    Two panels because the raw number cannot be read alone. At layer 0 a target
    token carries no image information at all, yet raw cosine is already well
    above zero - target space has a large shared mean component, so *any* vector
    pointing into the bulk scores respectably. The position-only prototype makes
    that concrete: it is a "predictor" that never sees the image and still scores
    a substantial cosine. The right panel therefore plots the *paired* margin of
    the lens over that prototype, which is the part of 6a a positional account
    cannot explain, with the depth at which its CI first clears zero marked.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for ci, res in enumerate(results):
        col = colours[ci % len(colours)]
        d = res["by_layer"]
        keys = [k for k in d if k != "proto"]
        n_l = res["architecture"]["predictor_depth"]
        xs = np.array([(int(k) if k != "out" else n_l) / n_l for k in keys])
        short = res["model"].replace("official ", "").replace("mini (", "").replace(".pth)", "")
        weak = res["verdict"]["verdict"] == "MODEL TOO WEAK"
        style = dict(color=col, alpha=0.35 if weak else 1.0)
        lbl = short + (" [untrained]" if weak else "")

        for key, marker, ls, tag in (("cos_target", "o", "-", "raw"),
                                     ("cos_centered", "s", "--", "position removed")):
            if all(key + "_ci" in d[k] for k in keys):
                lo = np.array([d[k][key + "_ci"]["lo"] for k in keys])
                hi = np.array([d[k][key + "_ci"]["hi"] for k in keys])
                axes[0].fill_between(xs, lo, hi, color=col, lw=0,
                                     alpha=0.08 if weak else 0.18)
            axes[0].plot(xs, [d[k][key] for k in keys], ls, marker=marker, ms=4,
                         label=f"{lbl}: {tag}", **style)
        axes[0].axhline(d["proto"]["cos_target"], color=col, ls=":", lw=1.2, alpha=0.7)
        axes[0].text(1.005, d["proto"]["cos_target"], " position-only", fontsize=6.5,
                     color=col, va="center")

        boot = res.get("bootstrap", {"n_boot": 0})
        if boot["n_boot"]:
            marg = boot["margin_vs_prototype"]["cos_target"]
            mx = np.array([(int(m["layer"]) if m["layer"] != "out" else n_l) / n_l
                           for m in marg])
            axes[1].fill_between(mx, [m["ci"]["lo"] for m in marg],
                                 [m["ci"]["hi"] for m in marg], color=col, lw=0,
                                 alpha=0.08 if weak else 0.18)
            axes[1].plot(mx, [m["delta"] for m in marg], "-o", ms=4, label=lbl, **style)
            cross = boot.get("margin_zero_crossing", {}).get("cos_target", {})
            if cross.get("relative_depth") is not None and not weak:
                rci = cross.get("ci_relative_depth")
                if rci:
                    axes[1].axvspan(rci["lo"], rci["hi"], color=col, alpha=0.13, lw=0)
                axes[1].axvline(cross["relative_depth"], color=col, ls=":", lw=1.2,
                                alpha=0.8)

    for ax in axes:
        ax.set_xlabel("relative predictor depth  (0 = mask+position, 1 = model output)")
        ax.set_xlim(-0.03, 1.03)
        ax.legend(fontsize=7)
    axes[0].set_title("6a: cosine to the true target for this patch\n"
                      "(no gallery; bands = 95% image-level bootstrap CI)")
    axes[0].set_ylabel("cosine similarity")
    axes[1].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[1].set_title("6a: paired margin over the position-only prototype\n"
                      "(vertical line + band = interpolated zero crossing, 95% CI)")
    axes[1].set_ylabel("cos_target(lens) - cos_target(prototype)")
    fig.tight_layout()
    path = os.path.join(outdir, "latent_lens_absolute_6a.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[save] {path}")


def make_margin_plot(results: List[Dict[str, Any]], outdir: str,
                     metric: str = "cos_target") -> None:
    """The complete per-layer margin curve, and the crossing on its own axis.

    The 6a claim is a statement about a *depth*: below some point in the stack
    the lens is worse than knowing only the position, above it better. Saying
    "the CI first clears zero at layer 5" reports that depth to the nearest
    block and makes it depend on where the lens happens to have been applied.
    The left panel therefore draws the whole curve with both a pointwise and a
    simultaneous band, and the right panel zooms on the sign change and shows
    the interpolated crossing with an interval on the crossing itself.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [r for r in results
              if r.get("bootstrap", {}).get("n_boot")
              and metric in r["bootstrap"].get("margin_vs_prototype", {})]
    if not usable:
        return

    os.makedirs(outdir, exist_ok=True)
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    zoom_pick = None

    for ci, res in enumerate(results):
        col = colours[ci % len(colours)]
        boot = res.get("bootstrap", {"n_boot": 0})
        if not boot["n_boot"] or metric not in boot.get("margin_vs_prototype", {}):
            continue
        marg = boot["margin_vs_prototype"][metric]
        cross = boot.get("margin_zero_crossing", {}).get(metric, {})
        n_l = res["architecture"]["predictor_depth"]
        weak = res["verdict"]["verdict"] == "MODEL TOO WEAK"
        short = (res["model"].replace("official ", "").replace("mini (", "")
                 .replace(".pth)", ""))
        lbl = short + (" [untrained]" if weak else "")
        style = dict(color=col, alpha=0.35 if weak else 1.0)

        xs = np.array([(int(m["layer"]) if m["layer"] != "out" else n_l) / n_l
                       for m in marg])
        delta = np.array([m["delta"] for m in marg])
        lo = np.array([m["ci"]["lo"] for m in marg])
        hi = np.array([m["ci"]["hi"] for m in marg])
        slo = np.array([m["ci_simultaneous"]["lo"] for m in marg])
        shi = np.array([m["ci_simultaneous"]["hi"] for m in marg])

        axes[0].fill_between(xs, slo, shi, color=col, lw=0, alpha=0.06 if weak else 0.11)
        axes[0].fill_between(xs, lo, hi, color=col, lw=0, alpha=0.08 if weak else 0.20)
        axes[0].plot(xs, delta, "-o", ms=4, label=lbl, **style)
        if cross.get("relative_depth") is not None and not weak:
            rd = cross["relative_depth"]
            rci = cross.get("ci_relative_depth")
            if rci:
                axes[0].axvspan(rci["lo"], rci["hi"], color=col, alpha=0.13, lw=0)
            axes[0].axvline(rd, color=col, ls=":", lw=1.2, alpha=0.85)
            if zoom_pick is None or n_l > zoom_pick[0]["architecture"]["predictor_depth"]:
                zoom_pick = (res, col, marg, cross, n_l)

    axes[0].axhline(0.0, color="k", ls="--", lw=0.9)
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_xlabel("relative predictor depth  (0 = mask+position, 1 = model output)")
    axes[0].set_ylabel(f"{metric}(lens) - {metric}(positional prototype)")
    axes[0].set_title("6a: complete per-layer margin over the position-only prototype\n"
                      "(dark band = pointwise 95% CI, pale = simultaneous sup-t band)")
    axes[0].legend(fontsize=7, loc="upper left")

    if zoom_pick is None:
        axes[1].set_axis_off()
    else:
        res, col, marg, cross, n_l = zoom_pick
        layers = np.array([(int(m["layer"]) if m["layer"] != "out" else n_l)
                           for m in marg], dtype=float)
        delta = np.array([m["delta"] for m in marg])
        lo = np.array([m["ci"]["lo"] for m in marg])
        hi = np.array([m["ci"]["hi"] for m in marg])
        x0 = cross["lens_point"]
        sel = (layers >= x0 - 2.2) & (layers <= x0 + 2.2)
        axes[1].errorbar(layers[sel], delta[sel],
                         yerr=[delta[sel] - lo[sel], hi[sel] - delta[sel]],
                         fmt="o-", ms=5, capsize=3, color=col, lw=1.4,
                         label="per-layer margin, 95% CI")
        axes[1].axhline(0.0, color="k", ls="--", lw=0.9)
        # Two bands, because they answer different questions: the inner one holds
        # the masks fixed, the outer one lets them vary too and is the interval a
        # claim about "the predictor" rather than "this mask set" has to live with.
        tw_cross = ((res["bootstrap"].get("cluster_image_mask") or {})
                    .get("margin_zero_crossing", {}).get(metric, {}))
        tci = tw_cross.get("ci")
        if tci:
            axes[1].axvspan(tci["lo"], tci["hi"], color=col, alpha=0.08, lw=0,
                            label="crossing 95% CI, images + masks resampled")
        rci = cross.get("ci")
        if rci:
            axes[1].axvspan(rci["lo"], rci["hi"], color=col, alpha=0.22, lw=0,
                            label="crossing 95% CI, images resampled")
        axes[1].plot([x0], [0.0], marker="D", ms=7, color="k", zorder=5)
        note = f"crossing at layer {x0:.2f}"
        if rci:
            note += f"\n95% CI [{rci['lo']:.2f}, {rci['hi']:.2f}]"
        if tci:
            note += f"\n+ masks [{tci['lo']:.2f}, {tci['hi']:.2f}]"
        axes[1].annotate(note, xy=(x0, 0.0), xytext=(8, 16),
                         textcoords="offset points", fontsize=8,
                         arrowprops=dict(arrowstyle="->", lw=0.8))
        short = res["model"].replace("official ", "")
        axes[1].set_xlabel("predictor layer")
        axes[1].set_ylabel("margin over the positional prototype")
        axes[1].set_title(f"{short}: where the margin crosses zero\n"
                          "(crossing interpolated inside each bootstrap replicate)")
        axes[1].legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(outdir, "latent_lens_margin_crossing.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[save] {path}")


def save_runs(results: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"runs": results}, fh, indent=2)
    print(f"[save] {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--model", default="both", choices=["mini", "vith", "both"])
    ap.add_argument("--mini_checkpoint", nargs="+",
                    default=["examples/ijepa/ijepa_mini.pth"],
                    help="one or more mini checkpoints to trace")
    ap.add_argument("--checkpoint", default="checkpoints/vith14_in1k_ep300.pth.tar")
    ap.add_argument("--slim_dir", default=None)
    ap.add_argument("--data_dir", default="data/mini_train")
    ap.add_argument("--n_images", type=int, default=64)
    ap.add_argument("--n_masks", type=int, default=4,
                    help="masks sampled per run, shared across all images")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="cluster bootstrap replicates (0 disables). Sets the "
                         "resolution of a p-value at 2/(B+1), which is what a "
                         "Bonferroni-corrected family of tests spends")
    ap.add_argument("--min_masks_two_way", type=int, default=8,
                    help="mask count at or above which masks are resampled as a "
                         "second cluster axis alongside images")
    ap.add_argument("--cache_dir", default="experiments/cache/latent_lens",
                    help="where a completed trace's sufficient statistics are kept "
                         "so the statistics can be redone without the model")
    ap.add_argument("--use_cache", action="store_true",
                    help="reuse a cached trace with the same key instead of running "
                         "the model")
    ap.add_argument("--reanalyze", nargs="*", default=None, metavar="NPZ",
                    help="rebuild the report and figures from cached traces (no "
                         "model, no GPU); with no paths, every cache in --cache_dir")
    ap.add_argument("--save_rows", action="store_true", help="keep per-token rows in the JSON")
    ap.add_argument("--save", default="experiments/results_latent_lens_trajectory.json")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from --save without running any model")
    ap.add_argument("--plots", default="experiments/plots")
    args = ap.parse_args()

    if args.replot:
        with open(args.save) as fh:
            make_plots(json.load(fh)["runs"], args.plots)
        return

    if args.reanalyze is not None:
        # Every statistic in this file is a function of the cached sufficient
        # statistics, so a new correction or a larger bootstrap costs seconds
        # rather than another pass over a 10 GB checkpoint.
        paths = args.reanalyze or sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
        if not paths:
            raise SystemExit(f"no cached traces in {args.cache_dir}")
        results = []
        for path in paths:
            res = summarise_cached(path, args)
            print_report(res)
            results.append(res)
        save_runs(results, args.save)
        make_plots(results, args.plots)
        return

    which: List[Tuple[str, str]] = []
    if args.model in ("mini", "both"):
        which += [("mini", c) for c in args.mini_checkpoint]
    if args.model in ("vith", "both"):
        which += [("vith", args.checkpoint)]

    results = []
    for m, ckpt in which:
        sub = argparse.Namespace(**vars(args))
        sub.model = m
        sub.mini_checkpoint = ckpt
        print("\n" + "#" * 86)
        print(f"# {m}: {os.path.basename(ckpt)}")
        print("#" * 86)
        cached = cache_path(sub, SimpleNamespace(name=m, meta={}),
                            args.n_images, args.n_masks)
        if args.use_cache and os.path.exists(cached):
            res = summarise_cached(cached, sub)
        else:
            res = run(sub)
        print_report(res)
        results.append(res)
        # Persist after every model. The ViT-H trace is long enough that
        # losing a completed run because a later one failed - a missing
        # checkpoint, an interrupted shell - is a real cost, and the file
        # is small.
        save_runs(results, args.save)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_runs(results, args.save)
    try:
        make_plots(results, args.plots)
    except Exception as exc:
        print(f"[warn] plotting failed: {exc}")


if __name__ == "__main__":
    main()
