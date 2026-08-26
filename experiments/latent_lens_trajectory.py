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
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
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


def rank_of_truth(scores: torch.Tensor, truth_idx: int) -> int:
    """1-based rank of ``truth_idx`` when ``scores`` is sorted descending."""
    order = torch.argsort(scores, descending=True)
    return int((order == truth_idx).nonzero()[0, 0].item()) + 1


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
    rows: List[Dict[str, Any]] = []
    l0_spreads = []

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for mi, (ctx, tgt) in enumerate(masks):
            n_ctx = len(ctx)
            layer0 = []
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
                t_proto = proto[tgt]                  # [n_tgt, d]
                gallery_b = targets[:, tgt, :]        # [n_img, n_tgt, d]
                nt_all = torch.nn.functional.normalize(targets[i], dim=-1)
                gb = torch.nn.functional.normalize(gallery_b, dim=-1)

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

                    for k, patch in enumerate(tgt):
                        rows.append(
                            {
                                "mask": mi,
                                "image": i,
                                "layer": lp,
                                "patch": int(patch),
                                "cos_target": float(cos_t[k]),
                                "cos_centered": float(cos_c[k]),
                                "mse": float(mse[k]),
                                "rank_a": rank_of_truth(sims_a[k], patch),
                                "rank_b": rank_of_truth(sims_b[k], i),
                            }
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

    return summarise(rows, lens_points, arch, model, args, n_img, len(masks), l0_spread)


def summarise(
    rows: List[Dict[str, Any]],
    lens_points: List[Any],
    arch: Dict[str, int],
    model: LensModel,
    args: argparse.Namespace,
    n_img: int,
    n_masks: int,
    l0_spread: float,
) -> Dict[str, Any]:
    chance_a = 1.0 / arch["num_patches"]
    chance_b = 1.0 / n_img

    by_layer: Dict[str, Any] = {}
    for lp in list(lens_points) + ["proto"]:
        sel = [r for r in rows if r["layer"] == lp]
        ra = np.array([r["rank_a"] for r in sel], dtype=np.float64)
        rb = np.array([r["rank_b"] for r in sel], dtype=np.float64)
        by_layer[str(lp)] = {
            "n": len(sel),
            "cos_target": float(np.mean([r["cos_target"] for r in sel])),
            "cos_target_std": float(np.std([r["cos_target"] for r in sel])),
            "cos_centered": float(np.mean([r["cos_centered"] for r in sel])),
            "cos_centered_std": float(np.std([r["cos_centered"] for r in sel])),
            "mse": float(np.mean([r["mse"] for r in sel])),
            "gallery_a_top1": float(np.mean(ra == 1)),
            "gallery_a_top5": float(np.mean(ra <= 5)),
            "gallery_a_mrr": float(np.mean(1.0 / ra)),
            "gallery_b_top1": float(np.mean(rb == 1)),
            "gallery_b_mrr": float(np.mean(1.0 / rb)),
        }

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
        "c2_emerges_across_layers": bool(largest_frac < 0.8) if total_rise > 1e-9 else False,
        "c3_layer0_within_2x_chance": bool(first <= 2 * chance_b),
        "layer0_gallery_b_top1": float(first),
    }
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
            "data_dir": args.data_dir,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "chance": {"gallery_a": chance_a, "gallery_b": chance_b},
        "layer0_target_token_spread_across_images": l0_spread,
        "by_layer": by_layer,
        "model_quality": quality,
        "verdict": crit,
        "rows": rows if args.save_rows else [],
    }


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

    print(f"  {'layer':<6} {'cos_target':>11} {'cos_centered':>13} {'mse':>9} "
          f"{'A@1':>7} {'A@5':>7} {'B@1':>7} {'B_mrr':>7}")
    for k, d in res["by_layer"].items():
        print(f"  {k:<6} {d['cos_target']:>11.4f} {d['cos_centered']:>13.4f} "
              f"{d['mse']:>9.4f} {d['gallery_a_top1']*100:>6.1f}% "
              f"{d['gallery_a_top5']*100:>6.1f}% {d['gallery_b_top1']*100:>6.1f}% "
              f"{d['gallery_b_mrr']:>7.3f}")

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

        axes[0].plot(xs, [d[k]["cos_target"] for k in keys], "-o", ms=4, label=lbl, **style)
        axes[0].plot(xs, [d[k]["cos_centered"] for k in keys], "--s", ms=3,
                     label=f"{short}: position removed", **style)
        axes[0].axhline(d["proto"]["cos_target"], color=col, ls=":", lw=1, alpha=0.6)
        axes[1].plot(xs, [d[k]["gallery_a_top1"] * 100 for k in keys], "-o", ms=4,
                     label=lbl, **style)
        axes[1].axhline(d["proto"]["gallery_a_top1"] * 100, color=col, ls=":", lw=1, alpha=0.6)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
        res = run(sub)
        print_report(res)
        results.append(res)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w") as fh:
        json.dump({"runs": results}, fh, indent=2)
    print(f"\n[save] {args.save}")
    try:
        make_plots(results, args.plots)
    except Exception as exc:
        print(f"[warn] plotting failed: {exc}")


if __name__ == "__main__":
    main()
