"""orchestrator.py — Unified I-JEPA Interpretability Orchestrator

    results = run_full_suite(image, checkpoint_path, seed=42)
    # -> JSON dict + orchestrator_results/orchestrator_summary.png

Architecture:
  SharedLatentState  — unified research substrate built from one forward pass per target
  ExecutionEngine    — owns model refs; builds state and drives the experiment graph
  EXPERIMENT_DAG     — explicit dependency graph; experiments run in topological order
  EXPERIMENT_REGISTRY— pure functions (state, engine, img, upstream, seed, n_steps) -> dict

Experiments:
  1. attribution  — predictor attention weights, middle + final layer
  2. circuit      — energy-threshold circuit patches
  3. faithfulness — Top-K ablation/reconstruction AOPC proxy   [depends: attribution]
  4. structural   — per-layer mean-ablation importance         [depends: forward]
  5. formal       — greedy minimal circuit at 85% threshold    [depends: attribution]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image


# ---------------------------------------------------------------------------
# Shared Latent State
# ---------------------------------------------------------------------------

@dataclass
class SharedLatentState:
    """Unified research substrate — populated once, consumed by all experiments.

    Fields added here (vs old ForwardCache) intentionally support Phase 2-4:
      ctx_latents           -> IG / CKA analysis
      predictor_activations -> mean ablation, rollout, layer-wise probing
      seed / mask record    -> reproducibility audit
    """
    context_ids: List[int]
    target_ids: List[int]
    seed: int
    # Per-target tensors
    target_gt: Dict[int, torch.Tensor] = field(default_factory=dict)            # [1, C]
    ctx_latents: Dict[int, torch.Tensor] = field(default_factory=dict)          # [1, N_ctx, C]
    predictor_attn: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)   # layer -> [N_ctx]
    predictor_activations: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)  # layer -> [1, seq, C]
    clean_mse: Dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experiment DAG + topo-sort
# ---------------------------------------------------------------------------

EXPERIMENT_DAG: Dict[str, List[str]] = {
    "forward":      [],
    "attribution":  ["forward"],
    "circuit":      ["forward"],
    "faithfulness": ["forward", "attribution"],   # consumes attribution's ranked patch list
    "structural":   ["forward"],
    "formal":       ["forward", "attribution"],   # consumes attribution's ranked patch list
}


def _topo_sort(dag: Dict[str, List[str]]) -> List[str]:
    visited: set = set()
    order: List[str] = []

    def visit(n: str) -> None:
        if n in visited:
            return
        visited.add(n)
        for dep in dag[n]:
            visit(dep)
        order.append(n)

    for n in dag:
        visit(n)
    return order


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """Owns model references; builds SharedLatentState and drives the experiment DAG."""

    def __init__(self, adapter: IJEPAAdapter, wm: HookedWorldModel) -> None:
        self.adapter = adapter
        self.wm = wm

    def build_state(
        self,
        img_tensor: torch.Tensor,
        context_ids: List[int],
        target_ids: List[int],
        seed: int,
    ) -> SharedLatentState:
        """One forward pass per target capturing all shared artifacts.

        Uses PyTorch forward hooks so nothing is computed twice:
          context_encoder hook  -> ctx_latents
          predictor block hooks -> predictor_activations
          predictor output hook -> prediction for clean_mse
          wm.run_with_cache     -> target_gt + last_attn_weights at every block
        """
        state = SharedLatentState(
            context_ids=list(context_ids),
            target_ids=list(target_ids),
            seed=seed,
        )
        adapter, wm = self.adapter, self.wm

        for target_id in target_ids:
            ctx = [c for c in context_ids if c != target_id]
            adapter.last_context_ids = ctx
            adapter.last_target_ids = [target_id]

            _layer_acts: Dict[int, torch.Tensor] = {}
            _pred_out: List[Optional[torch.Tensor]] = [None]
            _enc_out: List[Optional[torch.Tensor]] = [None]
            handles = []

            for i, block in enumerate(adapter.predictor.blocks):
                def _act_hook(m, inp, out, idx=i):
                    _layer_acts[idx] = out.detach()
                handles.append(block.register_forward_hook(_act_hook))

            handles.append(adapter.predictor.register_forward_hook(
                lambda m, inp, out: _pred_out.__setitem__(0, out.detach())
            ))
            handles.append(adapter.context_encoder.register_forward_hook(
                lambda m, inp, out: _enc_out.__setitem__(0, out.detach())
            ))

            with torch.no_grad():
                _, wm_cache = wm.run_with_cache(img_tensor)

                # Predictor attention at every layer (set by wm.run_with_cache)
                state.predictor_attn[target_id] = {}
                for i, block in enumerate(adapter.predictor.blocks):
                    raw = block.attn.last_attn_weights  # [B, heads, seq, seq]
                    w = raw[0].mean(0)[-1, : len(ctx)].cpu().numpy()
                    state.predictor_attn[target_id][i] = w

                # Per-layer predictor activations
                state.predictor_activations[target_id] = dict(_layer_acts)

                # Context latents: encoder output is already [1, len(ctx), C]
                if _enc_out[0] is not None:
                    state.ctx_latents[target_id] = _enc_out[0]

                # Target ground truth
                reps = wm_cache.get("target_encoding", 0)
                if reps is not None:
                    gt = reps[0, [target_id], :] if reps.dim() == 3 else reps[[target_id], :]
                else:
                    full = adapter.target_encode(img_tensor)
                    gt = full[:, [target_id], :].squeeze(0) if full.dim() == 3 else full[[target_id], :]
                state.target_gt[target_id] = gt

                # Clean MSE from captured predictor output — no second forward pass
                if _pred_out[0] is not None:
                    pred = _pred_out[0].squeeze(0)
                    state.clean_mse[target_id] = F.mse_loss(pred.flatten(), gt.flatten()).item()

            for h in handles:
                h.remove()

        return state

    def predict_mse(
        self,
        img_tensor: torch.Tensor,
        active_ids: List[int],
        target_id: int,
        target_gt: torch.Tensor,
    ) -> float:
        """MSE for a specific context subset — needed by ablation/reconstruction loops."""
        if not active_ids:
            return float("nan")
        self.adapter.last_context_ids = active_ids
        self.adapter.last_target_ids = [target_id]
        with torch.no_grad():
            ctx_lat, _ = self.adapter.encode(img_tensor)
            pred = self.adapter.dynamics(ctx_lat).squeeze(0)
        return F.mse_loss(pred.flatten(), target_gt.flatten()).item()

    def run_graph(
        self,
        state: SharedLatentState,
        img_tensor: torch.Tensor,
        seed: int,
        n_steps: int = 15,
    ) -> Dict[str, Any]:
        """Run experiments in topological order; pass upstream results downstream."""
        results: Dict[str, Any] = {}
        for exp in _topo_sort(EXPERIMENT_DAG):
            if exp == "forward":
                continue
            fn = EXPERIMENT_REGISTRY[exp]
            results[exp] = fn(state, self, img_tensor, results, seed, n_steps)
            print(f"[Orchestrator]   + {exp}")
        return results


# ---------------------------------------------------------------------------
# Experiment functions  (pure: state + engine + upstream -> dict)
# ---------------------------------------------------------------------------

def _exp_attribution(
    state: SharedLatentState,
    engine: ExecutionEngine,
    img_tensor: torch.Tensor,
    upstream: Dict[str, Any],
    seed: int,
    n_steps: int,
) -> Dict[str, Any]:
    predictor_depth = len(engine.adapter.predictor.blocks)
    layers = {"middle": predictor_depth // 2, "final": predictor_depth - 1}
    k = 6

    per_target: Dict[str, Any] = {}
    for target_id in state.target_ids:
        ctx = [c for c in state.context_ids if c != target_id]
        per_layer: Dict[str, Any] = {}
        for name, layer_idx in layers.items():
            w = state.predictor_attn[target_id][layer_idx]
            ranked = [ctx[i] for i in np.argsort(w)[::-1]]
            top_weights = [float(w[np.argsort(w)[::-1][i]]) for i in range(min(k, len(ranked)))]
            per_layer[name] = {
                "top_context_patches": ranked[:k],
                "top_weights": top_weights,
                "ranked_patches": ranked,          # consumed downstream by faithfulness + formal
                "mean_attn": float(w.mean()),
                "entropy": float(-(w * np.log(w + 1e-9)).sum()),
            }
        per_target[str(target_id)] = per_layer

    return {"targets": per_target, "k": k, "layers_analyzed": list(layers.keys())}


def _exp_circuit(
    state: SharedLatentState,
    engine: ExecutionEngine,
    img_tensor: torch.Tensor,
    upstream: Dict[str, Any],
    seed: int,
    n_steps: int,
) -> Dict[str, Any]:
    per_target: Dict[str, Any] = {}
    for target_id in state.target_ids:
        ctx = [c for c in state.context_ids if c != target_id]
        last_layer = max(state.predictor_attn[target_id])
        w = state.predictor_attn[target_id][last_layer]
        total_energy = w.sum()
        circuit_idx = np.where(w > total_energy * 0.005)[0]
        pairs = sorted([(ctx[i], float(w[i])) for i in circuit_idx], key=lambda x: -x[1])
        per_target[str(target_id)] = {
            "target_patch": target_id,
            "circuit_size": len(pairs),
            "circuit_patches": [p for p, _ in pairs],
            "circuit_weights": [wt for _, wt in pairs],
            "total_context_patches": len(ctx),
            "energy_coverage": float(sum(wt for _, wt in pairs) / (total_energy + 1e-9)),
        }
    return {"per_target": per_target}


def _exp_faithfulness(
    state: SharedLatentState,
    engine: ExecutionEngine,
    img_tensor: torch.Tensor,
    upstream: Dict[str, Any],
    seed: int,
    n_steps: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    attribution = upstream.get("attribution", {})

    per_target: Dict[str, Any] = {}
    for target_id in state.target_ids:
        ctx = [c for c in state.context_ids if c != target_id]
        gt = state.target_gt[target_id]

        # Consume pre-ranked list from attribution (explicit DAG dependency)
        tid_str = str(target_id)
        attr_targets = attribution.get("targets", {})
        if tid_str in attr_targets:
            top_ctx = attr_targets[tid_str]["final"]["ranked_patches"]
        else:
            last_layer = max(state.predictor_attn[target_id])
            w = state.predictor_attn[target_id][last_layer]
            top_ctx = [ctx[i] for i in np.argsort(w)[::-1]]

        rand_ctx = rng.permutation(ctx).tolist()
        step = max(1, len(ctx) // n_steps)
        k_values = list(range(1, len(ctx), step))[:n_steps]

        ablation: Dict[str, Any] = {"top": [], "random": [], "k_values": k_values}
        reconstruction: Dict[str, Any] = {"top": [], "random": [], "k_values": k_values}

        for k in k_values:
            ablation["top"].append(engine.predict_mse(img_tensor, [c for c in ctx if c not in top_ctx[:k]], target_id, gt))
            ablation["random"].append(engine.predict_mse(img_tensor, [c for c in ctx if c not in rand_ctx[:k]], target_id, gt))
            reconstruction["top"].append(engine.predict_mse(img_tensor, top_ctx[:k], target_id, gt))
            reconstruction["random"].append(engine.predict_mse(img_tensor, rand_ctx[:k], target_id, gt))

        def _auc(vals: List[float]) -> float:
            clean = [v for v in vals if v == v]
            return float(np.trapz(clean) / max(1, len(clean))) if clean else 0.0

        top_auc = _auc(ablation["top"])
        rand_auc = _auc(ablation["random"])
        per_target[tid_str] = {
            "target_patch": target_id,
            "ablation": ablation,
            "reconstruction": reconstruction,
            "aopc": top_auc - rand_auc,
            "top_ablation_auc": top_auc,
            "random_ablation_auc": rand_auc,
        }

    return {
        "per_target": per_target,
        "mean_aopc": float(np.mean([v["aopc"] for v in per_target.values()])),
    }


def _exp_structural(
    state: SharedLatentState,
    engine: ExecutionEngine,
    img_tensor: torch.Tensor,
    upstream: Dict[str, Any],
    seed: int,
    n_steps: int,
) -> Dict[str, Any]:
    adapter = engine.adapter

    per_target: Dict[str, Any] = {}
    for target_id in state.target_ids:
        ctx = [c for c in state.context_ids if c != target_id]
        gt = state.target_gt[target_id]
        clean_mse = state.clean_mse.get(target_id, float("nan"))

        layer_scores = []
        for layer_idx, block in enumerate(adapter.predictor.blocks):
            # Mean ablation: replace block output with its mean activation from shared state.
            # More semantically valid than zero-ablation: isolates the block's deviation
            # from its average contribution without collapsing the residual stream.
            saved = state.predictor_activations[target_id].get(layer_idx)
            if saved is not None:
                mean_act = saved.mean(dim=1, keepdim=True)  # [1, 1, C]
                handle = block.register_forward_hook(
                    lambda m, inp, out, mean=mean_act: mean.expand_as(out)
                )
            else:
                handle = block.register_forward_hook(lambda m, inp, out: torch.zeros_like(out))

            try:
                ablated_mse = engine.predict_mse(img_tensor, ctx, target_id, gt)
            finally:
                handle.remove()

            delta = ablated_mse - clean_mse
            layer_scores.append({
                "layer": layer_idx,
                "clean_mse": clean_mse,
                "ablated_mse": float(ablated_mse),
                "mse_delta": float(delta),
                "importance": float(delta) / (clean_mse + 1e-9),
            })

        best_layer = max(layer_scores, key=lambda x: x["mse_delta"])["layer"]
        per_target[str(target_id)] = {
            "target_patch": target_id,
            "clean_mse": clean_mse,
            "layer_importance": layer_scores,
            "most_important_layer": best_layer,
        }

    return {"per_target": per_target}


def _exp_formal(
    state: SharedLatentState,
    engine: ExecutionEngine,
    img_tensor: torch.Tensor,
    upstream: Dict[str, Any],
    seed: int,
    n_steps: int,
) -> Dict[str, Any]:
    attribution = upstream.get("attribution", {})
    threshold = 0.85

    per_target: Dict[str, Any] = {}
    for target_id in state.target_ids:
        ctx = [c for c in state.context_ids if c != target_id]
        gt = state.target_gt[target_id]

        # Consume pre-ranked list from attribution (explicit DAG dependency)
        tid_str = str(target_id)
        attr_targets = attribution.get("targets", {})
        if tid_str in attr_targets:
            ranked_ctx = attr_targets[tid_str]["final"]["ranked_patches"]
        else:
            last_layer = max(state.predictor_attn[target_id])
            atts = state.predictor_attn[target_id][last_layer]
            ranked_ctx = [ctx[i] for i in np.argsort(atts)[::-1]]

        baseline_mse = torch.mean(gt ** 2).item()

        def _perf(active: List[int]) -> float:
            if not active:
                return 0.0
            return 1.0 - engine.predict_mse(img_tensor, active, target_id, gt) / (baseline_mse + 1e-6)

        circuit: List[int] = []
        perf_curve: List[float] = []
        for patch in ranked_ctx:
            circuit.append(patch)
            perf = _perf(circuit)
            perf_curve.append(float(perf))
            if perf >= threshold:
                break

        per_target[tid_str] = {
            "target_patch": target_id,
            "threshold": threshold,
            "circuit_patches": circuit,
            "circuit_size": len(circuit),
            "total_context": len(ctx),
            "sparsity": 1.0 - len(circuit) / max(1, len(ctx)),
            "achieved_performance": perf_curve[-1] if perf_curve else 0.0,
            "full_context_performance": float(_perf(ranked_ctx)),
            "performance_curve": perf_curve,
        }

    return {
        "per_target": per_target,
        "mean_circuit_size": float(np.mean([v["circuit_size"] for v in per_target.values()])),
        "mean_sparsity": float(np.mean([v["sparsity"] for v in per_target.values()])),
    }


EXPERIMENT_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "attribution":  _exp_attribution,
    "circuit":      _exp_circuit,
    "faithfulness": _exp_faithfulness,
    "structural":   _exp_structural,
    "formal":       _exp_formal,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seeded_masks(
    seed: int,
    num_context: int = 80,
    num_target: int = 3,
    num_patches: int = 196,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_patches)
    context_ids = sorted(indices[:num_context].tolist())
    target_ids = sorted(indices[num_context : num_context + num_target].tolist())
    return context_ids, target_ids


def _load_model(
    checkpoint_path: Optional[str],
    config: WorldModelConfig,
) -> tuple[HookedWorldModel, IJEPAAdapter]:
    adapter = IJEPAAdapter(config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        adapter.load_state_dict(sd, strict=False)
        print(f"[Orchestrator] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[Orchestrator] No checkpoint — using random weights")
    adapter.eval()
    wm = HookedWorldModel(adapter, config)
    return wm, adapter


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def _plot_summary(results: Dict[str, Any], output_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("I-JEPA Interpretability Orchestrator — Summary", fontsize=14, fontweight="bold")

    # Panel 1: attribution weights (first target, final layer)
    ax = axes[0, 0]
    try:
        tid = next(iter(results["attribution"]["targets"]))
        final = results["attribution"]["targets"][tid].get("final", {})
        weights = final.get("top_weights", [])
        patch_ids = final.get("top_context_patches", [])
        ax.bar(range(len(weights)), weights, color="#2196F3")
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels([f"p{p}" for p in patch_ids], rotation=45, fontsize=8)
        ax.set_title(f"Attribution — Target {tid} (Final Layer)")
    except Exception:
        ax.set_title("Attribution — no data")
    ax.set_ylabel("Attention Weight")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: circuit size per target
    ax = axes[0, 1]
    try:
        tids = list(results["circuit"]["per_target"].keys())
        sizes = [results["circuit"]["per_target"][t]["circuit_size"] for t in tids]
        ax.bar(tids, sizes, color="#4CAF50")
        ax.set_title("Circuit Size per Target Patch")
        ax.set_ylabel("# Circuit Patches")
    except Exception:
        ax.set_title("Circuit Discovery — no data")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: faithfulness ablation curves (first target)
    ax = axes[1, 0]
    try:
        first_key = next(iter(results["faithfulness"]["per_target"]))
        f = results["faithfulness"]["per_target"][first_key]
        k_vals = f["ablation"]["k_values"]
        ax.plot(k_vals[: len(f["ablation"]["top"])], f["ablation"]["top"], "r-o", markersize=4, label="Top-K removed")
        ax.plot(k_vals[: len(f["ablation"]["random"])], f["ablation"]["random"], color="gray", linestyle="--", label="Random removed")
        ax.legend(fontsize=8)
        ax.set_title(f"Faithfulness Ablation — Target {first_key}")
    except Exception:
        ax.set_title("Faithfulness — no data")
    ax.set_xlabel("Patches Removed")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)

    # Panel 4: predictor layer importance (first target)
    ax = axes[1, 1]
    try:
        first_key = next(iter(results["structural"]["per_target"]))
        s = results["structural"]["per_target"][first_key]
        layers = [d["layer"] for d in s["layer_importance"]]
        deltas = [d["mse_delta"] for d in s["layer_importance"]]
        ax.bar(layers, deltas, color="#FF5722")
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_title(f"Layer Importance (mean-ablation) — Target {first_key}")
    except Exception:
        ax.set_title("Structural — no data")
    ax.set_xlabel("Predictor Layer")
    ax.set_ylabel("MSE Delta")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "orchestrator_summary.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_full_suite(
    image: Union[str, Image.Image, None] = None,
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
    output_dir: str = "orchestrator_results",
    target_ids: Optional[List[int]] = None,
    num_context: int = 80,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all I-JEPA interpretability experiments via a shared compute graph.

    Args:
        image: PIL image, path to image file, or None (downloads sample image).
        checkpoint_path: Path to .pth checkpoint. None uses random weights.
        seed: Seed for reproducible masking and random baselines.
        output_dir: Directory for orchestrator_results.json + summary figure.
        target_ids: Explicit target patch IDs. None samples from seed.
        num_context: Number of context patches.
        config_overrides: WorldModelConfig field overrides, e.g. {"n_layers": 12}.

    Returns:
        JSON-serializable dict: metadata, attribution, circuit, faithfulness,
        structural, formal_circuit, figures.
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if image is None:
        raw_img = get_sample_image()
    elif isinstance(image, str):
        raw_img = Image.open(image).convert("RGB")
    else:
        raw_img = image
    img_tensor = preprocess_image(raw_img)

    cfg_kwargs: Dict[str, Any] = dict(
        backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384
    )
    if config_overrides:
        cfg_kwargs.update(config_overrides)
    config = WorldModelConfig(**cfg_kwargs)

    wm, adapter = _load_model(checkpoint_path, config)

    context_ids, auto_targets = _seeded_masks(seed, num_context=num_context)
    if target_ids is None:
        target_ids = auto_targets[:2]

    print(
        f"[Orchestrator] seed={seed} | targets={target_ids} | "
        f"context={len(context_ids)} patches | output={output_dir}/"
    )

    engine = ExecutionEngine(adapter, wm)

    print("[Orchestrator] Building shared latent state...")
    state = engine.build_state(img_tensor, context_ids, target_ids, seed)

    print(f"[Orchestrator] Running experiment DAG ({len(EXPERIMENT_DAG) - 1} nodes)...")
    exp = engine.run_graph(state, img_tensor, seed)

    results: Dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "checkpoint": checkpoint_path,
            "target_ids": target_ids,
            "context_size": len(context_ids),
            "config": cfg_kwargs,
            "dag": EXPERIMENT_DAG,
        },
        "attribution":    exp.get("attribution", {}),
        "circuit":        exp.get("circuit", {}),
        "faithfulness":   exp.get("faithfulness", {}),
        "structural":     exp.get("structural", {}),
        "formal_circuit": exp.get("formal", {}),
    }

    figures: Dict[str, str] = {}
    try:
        figures["summary"] = _plot_summary(results, output_dir)
    except Exception as exc:
        print(f"[Orchestrator] Warning: figure failed — {exc}")

    results["figures"] = figures
    results["metadata"]["elapsed_seconds"] = round(time.time() - t0, 2)

    json_path = os.path.join(output_dir, "orchestrator_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"[Orchestrator] Done in {results['metadata']['elapsed_seconds']}s → {json_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full I-JEPA interpretability orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--checkpoint", default="ijepa_mini.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="orchestrator_results")
    parser.add_argument("--targets", type=int, nargs="+", default=None)
    parser.add_argument("--num-context", type=int, default=80)
    args = parser.parse_args()

    res = run_full_suite(
        image=args.image,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        output_dir=args.output_dir,
        target_ids=args.targets,
        num_context=args.num_context,
    )

    print("\n=== Orchestrator Summary ===")
    print(f"  Targets       : {res['metadata']['target_ids']}")
    print(f"  AOPC (mean)   : {res['faithfulness']['mean_aopc']:.4f}")
    print(f"  Circuit size  : {res['formal_circuit']['mean_circuit_size']:.1f} / {res['metadata']['context_size']}")
    print(f"  Sparsity      : {res['formal_circuit']['mean_sparsity']:.1%}")
    print(f"  Output dir    : {args.output_dir}/")
    sys.exit(0)
