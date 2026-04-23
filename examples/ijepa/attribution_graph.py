import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from pathlib import Path
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import os


@dataclass(frozen=True)
class AttributionRun:
    """Per-image attribution summary for a target/layer pair."""

    image_idx: int
    target_id: int
    layer_idx: int
    head_scores: np.ndarray
    mean_attributions: np.ndarray
    top_context_ids: List[int]
    top_heads: List[int]
    metrics: Dict[str, float]

def get_patch_rect(patch_id, grid_size=14, img_size=224):
    """Calculates [x, y, w, h] for a patch ID."""
    row = patch_id // grid_size
    col = patch_id % grid_size
    p_size = img_size // grid_size
    return col * p_size, row * p_size, p_size, p_size

def compute_structured_layout(G, target_node, mode="importance", grid_size=14):
    """Computes research-grade positions."""
    pos = {}
    context_nodes = [n for n in G.nodes() if "target" not in n]
    
    if mode == "importance":
        pos[target_node] = np.array([0, 0])
        weights = {n: G.get_edge_data(n, target_node)['weight'] for n in context_nodes}
        sorted_context = sorted(context_nodes, key=lambda n: weights[n], reverse=True)
        num_context = len(sorted_context)
        for i, node in enumerate(sorted_context):
            angle = np.pi/2 - (2 * np.pi * i / num_context)
            w = weights[node]
            radius = 1.0 - (w * 0.4) 
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    elif mode == "spatial":
        pos[target_node] = np.array([int(target_node.split("_")[1]) % grid_size, 
                                     grid_size - (int(target_node.split("_")[1]) // grid_size)])
        for node in context_nodes:
            pid = int(node.split("_")[1])
            pos[node] = np.array([pid % grid_size, grid_size - (pid // grid_size)])
    else: # Bipartite
        pos[target_node] = np.array([1, 0.5])
        sorted_context = sorted(context_nodes, key=lambda n: G.get_edge_data(n, target_node)['weight'], reverse=True)
        for i, node in enumerate(sorted_context):
            pos[node] = np.array([0, i / (max(1, len(sorted_context)-1))])
    return pos

def draw_attribution_viz(ax, G, pos, target_node, top_n_labels=3):
    context_nodes = [n for n in G.nodes() if "target" not in n]
    edges = list(G.edges())
    if not edges: return
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    edge_ranks = np.argsort(weights)[::-1]
    
    for i in edge_ranks:
        u, v = edges[i]
        cmap = plt.get_cmap('viridis')
        color = cmap(norm_weights[i])
        is_strongest = (i == edge_ranks[0])
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2 + 10 * norm_weights[i], 
                               edge_color=color if not is_strongest else "#FFD700", 
                               arrowsize=20, ax=ax, alpha=0.8, connectionstyle="arc3,rad=0.1")
        if i in edge_ranks[:top_n_labels]:
            rank_str = f"#{i+1}: {weights[i]:.3f}"
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): rank_str}, font_size=8, ax=ax, label_pos=0.6)

    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color="#F44336", node_size=1800, ax=ax, edgecolors="white", linewidths=3)
    nx.draw_networkx_nodes(G, pos, nodelist=context_nodes, node_color="#2196F3", node_size=1200, ax=ax, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax, font_color="white")

def plot_image_overlay(ax, img, target_id, top_context_ids, grid_size=14):
    ax.imshow(img.resize((224, 224)))
    tx, ty, tw, th = get_patch_rect(target_id, grid_size)
    ax.add_patch(patches.Rectangle((tx, ty), tw, th, linewidth=4, edgecolor='#F44336', facecolor='none', label='Target'))
    for i, pid in enumerate(top_context_ids):
        cx, cy, cw, ch = get_patch_rect(pid, grid_size)
        is_strongest = (i == 0)
        color = "#FFD700" if is_strongest else "#2196F3"
        ax.add_patch(patches.Rectangle((cx, cy), cw, ch, linewidth=4 if is_strongest else 2, edgecolor=color, facecolor='none', alpha=1.0 if is_strongest else 0.7))
        ax.text(cx, cy, f"#{i+1}", color="white", fontsize=8, weight='bold', bbox=dict(facecolor=color, alpha=0.5, pad=0))
    ax.axis('off')


def plot_image_gallery(ax, imgs, runs: Sequence[AttributionRun], grid_size=14):
    """Show every same-class image used for the consistency check."""
    if not imgs or not runs:
        ax.axis("off")
        return

    thumb_size = 112
    n_images = len(imgs)
    n_cols = min(3, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    canvas = np.ones((n_rows * thumb_size, n_cols * thumb_size, 3), dtype=np.uint8) * 255

    ax.imshow(canvas)
    for idx, (img, run) in enumerate(zip(imgs, runs)):
        row = idx // n_cols
        col = idx % n_cols
        x0 = col * thumb_size
        y0 = row * thumb_size
        thumb = np.asarray(img.resize((thumb_size, thumb_size)).convert("RGB"))
        ax.imshow(thumb, extent=(x0, x0 + thumb_size, y0 + thumb_size, y0))

        scale = thumb_size / 224.0
        tx, ty, tw, th = get_patch_rect(run.target_id, grid_size)
        ax.add_patch(
            patches.Rectangle(
                (x0 + tx * scale, y0 + ty * scale),
                tw * scale,
                th * scale,
                linewidth=2,
                edgecolor="#F44336",
                facecolor="none",
            )
        )

        for rank, pid in enumerate(run.top_context_ids):
            cx, cy, cw, ch = get_patch_rect(pid, grid_size)
            color = "#FFD700" if rank == 0 else "#2196F3"
            ax.add_patch(
                patches.Rectangle(
                    (x0 + cx * scale, y0 + cy * scale),
                    cw * scale,
                    ch * scale,
                    linewidth=2 if rank == 0 else 1.2,
                    edgecolor=color,
                    facecolor="none",
                    alpha=1.0 if rank == 0 else 0.75,
                )
            )
            ax.text(
                x0 + cx * scale,
                y0 + cy * scale,
                f"#{rank+1}",
                color="white",
                fontsize=7,
                weight="bold",
                bbox=dict(facecolor=color, alpha=0.6, pad=0),
            )

        ax.text(
            x0 + 2,
            y0 + 12,
            f"img {idx+1} | heads {run.top_heads}",
            color="white",
            fontsize=7,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.55, pad=1),
        )

    ax.set_xlim(0, n_cols * thumb_size)
    ax.set_ylim(n_rows * thumb_size, 0)
    ax.axis("off")

def load_same_class_images(
    raw_img: Image.Image,
    n_images: int = 5,
    image_dir: str | None = None,
) -> List[Image.Image]:
    """Load same-class images from disk, with deterministic fallback variants."""
    search_dir = Path(image_dir) if image_dir else Path(__file__).resolve().parent
    local_paths = []
    if search_dir.exists():
        patterns = ("dog*.jpg", "dog*.jpeg", "dog*.png", "Dog*.jpg", "Dog*.jpeg", "Dog*.png")
        for pattern in patterns:
            local_paths.extend(search_dir.glob(pattern))
        local_paths = sorted({path.resolve() for path in local_paths})[:n_images]

    if len(local_paths) >= n_images:
        loaded = []
        for path in local_paths:
            with Image.open(path) as img:
                loaded.append(img.convert("RGB"))
        print(f"Loaded {len(loaded)} same-class images from {search_dir}")
        return loaded

    # Fallback: create deterministic same-image variants when we do not have
    # enough local same-class images available on disk.
    if n_images <= 1:
        return [raw_img]

    print(
        f"Found {len(local_paths)} local dog image(s) in {search_dir}; "
        "using deterministic same-image variants as fallback."
    )
    width, height = raw_img.size
    variants = []
    crop_specs = [
        (0.00, 0.00, 1.00),
        (0.04, 0.00, 0.96),
        (0.00, 0.04, 0.96),
        (0.04, 0.04, 0.96),
        (0.02, 0.02, 0.92),
    ]
    for i in range(n_images):
        x_frac, y_frac, scale = crop_specs[i % len(crop_specs)]
        crop_w = int(width * scale)
        crop_h = int(height * scale)
        left = min(int(width * x_frac), width - crop_w)
        top = min(int(height * y_frac), height - crop_h)
        cropped = raw_img.crop((left, top, left + crop_w, top + crop_h)).resize(raw_img.size)

        # Small deterministic channel-neutral brightness shift.
        arr = np.asarray(cropped).astype(np.float32)
        arr = np.clip(arr * (0.94 + 0.03 * (i % 5)), 0, 255).astype(np.uint8)
        variants.append(Image.fromarray(arr))
    return variants


def attribution_metrics(attributions: np.ndarray, k: int) -> Dict[str, float]:
    """Compute compact scalar metrics for mean +/- std reporting."""
    safe = np.clip(attributions.astype(np.float64), 0.0, None)
    total = float(safe.sum() + 1e-12)
    probs = safe / total
    sorted_vals = np.sort(safe)[::-1]
    top_k = sorted_vals[: min(k, len(sorted_vals))]
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs)))
    return {
        "top_k_mass": float(top_k.sum() / total),
        "max_weight": float(sorted_vals[0]) if len(sorted_vals) else 0.0,
        "entropy": entropy,
        "effective_patches": float(np.exp(-(probs * np.log(probs + 1e-12)).sum())),
    }


def extract_predictor_attributions(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    img_tensor: torch.Tensor,
    context_ids: Sequence[int],
    target_id: int,
    layer_idx: int,
    k: int,
    top_heads: int = 2,
) -> Tuple[np.ndarray, List[int], List[int], np.ndarray, Dict[str, float]]:
    """Return patch attributions and per-head importance for one target."""
    adapter.last_context_ids = list(context_ids)
    adapter.last_target_ids = [target_id]

    with torch.no_grad():
        wm.run_with_cache(img_tensor)
        attn = adapter.predictor.blocks[layer_idx].attn.last_attn_weights
        if attn is None:
            raise RuntimeError("Predictor attention weights were not captured.")
        target_to_context_by_head = attn[0, :, -1, : len(context_ids)].cpu().numpy()

    mean_attributions = target_to_context_by_head.mean(axis=0)
    sorted_patch_indices = np.argsort(mean_attributions)[::-1][:k]
    top_context_ids = [int(context_ids[i]) for i in sorted_patch_indices]

    head_scores = target_to_context_by_head[:, sorted_patch_indices].sum(axis=1)
    top_head_ids = np.argsort(head_scores)[::-1][:top_heads].tolist()
    metrics = attribution_metrics(mean_attributions, k)
    metrics["head_concentration"] = float(head_scores.max() / (head_scores.sum() + 1e-12))
    return mean_attributions, top_context_ids, top_head_ids, head_scores, metrics


def summarize_attribution_runs(runs: Sequence[AttributionRun], k: int) -> Dict[str, Dict[str, float]]:
    metric_names = sorted({name for run in runs for name in run.metrics})
    summary = {}
    for name in metric_names:
        vals = np.array([run.metrics[name] for run in runs], dtype=np.float64)
        summary[name] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}

    if len(runs) > 1:
        top_sets = [set(run.top_context_ids[:k]) for run in runs]
        overlaps = []
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                union = top_sets[i] | top_sets[j]
                overlaps.append(len(top_sets[i] & top_sets[j]) / len(union) if union else 0.0)
        summary["top_patch_jaccard"] = {
            "mean": float(np.mean(overlaps)),
            "std": float(np.std(overlaps, ddof=1)) if len(overlaps) > 1 else 0.0,
        }

    head_counts: Dict[int, int] = {}
    for run in runs:
        for head in run.top_heads:
            head_counts[head] = head_counts.get(head, 0) + 1
    total_slots = max(1, sum(head_counts.values()))
    summary["dominant_head_frequency"] = {
        "mean": float(max(head_counts.values(), default=0) / total_slots),
        "std": 0.0,
    }
    return summary


def print_consistency_report(
    grouped_runs: Dict[Tuple[int, int], List[AttributionRun]], k: int, n_images: int
) -> None:
    print(f"\nCross-image attribution consistency ({n_images} same-class views)")
    print("Target | Layer | top-patch Jaccard | dominant heads | metrics mean +/- std")
    for (target_id, layer_idx), runs in sorted(grouped_runs.items()):
        summary = summarize_attribution_runs(runs, k)
        head_counts: Dict[int, int] = {}
        for run in runs:
            for head in run.top_heads:
                head_counts[head] = head_counts.get(head, 0) + 1
        heads = ", ".join(f"h{h}:{count}/{len(runs)}" for h, count in sorted(head_counts.items(), key=lambda x: -x[1])[:3])
        metrics = ", ".join(
            f"{name}={vals['mean']:.3f}+/-{vals['std']:.3f}"
            for name, vals in summary.items()
            if name not in {"top_patch_jaccard", "dominant_head_frequency"}
        )
        patch_consistency = summary.get("top_patch_jaccard", {"mean": 1.0, "std": 0.0})
        print(
            f"{target_id:>6} | {layer_idx:>5} | "
            f"{patch_consistency['mean']:.3f}+/-{patch_consistency['std']:.3f} | {heads} | {metrics}"
        )


def visualize_research_ijepa(
    target_ids=None,
    k=6,
    layout_mode="importance",
    n_consistency_images=5,
    same_class_image_dir: str | None = None,
):
    script_dir = Path(__file__).resolve().parent
    raw_img = get_sample_image()
    
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384)
    adapter = IJEPAAdapter(config)
    
    checkpoint_candidates = [
        script_dir / "ijepa_mini.pth",
        Path.cwd() / "ijepa_mini.pth",
    ]
    checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
    if checkpoint_path is not None:
        print(f"Loading weights from {checkpoint_path}")
        adapter.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True),
            strict=False,
        )
        
    wm = HookedWorldModel(adapter, config)
    wm.adapter.eval()
    
    if target_ids is None:
        target_ids = [42, 114]
    
    # We'll compare middle and final layers of the predictor
    predictor_depth = len(adapter.predictor.blocks)
    layers_to_compare = [predictor_depth // 2, predictor_depth - 1]
    layer_names = ["Middle", "Final"]
    
    num_targets = len(target_ids)
    num_layers = len(layers_to_compare)
    fig = plt.figure(figsize=(20, 6 * num_targets * num_layers))
    
    same_class_images = load_same_class_images(
        raw_img,
        n_images=n_consistency_images,
        image_dir=same_class_image_dir or str(script_dir),
    )
    same_class_tensors = [preprocess_image(img) for img in same_class_images]
    grouped_runs: Dict[Tuple[int, int], List[AttributionRun]] = {}

    for t_idx, target_id in enumerate(target_ids):
        # Keep masks fixed so consistency reflects model behavior, not mask resampling.
        context_ids, _ = get_ijepa_masks(num_context=80)
        if target_id in context_ids: context_ids.remove(target_id)

        for l_idx, (layer_idx, name) in enumerate(zip(layers_to_compare, layer_names)):
            runs = []
            for image_idx, tensor in enumerate(same_class_tensors):
                mean_attr, top_context_ids_i, top_heads, head_scores, metrics = extract_predictor_attributions(
                    wm, adapter, tensor, context_ids, target_id, layer_idx, k
                )
                runs.append(
                    AttributionRun(
                        image_idx=image_idx,
                        target_id=target_id,
                        layer_idx=layer_idx,
                        head_scores=head_scores,
                        mean_attributions=mean_attr,
                        top_context_ids=top_context_ids_i,
                        top_heads=top_heads,
                        metrics=metrics,
                    )
                )
            grouped_runs[(target_id, layer_idx)] = runs

            display_run = runs[0]
            target_to_context = display_run.mean_attributions
            top_context_ids = display_run.top_context_ids
            top_weights = [target_to_context[context_ids.index(pid)] for pid in top_context_ids]

            # Subplots: Target Row, Column 1=Graph, Column 2=Image
            row_idx = t_idx * num_layers + l_idx
            gs = fig.add_gridspec(num_targets * num_layers, 2, width_ratios=[1.2, 1])
            ax_graph = fig.add_subplot(gs[row_idx, 0])
            ax_img = fig.add_subplot(gs[row_idx, 1])
            
            # Build Graph
            G = nx.DiGraph()
            target_node = f"target_{target_id}"
            for pid, w in zip(top_context_ids, top_weights):
                G.add_edge(f"patch_{pid}", target_node, weight=float(w))

            pos = compute_structured_layout(G, target_node, mode=layout_mode)
            draw_attribution_viz(ax_graph, G, pos, target_node)
            summary = summarize_attribution_runs(runs, k)
            jaccard = summary.get("top_patch_jaccard", {"mean": 1.0, "std": 0.0})
            ax_graph.set_title(
                f"I-JEPA Flow | Layer: {name} ({layer_idx})\n"
                f"Target: {target_id} | Top-{k} consistency: {jaccard['mean']:.2f}+/-{jaccard['std']:.2f}",
                fontsize=12,
                loc='left',
            )
            ax_graph.axis('off')
            
            plot_image_gallery(ax_img, same_class_images, runs)
            ax_img.set_title(f"Grounding Across {len(runs)} Dog Images | {name} Layer")

    plt.tight_layout()
    print_consistency_report(grouped_runs, k, len(same_class_images))
    if os.environ.get("SAVE_PLOT"):
        plt.savefig("attribution_comparison.png")
        print("Comparison plot saved to attribution_comparison.png")
    else:
        plt.show()

if __name__ == "__main__":
    visualize_research_ijepa(target_ids=[42, 114], k=6, layout_mode="importance")
