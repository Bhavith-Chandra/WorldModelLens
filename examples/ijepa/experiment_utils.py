from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from world_model_lens import HookedWorldModel, LatentProber, WorldModelConfig
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter

from utils import (
    DEFAULT_CHECKPOINT,
    OUTPUT_DIR,
    PatchComponent,
    build_ijepa_config,
    build_patch_heatmap,
    build_patch_labels,
    collect_ijepa_artifacts,
    get_sample_image,
    preprocess_image,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

CURRENT_OUTPUT_DIR = OUTPUT_DIR


def set_output_dir(path: str | Path) -> None:
    global CURRENT_OUTPUT_DIR
    CURRENT_OUTPUT_DIR = Path(path)
    CURRENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ProbeMetric:
    component: str
    concept: str
    probe_family: str
    score: float
    score_name: str


@dataclass
class AttentionHeadProbeResult:
    head_name: str
    concept: str
    score: float
    score_name: str


_COMPONENT_ORDER = [
    "z_posterior",
    "target_encoding",
    "predictor_targets",
    "predictor_residual",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_condition_world_model(
    condition: str,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    config: Optional[WorldModelConfig] = None,
    seed: int = 42,
) -> tuple[HookedWorldModel, IJEPAAdapter, WorldModelConfig, str]:
    set_seed(seed)
    config = config or build_ijepa_config()
    adapter = IJEPAAdapter(config)

    condition_name = condition.lower()
    if condition_name == "pretrained":
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        adapter.load_state_dict(payload, strict=False)
        label = f"pretrained:{checkpoint.name}"
    elif condition_name == "random_init":
        label = "random_init"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if device is not None:
        adapter = adapter.to(torch.device(device))

    adapter.eval()
    wm = HookedWorldModel(adapter=adapter, config=config, name=f"ijepa_{condition_name}")
    return wm, adapter, config, label


def prepare_example_input(
    image_path: Optional[str | Path] = None,
    imagenet_root: Optional[str | Path] = None,
    image_size: int = 224,
):
    raw_image, metadata = get_sample_image(
        image_path=image_path,
        imagenet_root=imagenet_root,
        image_size=image_size,
    )
    image_tensor = preprocess_image(raw_image, image_size=image_size)
    return raw_image, image_tensor, metadata


def assign_fixed_masks(adapter: IJEPAAdapter, context_ids: list[int], target_ids: list[int]) -> None:
    adapter.last_context_ids = list(context_ids)
    adapter.last_target_ids = list(target_ids)


def collect_condition_artifacts(
    condition: str,
    raw_image,
    image_tensor: torch.Tensor,
    *,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    fixed_masks: Optional[tuple[list[int], list[int]]] = None,
    seed: int = 42,
):
    wm, adapter, config, label = load_condition_world_model(
        condition=condition,
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
    )
    if fixed_masks is not None:
        assign_fixed_masks(adapter, fixed_masks[0], fixed_masks[1])
    artifacts = collect_ijepa_artifacts(wm, adapter, raw_image, image_tensor)
    artifacts.metadata = {"condition": label}
    return wm, adapter, config, artifacts


class NonlinearProber:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def train_probe(
        self,
        activations: torch.Tensor,
        labels: np.ndarray,
        concept_name: str,
        activation_name: str,
    ) -> ProbeMetric:
        X = activations.detach().cpu().numpy() if isinstance(activations, torch.Tensor) else activations
        y = np.asarray(labels)
        is_classification = y.dtype.kind in {"i", "b"} or len(np.unique(y)) < 20

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.seed,
            stratify=y if is_classification else None,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if is_classification:
            model = MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-3,
                max_iter=400,
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            score = float(model.score(X_test, y_test))
            score_name = "accuracy"
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-3,
                max_iter=400,
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = float(r2_score(y_test, preds))
            score_name = "r2"

        return ProbeMetric(
            component=activation_name,
            concept=concept_name,
            probe_family="mlp",
            score=score,
            score_name=score_name,
        )


def _component_sort_key(name: str) -> tuple[int, str]:
    if name in _COMPONENT_ORDER:
        return (_COMPONENT_ORDER.index(name), name)
    return (len(_COMPONENT_ORDER), name)


def build_patch_appearance_labels(raw_image, patch_ids: list[int], grid_size: int) -> dict[str, np.ndarray]:
    array = np.asarray(raw_image.resize((224, 224))).astype(np.float32) / 255.0
    patch_extent = 224 // grid_size
    grayscale = array.mean(axis=2)
    gy, gx = np.gradient(grayscale)
    edge_map = np.sqrt(gx**2 + gy**2)

    brightness = []
    dominant_channel = []
    edge_density = []

    for patch_id in patch_ids:
        row, col = divmod(int(patch_id), grid_size)
        y0 = row * patch_extent
        x0 = col * patch_extent
        patch = array[y0 : y0 + patch_extent, x0 : x0 + patch_extent]
        patch_gray_edges = edge_map[y0 : y0 + patch_extent, x0 : x0 + patch_extent]

        channel_means = patch.mean(axis=(0, 1))
        brightness.append(float(channel_means.mean()))
        dominant_channel.append(int(np.argmax(channel_means)))
        edge_density.append(float(patch_gray_edges.mean()))

    brightness = np.asarray(brightness, dtype=np.float32)
    edge_density = np.asarray(edge_density, dtype=np.float32)
    return {
        "brightness_mean": brightness,
        "brightness_high": (brightness >= np.median(brightness)).astype(np.int64),
        "dominant_channel": np.asarray(dominant_channel, dtype=np.int64),
        "edge_density": edge_density,
        "edge_dense": (edge_density >= np.median(edge_density)).astype(np.int64),
    }


def build_probe_targets(raw_image, patch_ids: list[int], grid_size: int) -> dict[str, np.ndarray]:
    labels = build_patch_labels(patch_ids, grid_size)
    labels.update(build_patch_appearance_labels(raw_image, patch_ids, grid_size))
    return labels


@torch.no_grad()
def _forward_vit_blocks(vit, image_tensor: torch.Tensor, patch_ids: Optional[list[int]], prefix: str):
    x = vit.patch_embed(image_tensor)
    if patch_ids is not None:
        patch_ids_t = torch.tensor(patch_ids, device=x.device)
        x = x[:, patch_ids_t, :]
        pos_embed = vit.pos_embed[:, patch_ids_t, :]
        effective_patch_ids = list(patch_ids)
    else:
        pos_embed = vit.pos_embed
        effective_patch_ids = list(range(vit.patch_embed.n_patches))

    x = x + pos_embed
    x = vit.pos_drop(x)
    outputs = {}
    attention = {}
    for idx, block in enumerate(vit.blocks):
        x = block(x)
        outputs[f"{prefix}.block_{idx}"] = PatchComponent(
            name=f"{prefix}.block_{idx}",
            activations=x.squeeze(0).detach().cpu(),
            patch_ids=list(effective_patch_ids),
            description=f"{prefix} block {idx} outputs.",
        )
        if block.attn.last_attn_weights is not None:
            attention[f"{prefix}.block_{idx}"] = block.attn.last_attn_weights.detach().cpu()
    x = vit.norm(x)
    outputs[f"{prefix}.final"] = PatchComponent(
        name=f"{prefix}.final",
        activations=x.squeeze(0).detach().cpu(),
        patch_ids=list(effective_patch_ids),
        description=f"{prefix} final normalized outputs.",
    )
    return x, outputs, attention


@torch.no_grad()
def _forward_predictor_blocks(
    predictor,
    context_latents: torch.Tensor,
    context_ids: list[int],
    target_ids: list[int],
    prefix: str = "predictor",
):
    context_inputs = predictor.predictor_embed(context_latents)
    context_inputs = context_inputs + predictor.pos_embed[:, context_ids, :]
    target_inputs = predictor.mask_token.expand(context_latents.shape[0], len(target_ids), -1)
    target_inputs = target_inputs + predictor.pos_embed[:, target_ids, :]
    x = torch.cat([context_inputs, target_inputs], dim=1)

    outputs = {}
    attention = {}
    for idx, block in enumerate(predictor.blocks):
        x = block(x)
        target_slice = x[:, len(context_ids) :, :].squeeze(0)
        outputs[f"{prefix}.block_{idx}"] = PatchComponent(
            name=f"{prefix}.block_{idx}",
            activations=target_slice.detach().cpu(),
            patch_ids=list(target_ids),
            description=f"{prefix} target-token outputs after block {idx}.",
        )
        if block.attn.last_attn_weights is not None:
            attention[f"{prefix}.block_{idx}"] = block.attn.last_attn_weights.detach().cpu()
    x = predictor.norm(x)
    target_slice = x[:, len(context_ids) :, :]
    outputs[f"{prefix}.final_hidden"] = PatchComponent(
        name=f"{prefix}.final_hidden",
        activations=target_slice.squeeze(0).detach().cpu(),
        patch_ids=list(target_ids),
        description=f"{prefix} final hidden target tokens before projection.",
    )
    preds = predictor.predictor_project_back(target_slice)
    outputs[f"{prefix}.final"] = PatchComponent(
        name=f"{prefix}.final",
        activations=preds.squeeze(0).detach().cpu(),
        patch_ids=list(target_ids),
        description=f"{prefix} projected target predictions.",
    )
    return preds, outputs, attention


@torch.no_grad()
def collect_manual_components_and_attention(
    adapter: IJEPAAdapter,
    image_tensor: torch.Tensor,
    context_ids: list[int],
    target_ids: list[int],
):
    model_device = next(adapter.context_encoder.parameters()).device
    image_tensor = image_tensor.to(model_device)

    context_final, context_outputs, context_attention = _forward_vit_blocks(
        adapter.context_encoder,
        image_tensor,
        context_ids,
        prefix="context_encoder",
    )
    target_final, target_outputs, target_attention = _forward_vit_blocks(
        adapter.target_encoder,
        image_tensor,
        None,
        prefix="target_encoder",
    )
    predictor_final, predictor_outputs, predictor_attention = _forward_predictor_blocks(
        adapter.predictor,
        context_final,
        context_ids,
        target_ids,
        prefix="predictor",
    )

    derived = {}
    target_subset = target_final[:, target_ids, :].squeeze(0).detach().cpu()
    residual = predictor_final.squeeze(0).detach().cpu() - target_subset
    derived["predictor_residual"] = PatchComponent(
        name="predictor_residual",
        activations=residual,
        patch_ids=list(target_ids),
        description="Predictor targets minus target-encoder targets for masked patches.",
    )

    components = {}
    components.update(context_outputs)
    components.update(target_outputs)
    components.update(predictor_outputs)
    components.update(derived)

    attention = {}
    attention.update(context_attention)
    attention.update(target_attention)
    attention.update(predictor_attention)
    return components, attention


def run_linear_probe_metrics(
    component_name: str,
    activations: torch.Tensor,
    labels_dict: dict[str, np.ndarray],
    *,
    probe_type_overrides: Optional[dict[str, str]] = None,
    seed: int = 42,
) -> list[ProbeMetric]:
    prober = LatentProber(seed=seed)
    metrics = []
    probe_type_overrides = probe_type_overrides or {}
    for concept_name, labels in labels_dict.items():
        probe_type = probe_type_overrides.get(concept_name, "logistic")
        result = prober.train_probe(
            activations=activations,
            labels=labels,
            concept_name=concept_name,
            activation_name=component_name,
            probe_type=probe_type,
        )
        score_name = "r2" if result.r2 is not None else "accuracy"
        score = float(result.r2 if result.r2 is not None else result.accuracy)
        metrics.append(
            ProbeMetric(
                component=component_name,
                concept=concept_name,
                probe_family="linear",
                score=score,
                score_name=score_name,
            )
        )
    return metrics


def run_probe_family_comparison(
    component_name: str,
    activations: torch.Tensor,
    labels_dict: dict[str, np.ndarray],
    *,
    seed: int = 42,
) -> list[ProbeMetric]:
    metrics = run_linear_probe_metrics(
        component_name,
        activations,
        labels_dict,
        probe_type_overrides={
            concept: ("ridge" if labels.dtype.kind == "f" and len(np.unique(labels)) >= 20 else "logistic")
            for concept, labels in labels_dict.items()
        },
        seed=seed,
    )
    nonlinear = NonlinearProber(seed=seed)
    for concept_name, labels in labels_dict.items():
        metrics.append(nonlinear.train_probe(activations, labels, concept_name, component_name))
    return metrics


def summarize_probe_metrics(metrics: list[ProbeMetric]) -> str:
    lines = []
    for metric in sorted(metrics, key=lambda item: (_component_sort_key(item.component), item.concept, item.probe_family)):
        lines.append(
            f"{metric.component}\t{metric.concept}\t{metric.probe_family}\t{metric.score_name}={metric.score:.4f}"
        )
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_baseline_plot(baseline_rows: list[dict[str, Any]], output_path: Path) -> None:
    focus_rows = [
        row for row in baseline_rows if row["component"] in {"z_posterior", "target_encoding", "predictor_targets"} and row["concept"] in {"top_half", "left_half", "brightness_high", "edge_dense"}
    ]
    labels = [f"{row['component']}\n{row['concept']}" for row in focus_rows]
    pretrained = [row["pretrained"] for row in focus_rows]
    random_init = [row["random_init"] for row in focus_rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, pretrained, width=width, label="pretrained", color="#1f77b4")
    ax.bar(x + width / 2, random_init, width=width, label="random_init", color="#ff7f0e")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Pretrained vs Random-Init Probe Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_layerwise_plot(layer_rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    families = [
        ("context_encoder", axes[0]),
        ("target_encoder", axes[1]),
        ("predictor", axes[2]),
    ]
    concept_order = ["top_half", "left_half", "brightness_high"]

    for family_name, ax in families:
        family_rows = [row for row in layer_rows if row["family"] == family_name and row["concept"] in concept_order]
        for concept in concept_order:
            concept_rows = sorted(
                [row for row in family_rows if row["concept"] == concept],
                key=lambda row: row["layer_index"],
            )
            if concept_rows:
                ax.plot(
                    [row["layer_index"] for row in concept_rows],
                    [row["score"] for row in concept_rows],
                    marker="o",
                    label=concept,
                )
        ax.set_title(family_name)
        ax.set_xlabel("layer")
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("probe score")
    axes[2].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_nonlinear_gap_plot(metric_rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped = {}
    for row in metric_rows:
        # Skip regression metrics — R² is numerically unstable with small MLP on tiny datasets
        if row.get("score_name") != "accuracy":
            continue
        key = (row["component"], row["concept"])
        grouped.setdefault(key, {})[row["probe_family"]] = row

    labels = []
    gaps = []
    for (component, concept), entries in grouped.items():
        if "linear" in entries and "mlp" in entries:
            labels.append(f"{component}\n{concept}")
            gaps.append(entries["mlp"]["score"] - entries["linear"]["score"])

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ca02c" if gap >= 0 else "#d62728" for gap in gaps]
    ax.bar(np.arange(len(labels)), gaps, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.0, color="black", linewidth=1)
    for i, gap in enumerate(gaps):
        ax.text(i, gap + (0.005 if gap >= 0 else -0.012), f"{gap:+.2f}", ha="center", va="bottom" if gap >= 0 else "top", fontsize=7.5)
    ax.set_title("Nonlinear Probe Gain (MLP − Linear accuracy)\nclassification concepts only", fontweight="bold", pad=8)
    ax.set_ylabel("Accuracy delta (MLP − linear)")
    ax.set_ylim(-0.25, 0.25)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_mask_sensitivity_plot(mask_rows: list[dict[str, Any]], output_path: Path) -> None:
    seeds = sorted({row["mask_seed"] for row in mask_rows})
    x = np.arange(len(seeds))
    seed_to_x = {s: i for i, s in enumerate(seeds)}
    concepts = ["top_half", "left_half", "brightness_high"]
    concept_colors = {"top_half": "#4C72B0", "left_half": "#DD8452", "brightness_high": "#55A868"}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for concept in concepts:
        rows = sorted([row for row in mask_rows if row["concept"] == concept], key=lambda r: r["mask_seed"])
        ys = [row["score"] for row in rows]
        xs = [seed_to_x[row["mask_seed"]] for row in rows]
        ax.plot(xs, ys, marker="o", label=concept.replace("_", " "), color=concept_colors.get(concept, None), linewidth=1.8)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1.0, alpha=0.7, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"mask config {s}" for s in seeds], fontsize=9)
    ax.set_ylabel("Probe accuracy (predictor targets)", fontsize=9)
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Mask Configuration Sensitivity\n(predictor spatial decodability across different mask seeds)", fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_attention_head_plot(head_rows: list[dict[str, Any]], output_path: Path) -> None:
    # Only show heads that beat chance, top 15 per concept
    concepts = sorted({row["concept"] for row in head_rows})
    concept_colors = {c: color for c, color in zip(concepts, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"])}
    above_chance = [row for row in head_rows if row["score"] > 0.5]
    top_rows = sorted(above_chance, key=lambda row: row["score"], reverse=True)[:15]

    labels = [f"{row['head_name'].replace('predictor.', '')}\n({row['concept'].replace('_', ' ')})" for row in top_rows]
    scores = [row["score"] for row in top_rows]
    colors = [concept_colors.get(row["concept"], "#9467bd") for row in top_rows]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(np.arange(len(labels)), scores, color=colors, alpha=0.85, edgecolor="white")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1.0, alpha=0.7, label="chance (0.5)")
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Probe accuracy", fontsize=9)
    ax.set_title("Top Predictor Attention Heads by Probe Score\n(above-chance only, colored by concept)", fontweight="bold", pad=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    import matplotlib.patches as _mp
    import matplotlib.lines as _ml
    legend_handles: list[Any] = [_mp.Patch(color=concept_colors[c], alpha=0.85, label=c.replace("_", " ")) for c in concepts if c in concept_colors]
    legend_handles.append(_ml.Line2D([0], [0], color="#888", linestyle="--", label="chance (0.5)"))
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def summarize_attention_heads(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in sorted(rows, key=lambda item: (-item["score"], item["head_name"], item["concept"])):
        lines.append(f"{row['head_name']}\t{row['concept']}\t{row['score_name']}={row['score']:.4f}")
    return "\n".join(lines)


def probe_predictor_attention_heads(
    attention_maps: dict[str, torch.Tensor],
    context_ids: list[int],
    target_ids: list[int],
    grid_size: int,
    *,
    seed: int = 42,
) -> list[AttentionHeadProbeResult]:
    labels = build_patch_labels(target_ids, grid_size)
    target_ids_np = np.asarray(target_ids, dtype=np.int64)
    labels["target_row_bucket"] = target_ids_np // grid_size
    labels["target_col_bucket"] = target_ids_np % grid_size
    concepts = ["top_half", "left_half", "target_row_bucket", "target_col_bucket"]

    results = []
    prober = LatentProber(seed=seed)
    for block_name, attn in attention_maps.items():
        if not block_name.startswith("predictor.block_"):
            continue
        n_context = len(context_ids)
        target_attention = attn[0, :, n_context:, :n_context]
        for head_idx in range(target_attention.shape[0]):
            features = target_attention[head_idx].detach().cpu()
            for concept in concepts:
                result = prober.train_probe(
                    features,
                    labels[concept],
                    concept,
                    f"{block_name}.head_{head_idx}",
                    probe_type="logistic",
                    use_cv=False,
                )
                score_name = "r2" if result.r2 is not None else "accuracy"
                score = float(result.r2 if result.r2 is not None else result.accuracy)
                results.append(
                    AttentionHeadProbeResult(
                        head_name=f"{block_name}.head_{head_idx}",
                        concept=concept,
                        score=score,
                        score_name=score_name,
                    )
                )
    return results


def save_heatmap_for_component(component: PatchComponent, grid_size: int, raw_image, output_path: Path) -> None:
    values = component.activations.norm(dim=1)
    heatmap = build_patch_heatmap(component.patch_ids, values, grid_size)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(raw_image.resize((224, 224)), alpha=0.55)
    ax.imshow(heatmap, cmap="coolwarm", alpha=0.55, extent=(0, 224, 224, 0), interpolation="nearest")
    ax.set_title(component.name)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def output_path(filename: str) -> Path:
    return CURRENT_OUTPUT_DIR / filename
