from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import (
    build_probe_targets,
    collect_condition_artifacts,
    collect_manual_components_and_attention,
    output_path,
    prepare_example_input,
    probe_predictor_attention_heads,
    run_linear_probe_metrics,
    run_probe_family_comparison,
    save_attention_head_plot,
    save_baseline_plot,
    save_heatmap_for_component,
    save_layerwise_plot,
    save_mask_sensitivity_plot,
    save_nonlinear_gap_plot,
    set_output_dir,
    summarize_attention_heads,
    summarize_probe_metrics,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="I-JEPA experiment suite for probing controls and extensions."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Optional pretrained checkpoint path."
    )
    parser.add_argument("--image", type=str, default=None, help="Optional image path.")
    parser.add_argument("--imagenet-root", type=str, default=None, help="Optional ImageNet root.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: auto-detect cuda).")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Optional output directory override."
    )
    parser.add_argument(
        "--mask-seeds",
        type=int,
        nargs="*",
        default=[7, 11, 19, 23, 31, 47],
        help="Mask seeds for sensitivity sweep.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic comparisons."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run a smaller sweep for smoke testing."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        set_output_dir(args.output_dir)

    mask_seeds = args.mask_seeds[:3] if args.fast else args.mask_seeds

    raw_image, image_tensor, metadata = prepare_example_input(
        image_path=args.image,
        imagenet_root=args.imagenet_root,
        image_size=224,
    )

    _, pretrained_adapter, _, pretrained_artifacts = collect_condition_artifacts(
        condition="pretrained",
        checkpoint_path=args.checkpoint,
        raw_image=raw_image,
        image_tensor=image_tensor,
        device=args.device,
        seed=args.seed,
    )
    fixed_masks = (pretrained_artifacts.context_ids, pretrained_artifacts.target_ids)

    _, random_adapter, _, random_artifacts = collect_condition_artifacts(
        condition="random_init",
        raw_image=raw_image,
        image_tensor=image_tensor,
        device=args.device,
        fixed_masks=fixed_masks,
        seed=args.seed,
    )

    baseline_rows = []
    for component_name in ["encoder.out", "target_encoder_out", "predictor_out"]:
        pretrained_component = pretrained_artifacts.components[component_name]
        random_component = random_artifacts.components[component_name]
        labels = build_probe_targets(
            raw_image, pretrained_component.patch_ids, pretrained_artifacts.grid_size
        )
        pretrained_metrics = run_linear_probe_metrics(
            component_name, pretrained_component.activations, labels, seed=args.seed
        )
        random_metrics = run_linear_probe_metrics(
            component_name, random_component.activations, labels, seed=args.seed
        )
        random_by_concept = {metric.concept: metric for metric in random_metrics}
        for metric in pretrained_metrics:
            baseline_rows.append(
                {
                    "component": component_name,
                    "concept": metric.concept,
                    "score_name": metric.score_name,
                    "pretrained": metric.score,
                    "random_init": random_by_concept[metric.concept].score,
                    "delta": metric.score - random_by_concept[metric.concept].score,
                }
            )

    baseline_lines = [
        "I-JEPA Random-Init Baseline",
        "=" * 40,
        f"image_source: {metadata.get('source', 'unknown')}",
        f"context_patches: {len(pretrained_artifacts.context_ids)}",
        f"target_patches: {len(pretrained_artifacts.target_ids)}",
        "",
    ]
    for row in baseline_rows:
        baseline_lines.append(
            f"{row['component']}\t{row['concept']}\t{row['score_name']}\tpretrained={row['pretrained']:.4f}\trandom_init={row['random_init']:.4f}\tdelta={row['delta']:.4f}"
        )
    write_text(output_path("ijepa_random_init_baseline.txt"), "\n".join(baseline_lines))
    save_baseline_plot(baseline_rows, output_path("ijepa_random_init_baseline.png"))

    manual_components, attention_maps = collect_manual_components_and_attention(
        pretrained_adapter,
        image_tensor,
        pretrained_artifacts.context_ids,
        pretrained_artifacts.target_ids,
    )

    residual_component = manual_components["predictor_residual"]
    residual_labels = build_probe_targets(
        raw_image, residual_component.patch_ids, pretrained_artifacts.grid_size
    )
    residual_metrics = []
    for component_name in ["predictor_out", "target_encoder_out"]:
        component = pretrained_artifacts.components[component_name]
        component_labels = build_probe_targets(
            raw_image, component.patch_ids, pretrained_artifacts.grid_size
        )
        residual_metrics.extend(
            run_linear_probe_metrics(
                component_name, component.activations, component_labels, seed=args.seed
            )
        )
    residual_metrics.extend(
        run_linear_probe_metrics(
            "predictor_residual", residual_component.activations, residual_labels, seed=args.seed
        )
    )
    write_text(
        output_path("ijepa_predictor_residual_summary.txt"),
        "I-JEPA Predictor Residual Probing\n"
        + "=" * 40
        + "\n"
        + summarize_probe_metrics(residual_metrics),
    )
    save_heatmap_for_component(
        residual_component,
        pretrained_artifacts.grid_size,
        raw_image,
        output_path("ijepa_predictor_residual_heatmap.png"),
    )

    layer_rows = []
    layer_component_names = [
        name
        for name in manual_components
        if name.startswith("context_encoder.block_")
        or name.startswith("target_encoder.block_")
        or name.startswith("predictor.block_")
    ]
    probe_concepts = {"top_half", "left_half", "brightness_high"}
    for component_name in sorted(layer_component_names):
        component = manual_components[component_name]
        labels = {
            key: value
            for key, value in build_probe_targets(
                raw_image, component.patch_ids, pretrained_artifacts.grid_size
            ).items()
            if key in probe_concepts
        }
        metrics = run_linear_probe_metrics(
            component_name, component.activations, labels, seed=args.seed
        )
        family = component_name.split(".", 1)[0]
        layer_index = int(component_name.split("block_")[1])
        for metric in metrics:
            layer_rows.append(
                {
                    "component": component_name,
                    "family": family,
                    "layer_index": layer_index,
                    "concept": metric.concept,
                    "score": metric.score,
                    "score_name": metric.score_name,
                }
            )
    layer_lines = ["I-JEPA Layer-wise Probing", "=" * 40]
    for row in sorted(
        layer_rows, key=lambda item: (item["family"], item["layer_index"], item["concept"])
    ):
        layer_lines.append(
            f"{row['family']}\tlayer={row['layer_index']}\t{row['concept']}\t{row['score_name']}={row['score']:.4f}"
        )
    write_text(output_path("ijepa_layerwise_probing.txt"), "\n".join(layer_lines))
    save_layerwise_plot(layer_rows, output_path("ijepa_layerwise_probing.png"))

    nonlinear_rows = []
    nonlinear_components = [
        "encoder.out",
        "target_encoder_out",
        "predictor_out",
        "predictor_residual",
    ]
    nonlinear_concepts = {
        "top_half",
        "left_half",
        "brightness_high",
        "edge_dense",
        "brightness_mean",
        "edge_density",
    }
    for component_name in nonlinear_components:
        component = (
            residual_component
            if component_name == "predictor_residual"
            else pretrained_artifacts.components[component_name]
        )
        labels = {
            key: value
            for key, value in build_probe_targets(
                raw_image, component.patch_ids, pretrained_artifacts.grid_size
            ).items()
            if key in nonlinear_concepts
        }
        metrics = run_probe_family_comparison(
            component_name, component.activations, labels, seed=args.seed
        )
        for metric in metrics:
            nonlinear_rows.append(
                {
                    "component": metric.component,
                    "concept": metric.concept,
                    "probe_family": metric.probe_family,
                    "score": metric.score,
                    "score_name": metric.score_name,
                }
            )
    nonlinear_lines = ["I-JEPA Nonlinear Probe Comparison", "=" * 40]
    for row in sorted(
        nonlinear_rows, key=lambda item: (item["component"], item["concept"], item["probe_family"])
    ):
        nonlinear_lines.append(
            f"{row['component']}\t{row['concept']}\t{row['probe_family']}\t{row['score_name']}={row['score']:.4f}"
        )
    write_text(output_path("ijepa_nonlinear_probe_comparison.txt"), "\n".join(nonlinear_lines))
    save_nonlinear_gap_plot(nonlinear_rows, output_path("ijepa_nonlinear_probe_gap.png"))

    mask_rows = []
    for mask_seed in mask_seeds:
        _, _, _, mask_artifacts = collect_condition_artifacts(
            condition="pretrained",
            checkpoint_path=args.checkpoint,
            raw_image=raw_image,
            image_tensor=image_tensor,
            device=args.device,
            seed=mask_seed,
        )
        component = mask_artifacts.components["predictor_out"]
        labels = build_probe_targets(raw_image, component.patch_ids, mask_artifacts.grid_size)
        metrics = run_linear_probe_metrics(
            "predictor_out", component.activations, labels, seed=args.seed
        )
        for metric in metrics:
            if metric.concept in {"top_half", "left_half", "brightness_high"}:
                mask_rows.append(
                    {
                        "mask_seed": mask_seed,
                        "concept": metric.concept,
                        "score": metric.score,
                        "score_name": metric.score_name,
                        "target_patches": len(component.patch_ids),
                    }
                )
    mask_lines = ["I-JEPA Mask Sensitivity", "=" * 40]
    for row in mask_rows:
        mask_lines.append(
            f"seed={row['mask_seed']}\t{row['concept']}\t{row['score_name']}={row['score']:.4f}\ttarget_patches={row['target_patches']}"
        )
    write_text(output_path("ijepa_mask_sensitivity.txt"), "\n".join(mask_lines))
    save_mask_sensitivity_plot(mask_rows, output_path("ijepa_mask_sensitivity.png"))

    attention_results = probe_predictor_attention_heads(
        attention_maps,
        pretrained_artifacts.context_ids,
        pretrained_artifacts.target_ids,
        pretrained_artifacts.grid_size,
        seed=args.seed,
    )
    attention_rows = [
        {
            "head_name": result.head_name,
            "concept": result.concept,
            "score": result.score,
            "score_name": result.score_name,
        }
        for result in attention_results
    ]
    write_text(
        output_path("ijepa_attention_head_probing.txt"),
        "I-JEPA Predictor Attention Head Probing\n"
        + "=" * 40
        + "\n"
        + summarize_attention_heads(attention_rows),
    )
    save_attention_head_plot(attention_rows, output_path("ijepa_attention_head_probing.png"))

    print("=" * 70)
    print("I-JEPA experiment suite completed.")
    print(f"image_source: {metadata.get('source', 'unknown')}")
    print(f"outputs: {Path(output_path('ijepa_random_init_baseline.txt')).parent}")
    print("generated:")
    print("  - random-init baseline")
    print("  - predictor residual probing")
    print("  - layer-wise probing")
    print("  - nonlinear probe comparison")
    print("  - mask sensitivity sweep")
    print("  - predictor attention-head probing")
    print("=" * 70)


if __name__ == "__main__":
    main()
