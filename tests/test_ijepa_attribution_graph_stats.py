import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "ijepa"
sys.path.insert(0, str(EXAMPLE_DIR))
spec = importlib.util.spec_from_file_location(
    "ijepa_attribution_graph", EXAMPLE_DIR / "attribution_graph.py"
)
attribution_graph = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = attribution_graph
spec.loader.exec_module(attribution_graph)


def make_image_dir():
    path = Path(__file__).resolve().parent / f".tmp_ijepa_images_{uuid4().hex}"
    path.mkdir()
    return path


def cleanup_image_dir(path):
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


def make_run(image_idx, top_context_ids, top_heads, head_scores, metrics=None):
    return attribution_graph.AttributionRun(
        image_idx=image_idx,
        target_id=42,
        layer_idx=3,
        head_scores=np.asarray(head_scores, dtype=np.float64),
        mean_attributions=np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
        top_context_ids=list(top_context_ids),
        top_heads=list(top_heads),
        metrics=metrics or {"top_k_mass": 0.5 + image_idx * 0.1},
    )


def test_attribution_metrics_reports_top_mass_and_entropy():
    metrics = attribution_graph.attribution_metrics(np.array([0.4, 0.3, 0.2, 0.1]), k=2)

    assert np.isclose(metrics["top_k_mass"], 0.7)
    assert metrics["max_weight"] == 0.4
    assert 0.0 < metrics["entropy"] <= 1.0
    assert metrics["effective_patches"] > 1.0


def test_summarize_attribution_runs_uses_image_level_mean_std():
    runs = [
        make_run(0, [1, 2], [0, 1], [0.9, 0.8, 0.1], {"top_k_mass": 0.4}),
        make_run(1, [1, 3], [0, 2], [0.8, 0.1, 0.7], {"top_k_mass": 0.6}),
        make_run(2, [1, 2], [0, 1], [0.7, 0.6, 0.2], {"top_k_mass": 0.8}),
    ]

    summary = attribution_graph.summarize_attribution_runs(runs, k=2)

    assert np.isclose(summary["top_k_mass"]["mean"], 0.6)
    assert np.isclose(summary["top_k_mass"]["std"], 0.2)
    assert np.isclose(summary["top_patch_jaccard"]["mean"], (1 / 3 + 1 + 1 / 3) / 3)
    assert np.isclose(summary["top_head_jaccard"]["mean"], (1 / 3 + 1 + 1 / 3) / 3)
    assert "head_rank_spearman" in summary
    assert -1.0 <= summary["head_rank_spearman"]["mean"] <= 1.0


def test_rank_correlation_rewards_matching_head_order():
    same = attribution_graph._rank_correlation(
        np.array([0.1, 0.5, 0.9]),
        np.array([0.2, 0.6, 1.0]),
    )
    opposite = attribution_graph._rank_correlation(
        np.array([0.1, 0.5, 0.9]),
        np.array([1.0, 0.6, 0.2]),
    )

    assert np.isclose(same, 1.0)
    assert np.isclose(opposite, -1.0)


def test_load_consistency_image_set_is_filename_agnostic():
    image_dir = make_image_dir()
    raw = Image.new("RGB", (16, 16), color=(10, 20, 30))
    try:
        for name in ["sample_a.jpg", "anything.png", "class-member.bmp"]:
            Image.new("RGB", (16, 16), color=(40, 50, 60)).save(image_dir / name)

        image_set = attribution_graph.load_consistency_image_set(
            raw,
            n_images=3,
            image_dir=str(image_dir),
            dataset_label="test class",
        )

        assert image_set.mode == "cross_image"
        assert image_set.label == "test class"
        assert len(image_set.images) == 3
        assert {path.name for path in image_set.paths} == {"anything.png", "class-member.bmp", "sample_a.jpg"}
    finally:
        cleanup_image_dir(image_dir)


def test_load_consistency_image_set_requires_explicit_augmented_fallback():
    image_dir = make_image_dir()
    raw = Image.new("RGB", (16, 16), color=(10, 20, 30))
    try:
        Image.new("RGB", (16, 16), color=(40, 50, 60)).save(image_dir / "single.jpg")

        with pytest.raises(ValueError, match="allow_augmented_fallback=True"):
            attribution_graph.load_consistency_image_set(raw, n_images=2, image_dir=str(image_dir))

        image_set = attribution_graph.load_consistency_image_set(
            raw,
            n_images=2,
            image_dir=str(image_dir),
            allow_augmented_fallback=True,
        )

        assert image_set.mode == "augmented_fallback"
        assert len(image_set.images) == 2
        assert image_set.paths == []
    finally:
        cleanup_image_dir(image_dir)
