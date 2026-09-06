import torch

from world_model_lens.analysis import EMAStructureAnalyzer
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig


def _adapter() -> IJEPAAdapter:
    config = WorldModelConfig(
        backend="ijepa",
        img_size=32,
        patch_size=8,
        d_embed=16,
        n_layers=2,
        n_heads=2,
        predictor_embed_dim=32,
        predictor_depth=2,
        predictor_heads=2,
    )
    return IJEPAAdapter(config)


def test_ema_structure_divergence_is_zero_for_copied_encoders():
    adapter = _adapter()
    result = EMAStructureAnalyzer(adapter).compare_encoders(torch.randn(2, 3, 32, 32))

    assert result.delta.shape == (2, 16, 16)
    assert result.token_l2.shape == (2, 16)
    assert result.mean_l2 == 0.0
    assert result.layer_mean_l2 == {0: 0.0, 1: 0.0}


def test_ema_structure_divergence_detects_context_target_drift():
    adapter = _adapter()
    with torch.no_grad():
        adapter.context_encoder.pos_embed.add_(0.5)

    result = EMAStructureAnalyzer(adapter).compare_encoders(torch.randn(1, 3, 32, 32))

    assert result.mean_l2 > 0.0
    assert all(value > 0.0 for value in result.layer_mean_l2.values())


def test_inpainting_candidates_are_scored_against_fixed_context_prediction():
    adapter = _adapter()
    source = torch.randn(1, 3, 32, 32)
    candidates = torch.cat([source, source + 0.25], dim=0)
    scores = EMAStructureAnalyzer(adapter).score_inpainting_candidates(
        source,
        candidates,
        context_ids=list(range(8)),
        target_ids=list(range(8, 12)),
    )

    assert len(scores) == 2
    assert scores[0].reference_mse == 0.0
    assert scores[1].reference_mse > scores[0].reference_mse
