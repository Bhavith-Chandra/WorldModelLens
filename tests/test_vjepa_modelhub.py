"""V-JEPA checkpoint mapping and ModelHub wiring without a network download."""

from unittest.mock import patch

import pytest
import torch

from world_model_lens.backends import VJEPAAdapter
from world_model_lens.backends.registry import REGISTRY
from world_model_lens.hub import ModelHub
from world_model_lens.core.config import WorldModelConfig


def test_vjepa_vitl16_defaults_match_meta_config():
    config = WorldModelConfig.vjepa_vitl16()
    assert (config.img_size, config.patch_size, config.num_frames, config.tubelet_size) == (224, 16, 16, 2)
    assert (config.d_embed, config.n_layers, config.n_heads) == (1024, 24, 16)
    assert (config.predictor_embed_dim, config.predictor_depth, config.predictor_heads) == (384, 12, 16)


def test_adapter_default_builds_vitl16_architecture_without_allocating_weights():
    with torch.device("meta"):
        adapter = VJEPAAdapter()
    assert len(adapter.context_encoder.blocks) == 24
    assert adapter.context_encoder.blocks[0].attn.num_heads == 16
    assert adapter.context_encoder.blocks[0].attn.qkv.bias is not None
    assert adapter.context_encoder.blocks[0].norm1.eps == 1e-6
    assert len(adapter.predictor.blocks) == 12
    assert adapter.predictor.blocks[0].attn.num_heads == 16


def test_official_vjepa_checkpoint_is_registered_for_download():
    info = ModelHub.info("vjepa-vit-l-224")
    assert info.backend == "vjepa"
    assert info.is_downloadable
    assert info.source_url == "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar"


def test_official_vjepa_pull_uses_meta_url():
    with patch.object(ModelHub, "_download_direct_url", return_value="checkpoint.pth.tar") as download:
        assert ModelHub.pull("vjepa-vit-l-224") == "checkpoint.pth.tar"
    download.assert_called_once_with(
        "https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar",
        "vjepa_vitl16.pth.tar",
        cache_dir=None,
        force=False,
    )


def test_official_vjepa_load_calls_adapter(tmp_path):
    path = tmp_path / "model.pth.tar"
    with patch.object(ModelHub, "pull", return_value=str(path)) as pull, patch.object(
        VJEPAAdapter, "from_checkpoint", return_value=object()
    ) as load:
        # Use a real tiny adapter so the device/eval integration is exercised.
        config = WorldModelConfig(
            backend="vjepa", img_size=32, patch_size=16, num_frames=2,
            d_embed=12, n_layers=1, n_heads=3,
            predictor_embed_dim=12, predictor_depth=1, predictor_heads=3,
        )
        adapter = VJEPAAdapter(config)
        load.return_value = adapter
        assert ModelHub.load("vjepa-vit-l-224") is adapter
    pull.assert_called_once()
    load.assert_called_once_with(str(path))
    assert not adapter.training


def test_adapter_exported():
    assert VJEPAAdapter.__name__ == "VJEPAAdapter"
    info = REGISTRY.get_info("vjepa")
    assert info.supports_video and not info.supports_rl


def test_meta_style_checkpoint_loads_all_three_components_strictly(tmp_path):
    config = WorldModelConfig(
        backend="vjepa", img_size=32, patch_size=16, num_frames=2,
        d_embed=12, n_layers=1, n_heads=3,
        predictor_embed_dim=12, predictor_depth=1, predictor_heads=3,
    )
    original = VJEPAAdapter(config)
    with torch.no_grad():
        original.context_encoder.pos_embed.fill_(1)
        original.target_encoder.pos_embed.fill_(2)
        original.predictor.pos_embed.fill_(3)
        original.predictor.mask_tokens[1].fill_(4)

    def official_keys(state, predictor=False):
        mapped = {}
        for key, value in state.items():
            key = key.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
            if predictor:
                for ours, official in (
                    ("blocks.", "predictor_blocks."),
                    ("norm.", "predictor_norm."),
                    ("predictor_project_back.", "predictor_proj."),
                    ("pos_embed", "predictor_pos_embed"),
                ):
                    if key.startswith(ours):
                        key = official + key[len(ours):]
                        break
            mapped["module.backbone." + key] = value
        return mapped

    checkpoint = {
        "encoder": official_keys(original.context_encoder.state_dict()),
        "target_encoder": official_keys(original.target_encoder.state_dict()),
        "predictor": official_keys(original.predictor.state_dict(), predictor=True),
    }
    path = tmp_path / "official-style.pth.tar"
    torch.save(checkpoint, path)
    loaded = VJEPAAdapter.from_checkpoint(str(path), config=config)
    assert torch.equal(loaded.context_encoder.pos_embed, torch.ones_like(loaded.context_encoder.pos_embed))
    assert torch.equal(loaded.target_encoder.pos_embed, 2 * torch.ones_like(loaded.target_encoder.pos_embed))
    assert torch.equal(loaded.predictor.pos_embed, 3 * torch.ones_like(loaded.predictor.pos_embed))
    assert torch.equal(loaded.predictor.mask_tokens[1], 4 * torch.ones_like(loaded.predictor.mask_tokens[1]))
    assert loaded.predict_masked(torch.randn(2, 3, 2, 32, 32), torch.tensor([[0, 1], [1, 0]]), torch.tensor([[2, 3], [3, 2]])).shape == (2, 2, 12)

    del checkpoint["target_encoder"]["module.backbone.pos_embed"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        VJEPAAdapter.from_checkpoint(str(path), config=config)
