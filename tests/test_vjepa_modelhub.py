"""ModelHub must expose the official V-JEPA file without claiming a valid load."""

from unittest.mock import patch

import pytest
import torch

from world_model_lens.backends import VJEPAAdapter
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


def test_official_vjepa_load_fails_before_download():
    with patch.object(ModelHub, "pull") as pull:
        with pytest.raises(NotImplementedError, match="faithful checkpoint load"):
            ModelHub.load("vjepa-vit-l-224")
    pull.assert_not_called()


def test_adapter_exported():
    assert VJEPAAdapter.__name__ == "VJEPAAdapter"


def test_meta_style_checkpoint_cannot_silently_replace_target_encoder(tmp_path):
    path = tmp_path / "official-style.pth.tar"
    torch.save({"encoder": {}, "target_encoder": {}, "predictor": {}}, path)
    with pytest.raises(NotImplementedError, match="separate target_encoder"):
        VJEPAAdapter.from_checkpoint(str(path))
