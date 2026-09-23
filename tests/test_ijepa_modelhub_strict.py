"""Coverage tests for the strict official I-JEPA ModelHub path."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.hub.model_hub import ModelHub


def _official_encoder(state):
    return {
        "module."
        + key.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2."): value
        for key, value in state.items()
    }


def _official_predictor(state):
    official = {}
    for key, value in state.items():
        if key == "pos_embed":
            key = "predictor_pos_embed"
        elif key.startswith("blocks."):
            key = "predictor_blocks." + key[len("blocks.") :]
        elif key.startswith("norm."):
            key = "predictor_norm." + key[len("norm.") :]
        elif key.startswith("predictor_project_back."):
            key = "predictor_proj." + key[len("predictor_project_back.") :]
        key = key.replace(".mlp.0.", ".mlp.fc1.")
        key = key.replace(".mlp.2.", ".mlp.fc2.")
        official["module." + key] = value
    return official


def _checkpoint():
    config = WorldModelConfig(
        backend="ijepa",
        d_embed=16,
        n_layers=1,
        n_heads=16,
        predictor_embed_dim=16,
        predictor_depth=1,
        predictor_heads=16,
        img_size=16,
        patch_size=16,
    )
    config.qkv_bias = True
    config.norm_eps = 1e-6
    source = IJEPAAdapter(config)

    encoder = _official_encoder(source.context_encoder.state_dict())
    target = deepcopy(encoder)
    target["module.patch_embed.proj.bias"] = (
        target["module.patch_embed.proj.bias"].clone() + 1.0
    )
    return {
        "encoder": encoder,
        "target_encoder": target,
        "predictor": _official_predictor(source.predictor.state_dict()),
    }


def test_strict_official_checkpoint_loads_every_component(tmp_path):
    path = tmp_path / "official_ijepa.pth"
    torch.save(_checkpoint(), path)

    adapter = ModelHub.load_checkpoint(path, backend="ijepa", device="cpu")

    assert adapter.checkpoint_coverage["context_encoder"]["missing"] == []
    assert adapter.checkpoint_coverage["target_encoder"]["unexpected"] == []
    assert adapter.context_encoder.blocks[0].attn.qkv.bias is not None
    assert adapter.context_encoder.blocks[0].norm1.eps == pytest.approx(1e-6)
    assert not torch.equal(
        adapter.context_encoder.patch_embed.proj.bias,
        adapter.target_encoder.patch_embed.proj.bias,
    )


def test_official_checkpoint_requires_target_encoder(tmp_path):
    checkpoint = _checkpoint()
    del checkpoint["target_encoder"]
    path = tmp_path / "missing_target.pth"
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="target_encoder"):
        ModelHub.load_checkpoint(path, backend="ijepa", device="cpu")
