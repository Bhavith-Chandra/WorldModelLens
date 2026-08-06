"""Smoke tests for World Model Lens."""

import pytest


def test_all_exports_importable():
    """Test that all exports are importable."""
    from world_model_lens import (
        __version__,
        HookedWorldModel,
        WorldState,
        WorldTrajectory,
        ActivationCache,
        HookPoint,
        HookContext,
        HookRegistry,
        WorldModelConfig,
        BaseModelAdapter,
        WorldDynamics,
        WorldModelOutput,
    )

    assert __version__ == "0.2.0"


def test_config_instantiation():
    """Test config can be instantiated."""
    from world_model_lens import WorldModelConfig

    cfg = WorldModelConfig()
    assert cfg.d_h == 512
    assert cfg.n_cat == 32
    assert cfg.n_cls == 32


def test_adapter_instantiation():
    """Test adapter can be instantiated."""
    from world_model_lens import WorldModelConfig
    from world_model_lens.backends.dreamerv3 import DreamerV3Adapter

    cfg = WorldModelConfig(d_h=32, n_cat=4, n_cls=4, d_action=4)
    adapter = DreamerV3Adapter(cfg)
    assert adapter is not None


def test_hooked_world_model_instantiation():
    """Test HookedWorldModel can be instantiated."""
    from world_model_lens import WorldModelConfig, HookedWorldModel
    from world_model_lens.backends.dreamerv3 import DreamerV3Adapter

    cfg = WorldModelConfig(d_h=32, n_cat=4, n_cls=4, d_action=4)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    assert wm is not None
