"""Tests for the faithful DreamerV3 port and its official-checkpoint loader."""

import os
import pickle

import numpy as np
import pytest
import torch

from world_model_lens.backends.dreamerv3 import (
    DreamerV3Adapter,
    arch_from_params,
    default_arch,
    read_official_checkpoint,
    twohot_bins,
    twohot_pred,
)

SMALL = dict(deter=64, hidden=16, units=16, depth=2, stoch=8, classes=4, action_dim=6,
             img_shape=(32, 32, 1))


def _official_params(adapter: DreamerV3Adapter) -> dict:
    """Export an adapter's parameters under official names, plus optimizer junk."""
    params = {k.replace(".", "/"): p.detach().numpy().copy() for k, p in adapter.named_parameters()}
    params["opt/state/1/0"] = np.zeros((), np.int32)
    params["retnorm/hi/value"] = np.float32(1.0)
    return params


def _step(adapter, seed=0):
    g = torch.Generator().manual_seed(seed)
    h, z = adapter.initial_state(2)
    img = torch.randint(0, 255, (2, 32, 32, 1), generator=g).float()
    a = torch.tensor([1, 4])
    h = adapter.transition(h, z, a)
    post, _ = adapter.encode(img, h)
    z = adapter.sample_z(post, noise=torch.rand(post.shape, generator=g))
    return dict(h=h, post=post, prior=adapter.dynamics(h), z=z,
                rew=adapter.predict_reward(h, z), con=adapter.predict_continue(h, z),
                pol=adapter.actor_forward(h, z), val=adapter.critic_forward(h, z),
                img=adapter.decode(h, z))


def test_arch_inference_roundtrip():
    adapter = DreamerV3Adapter(arch=default_arch(**SMALL))
    arch = arch_from_params(_official_params(adapter), stoch=8)
    for key in ("deter", "hidden", "stoch", "classes", "blocks", "action_dim", "img_shape",
                "depth", "mults", "units", "bspace", "rew_layers", "pol_layers"):
        assert arch[key] == adapter.arch[key], key


def test_official_pkl_loads_one_to_one(tmp_path):
    torch.manual_seed(0)
    src = DreamerV3Adapter(arch=default_arch(**SMALL)).eval()
    path = tmp_path / "agent.pkl"
    with open(path, "wb") as f:
        pickle.dump({"params": _official_params(src), "counters": {}}, f)
    dst = DreamerV3Adapter.from_checkpoint(path, stoch=8)
    rep = dst.load_report
    assert rep["mapped"] == rep["total_adapter_params"]
    assert not rep["missing"] and not rep["unexpected"] and not rep["shape_mismatch"]
    assert rep["skipped_optimizer"] == 1 and rep["extras"] == ["retnorm/hi/value"]
    with torch.no_grad():
        a, b = _step(src), _step(dst)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_state_dict_roundtrip_keeps_latent_split(tmp_path):
    src = DreamerV3Adapter(arch=default_arch(**SMALL))
    path = tmp_path / "adapter.pt"
    torch.save(src.state_dict(), path)
    dst = DreamerV3Adapter.from_checkpoint(path)
    assert dst.stoch_shape == (8, 4)


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_loader_refuses_non_numpy_globals(tmp_path):
    path = tmp_path / "agent.pkl"
    with open(path, "wb") as f:
        pickle.dump({"params": {"x": _Evil()}}, f)
    with pytest.raises(pickle.UnpicklingError, match="only numpy arrays"):
        read_official_checkpoint(path)


def test_sample_z_fixed_noise_is_deterministic_and_onehot():
    adapter = DreamerV3Adapter(arch=default_arch(**SMALL))
    logits = torch.randn(3, 8, 4)
    noise = torch.rand(3, 8, 4)
    z1, z2 = adapter.sample_z(logits, noise=noise), adapter.sample_z(logits, noise=noise)
    assert torch.equal(z1, z2)
    assert torch.equal(z1.sum(-1), torch.ones(3, 8))
    assert set(z1.unique().tolist()) <= {0.0, 1.0}


def test_twohot_symmetric_sum_is_zero_for_uniform_logits():
    bins = twohot_bins(255)
    assert torch.allclose(bins, -bins.flip(0))
    assert twohot_pred(torch.zeros(4, 255), bins).abs().max().item() == 0.0
