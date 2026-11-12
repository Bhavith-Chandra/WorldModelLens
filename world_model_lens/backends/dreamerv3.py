"""DreamerV3 adapter: a faithful PyTorch port of the official implementation.

Source of truth: ``danijar/dreamerv3`` at commit ``e3f02248693a`` (2026-05-25),
files ``dreamerv3/rssm.py``, ``dreamerv3/agent.py``, ``embodied/jax/nets.py``,
``embodied/jax/heads.py`` and ``embodied/jax/outs.py``.

Design choices:

* Every module and parameter is named after the official parameter path, with
  ``/`` replaced by ``.`` (``dyn/dynin0/kernel`` -> ``dyn.dynin0.kernel``).
  Parameters keep the official layouts (Linear ``(in, out)``, BlockLinear
  ``(groups, in/groups, out/groups)``, Conv2D ``HWIO``), so loading an official
  checkpoint is an identity mapping and key coverage can be checked exactly.
* The forward pass follows the official code line by line, including the
  block-diagonal GRU and its gate grouping, RMSNorm (eps=1e-4), 1% unimix on the
  latent distribution, the symmetric two-hot decoding of reward and value, and
  the input scaling ``image / 255 - 0.5``.
* The port computes in float32. The official code computes in bfloat16, so
  outputs agree up to bfloat16 rounding.

Official checkpoints (``agent.pkl``) are read with a restricted unpickler that
only allows numpy array reconstruction. No JAX install is needed.
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.base_adapter import (
    AdapterConfig,
    BaseModelAdapter,
    WorldModelCapabilities,
)

OFFICIAL_COMMIT = "e3f02248693a"


# ─────────────────────────────────────────────────────────────────────────────
# Numerics shared with the official code
# ─────────────────────────────────────────────────────────────────────────────


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def twohot_bins(n: int = 255) -> torch.Tensor:
    """Bin locations of ``Head.symexp_twohot`` (heads.py)."""
    if n % 2 == 1:
        half = symexp(torch.linspace(-20, 0, (n - 1) // 2 + 1, dtype=torch.float64))
        bins = torch.cat([half, -half[:-1].flip(0)], 0)
    else:
        half = symexp(torch.linspace(-20, 0, n // 2, dtype=torch.float64))
        bins = torch.cat([half, -half.flip(0)], 0)
    return bins.float()


def twohot_pred(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value of a two-hot head with the official symmetric sum (outs.TwoHot.pred)."""
    probs = torch.softmax(logits.float(), -1)
    n = logits.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1, p2, p3 = probs[..., :m], probs[..., m : m + 1], probs[..., m + 1 :]
        b1, b2, b3 = bins[:m], bins[m : m + 1], bins[m + 1 :]
        return (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1)
    p1, p2 = probs[..., : n // 2], probs[..., n // 2 :]
    b1, b2 = bins[: n // 2], bins[n // 2 :]
    return ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1)


def unimix_logits(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Log-probabilities after mixing in ``unimix`` of the uniform (outs.Categorical)."""
    logits = logits.float()
    if not unimix:
        return logits
    probs = torch.softmax(logits, -1)
    probs = (1 - unimix) * probs + unimix / probs.shape[-1]
    return torch.log(probs)


def _act(name: str):
    if name == "none":
        return lambda x: x
    if name == "silu":
        return F.silu
    if name == "gelu":
        # jax.nn.gelu defaults to the tanh approximation.
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise NotImplementedError(name)


# ─────────────────────────────────────────────────────────────────────────────
# Layers (embodied/jax/nets.py)
# ─────────────────────────────────────────────────────────────────────────────


class DV3Linear(nn.Module):
    """``nets.Linear``: kernel ``(in, prod(units))``, output reshaped to ``units``."""

    def __init__(self, fan_in: int, units, bias: bool = True):
        super().__init__()
        self.units = (units,) if isinstance(units, int) else tuple(units)
        size = math.prod(self.units)
        self.kernel = nn.Parameter(torch.empty(fan_in, size))
        nn.init.trunc_normal_(self.kernel, std=1.1368 / math.sqrt(fan_in), a=-2.0, b=2.0)
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.kernel
        if self.bias is not None:
            x = x + self.bias
        return x.reshape(*x.shape[:-1], *self.units)


class DV3BlockLinear(nn.Module):
    """``nets.BlockLinear``: block-diagonal kernel ``(blocks, in/blocks, units/blocks)``."""

    def __init__(self, fan_in: int, units: int, blocks: int, bias: bool = True):
        super().__init__()
        assert fan_in % blocks == 0 and units % blocks == 0, (fan_in, units, blocks)
        self.blocks, self.units = blocks, units
        self.kernel = nn.Parameter(torch.empty(blocks, fan_in // blocks, units // blocks))
        nn.init.trunc_normal_(
            self.kernel, std=1.1368 / math.sqrt(fan_in // blocks * blocks), a=-2.0, b=2.0
        )
        self.bias = nn.Parameter(torch.zeros(units)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(*lead, self.blocks, x.shape[-1] // self.blocks)
        x = torch.einsum("...ki,kio->...ko", x, self.kernel).reshape(*lead, self.units)
        if self.bias is not None:
            x = x + self.bias
        return x


class DV3Conv2D(nn.Module):
    """``nets.Conv2D`` with stride 1 and SAME padding; kernel stored as HWIO, data NHWC."""

    def __init__(self, cin: int, depth: int, kernel: int = 5, bias: bool = True):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(kernel, kernel, cin, depth))
        nn.init.trunc_normal_(
            self.kernel, std=1.1368 / math.sqrt(kernel * kernel * cin), a=-2.0, b=2.0
        )
        self.bias = nn.Parameter(torch.zeros(depth)) if bias else None
        self.pad = kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.kernel.permute(3, 2, 0, 1)  # HWIO -> OIHW
        y = F.conv2d(x.permute(0, 3, 1, 2), w, self.bias, padding=self.pad)
        return y.permute(0, 2, 3, 1)


class DV3Norm(nn.Module):
    """``nets.Norm('rms')`` over the last axis, computed in float32."""

    def __init__(self, size: int, eps: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean2 = x.square().mean(-1, keepdim=True)
        return (x * (torch.rsqrt(mean2 + self.eps) * self.scale)).to(dtype)


class DV3MLP(nn.Module):
    """``nets.MLP``: repeated Linear -> Norm -> act."""

    def __init__(self, fan_in: int, layers: int, units: int, act: str = "silu"):
        super().__init__()
        self.layers = layers
        self._act = _act(act)
        for i in range(layers):
            setattr(self, f"linear{i}", DV3Linear(fan_in if i == 0 else units, units))
            setattr(self, f"norm{i}", DV3Norm(units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.layers):
            x = self._act(getattr(self, f"norm{i}")(getattr(self, f"linear{i}")(x)))
        return x


class DV3Head(nn.Module):
    """One ``heads.Head`` output. The submodule name matches the official one."""

    def __init__(self, fan_in: int, kind: str, size: int = 1, bins: int = 255):
        super().__init__()
        self.kind = kind
        if kind == "symexp_twohot":
            self.logits = DV3Linear(fan_in, bins)
            self.register_buffer("bins", twohot_bins(bins), persistent=False)
        elif kind == "binary":
            self.logit = DV3Linear(fan_in, 1)
        elif kind == "categorical":
            self.logits = DV3Linear(fan_in, size)
        elif kind == "bounded_normal":
            self.mean = DV3Linear(fan_in, size)
            self.stddev = DV3Linear(fan_in, size)
        elif kind == "symlog_mse":
            self.pred = DV3Linear(fan_in, size)
        else:
            raise NotImplementedError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "symexp_twohot":
            return self.logits(x)
        if self.kind == "binary":
            return self.logit(x)
        if self.kind == "categorical":
            return self.logits(x)
        if self.kind == "symlog_mse":
            return self.pred(x)
        return torch.cat([self.mean(x), self.stddev(x)], -1)


class DV3MLPHead(nn.Module):
    """``heads.MLPHead``: ``mlp`` then ``head`` (a ``DictHead`` when ``key`` is set)."""

    def __init__(
        self,
        fan_in: int,
        layers: int,
        units: int,
        kind: str,
        size: int = 1,
        key: Optional[str] = None,
        act: str = "silu",
        bins: int = 255,
        minstd: float = 0.1,
        maxstd: float = 1.0,
    ):
        super().__init__()
        self.mlp = DV3MLP(fan_in, layers, units, act)
        out = DV3Head(units, kind, size, bins)
        self.key = key
        self.head = nn.ModuleDict({key: out}) if key else out
        self.minstd, self.maxstd = minstd, maxstd

    @property
    def out(self) -> DV3Head:
        return cast(DV3Head, self.head[self.key] if self.key else self.head)  # type: ignore[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(x))


# ─────────────────────────────────────────────────────────────────────────────
# Encoder, RSSM, decoder (dreamerv3/rssm.py)
# ─────────────────────────────────────────────────────────────────────────────


class DV3Encoder(nn.Module):
    def __init__(self, arch: Dict[str, Any]):
        super().__init__()
        self.arch = arch
        self._act = _act(arch["act"])
        self.vec_dim = arch["vec_dim"]
        self.img_shape = arch["img_shape"]  # (H, W, C) or None
        units, K = arch["units"], arch["kernel"]
        for i in range(arch["enc_layers"] if self.vec_dim else 0):
            setattr(self, f"mlp{i}", DV3Linear(self.vec_dim if i == 0 else units, units))
            setattr(self, f"mlp{i}norm", DV3Norm(units))
        self.depths = tuple(arch["depth"] * m for m in arch["mults"]) if self.img_shape else ()
        cin = self.img_shape[-1] if self.img_shape else 0
        for i, depth in enumerate(self.depths):
            setattr(self, f"cnn{i}", DV3Conv2D(cin, depth, K))
            setattr(self, f"cnn{i}norm", DV3Norm(depth))
            cin = depth

    @property
    def out_dim(self) -> int:
        dim = self.arch["units"] if self.vec_dim else 0
        if self.img_shape:
            r = self.img_shape[0] // 2 ** len(self.depths)
            dim += r * r * self.depths[-1]
        return dim

    def to_nhwc(self, obs: torch.Tensor) -> torch.Tensor:
        """Accept ``[B, H, W, C]`` or ``[B, C, H, W]`` images with values in [0, 255]."""
        H, W, C = self.img_shape
        if obs.shape[-3:] == (H, W, C):
            return obs
        if obs.shape[-3:] == (C, H, W):
            return obs.movedim(-3, -1)
        raise ValueError(f"expected image [..., {H}, {W}, {C}] or [..., {C}, {H}, {W}], got {tuple(obs.shape)}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        outs = []
        if self.vec_dim:
            x = symlog(obs.float()) if self.arch["symlog"] else obs.float()
            for i in range(self.arch["enc_layers"]):
                x = self._act(getattr(self, f"mlp{i}norm")(getattr(self, f"mlp{i}")(x)))
            outs.append(x)
        if self.img_shape:
            x = self.to_nhwc(obs).float() / 255.0 - 0.5
            lead = x.shape[:-3]
            x = x.reshape(-1, *x.shape[-3:])
            for i in range(len(self.depths)):
                x = getattr(self, f"cnn{i}")(x)
                B, H, W, C = x.shape
                x = x.reshape(B, H // 2, 2, W // 2, 2, C).amax(dim=(2, 4))
                x = self._act(getattr(self, f"cnn{i}norm")(x))
            outs.append(x.reshape(*lead, -1))
        return torch.cat(outs, -1)


class DV3RSSM(nn.Module):
    def __init__(self, arch: Dict[str, Any], tokens: int):
        super().__init__()
        self.arch = arch
        D, Hd, S, C = arch["deter"], arch["hidden"], arch["stoch"], arch["classes"]
        g, A = arch["blocks"], arch["action_dim"]
        self.D, self.Hd, self.S, self.C, self.g = D, Hd, S, C, g
        self._act = _act(arch["act"])
        self.dynin0, self.dynin0norm = DV3Linear(D, Hd), DV3Norm(Hd)
        self.dynin1, self.dynin1norm = DV3Linear(S * C, Hd), DV3Norm(Hd)
        self.dynin2, self.dynin2norm = DV3Linear(A, Hd), DV3Norm(Hd)
        fan = D + 3 * Hd * g
        for i in range(arch["dynlayers"]):
            setattr(self, f"dynhid{i}", DV3BlockLinear(fan if i == 0 else D, D, g))
            setattr(self, f"dynhid{i}norm", DV3Norm(D))
        self.dyngru = DV3BlockLinear(D if arch["dynlayers"] else fan, 3 * D, g)
        for i in range(arch["imglayers"]):
            setattr(self, f"prior{i}", DV3Linear(D if i == 0 else Hd, Hd))
            setattr(self, f"prior{i}norm", DV3Norm(Hd))
        self.priorlogit = DV3Linear(Hd, (S, C))
        obs_in = tokens if arch["absolute"] else D + tokens
        for i in range(arch["obslayers"]):
            setattr(self, f"obs{i}", DV3Linear(obs_in if i == 0 else Hd, Hd))
            setattr(self, f"obs{i}norm", DV3Norm(Hd))
        self.obslogit = DV3Linear(Hd, (S, C))

    def core(self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``RSSM._core``: one block-GRU step ``(deter, stoch, action) -> deter``."""
        g, D = self.g, self.D
        stoch = stoch.reshape(*deter.shape[:-1], -1)
        action = action / torch.clamp(action.abs(), min=1.0).detach()
        x0 = self._act(self.dynin0norm(self.dynin0(deter)))
        x1 = self._act(self.dynin1norm(self.dynin1(stoch)))
        x2 = self._act(self.dynin2norm(self.dynin2(action)))
        lead = deter.shape[:-1]
        x = torch.cat([x0, x1, x2], -1).unsqueeze(-2).expand(*lead, g, 3 * self.Hd)
        x = torch.cat([deter.reshape(*lead, g, D // g), x], -1).reshape(*lead, -1)
        for i in range(self.arch["dynlayers"]):
            x = self._act(getattr(self, f"dynhid{i}norm")(getattr(self, f"dynhid{i}")(x)))
        x = self.dyngru(x).reshape(*lead, g, 3 * D // g)
        reset, cand, update = [t.reshape(*lead, D) for t in x.chunk(3, -1)]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter

    def prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        x = deter
        for i in range(self.arch["imglayers"]):
            x = self._act(getattr(self, f"prior{i}norm")(getattr(self, f"prior{i}")(x)))
        return self.priorlogit(x)

    def post_logits(self, deter: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens if self.arch["absolute"] else torch.cat([deter, tokens], -1)
        for i in range(self.arch["obslayers"]):
            x = self._act(getattr(self, f"obs{i}norm")(getattr(self, f"obs{i}")(x)))
        return self.obslogit(x)


class DV3Decoder(nn.Module):
    def __init__(self, arch: Dict[str, Any]):
        super().__init__()
        self.arch = arch
        self._act = _act(arch["act"])
        D, Z, units, K = arch["deter"], arch["stoch"] * arch["classes"], arch["units"], arch["kernel"]
        self.vec_dim = arch["vec_dim"]
        self.img_shape = arch["img_shape"]
        if self.vec_dim:
            self.mlp = DV3MLP(Z + D, arch["dec_layers"], units, arch["act"])
            self.vec = nn.ModuleDict({"obs": DV3Head(units, "symlog_mse", self.vec_dim)})
        if self.img_shape:
            self.depths = tuple(arch["depth"] * m for m in arch["mults"])
            H, W, Cimg = self.img_shape
            self.minres = (H // 2 ** len(self.depths), W // 2 ** len(self.depths))
            shape = (*self.minres, self.depths[-1])
            u, g = math.prod(shape), arch["bspace"]
            self.shape, self.g = shape, g
            self.sp0 = DV3BlockLinear(D, u, g)
            self.sp1, self.sp1norm = DV3Linear(Z, 2 * units), DV3Norm(2 * units)
            self.sp2 = DV3Linear(2 * units, shape)
            self.spnorm = DV3Norm(self.depths[-1])
            cin = self.depths[-1]
            for i in reversed(range(len(self.depths) - 1)):
                setattr(self, f"conv{i}", DV3Conv2D(cin, self.depths[i], K))
                setattr(self, f"conv{i}norm", DV3Norm(self.depths[i]))
                cin = self.depths[i]
            self.imgout = DV3Conv2D(cin, Cimg, K)

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> Dict[str, torch.Tensor]:
        lead = deter.shape[:-1]
        stoch = stoch.reshape(*lead, -1)
        out = {}
        if self.vec_dim:
            out["obs"] = self.vec["obs"](self.mlp(torch.cat([stoch, deter], -1)))
        if self.img_shape:
            h, w = self.minres
            c = self.shape[-1] // self.g
            x0 = self.sp0(deter.reshape(-1, deter.shape[-1]))
            x0 = x0.reshape(-1, self.g, h, w, c).permute(0, 2, 3, 1, 4).reshape(-1, h, w, self.g * c)
            x1 = self._act(self.sp1norm(self.sp1(stoch.reshape(-1, stoch.shape[-1]))))
            x1 = self.sp2(x1)
            x = self._act(self.spnorm(x0 + x1))
            for i in reversed(range(len(self.depths) - 1)):
                x = x.repeat_interleave(2, 1).repeat_interleave(2, 2)
                x = self._act(getattr(self, f"conv{i}norm")(getattr(self, f"conv{i}")(x)))
            x = x.repeat_interleave(2, 1).repeat_interleave(2, 2)
            x = torch.sigmoid(self.imgout(x))
            out["image"] = x.reshape(*lead, *x.shape[1:])  # NHWC in [0, 1]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Architecture description and inference from parameter shapes
# ─────────────────────────────────────────────────────────────────────────────


def default_arch(**overrides: Any) -> Dict[str, Any]:
    arch = dict(
        deter=512, hidden=64, stoch=32, classes=4, blocks=8, action_dim=18,
        discrete_actions=True, act="silu", unimix=0.01, absolute=False,
        imglayers=2, obslayers=1, dynlayers=1,
        img_shape=(64, 64, 3), vec_dim=0, depth=4, mults=(2, 3, 4, 4), kernel=5,
        enc_layers=3, dec_layers=3, symlog=True, bspace=8, units=64,
        rew_layers=1, con_layers=1, pol_layers=3, val_layers=3, bins=255,
        minstd=0.1, maxstd=1.0, slowval=True,
    )
    arch.update(overrides)
    return arch


def arch_from_adapter_config(cfg: Any, **overrides: Any) -> Dict[str, Any]:
    """Map the generic ``AdapterConfig`` onto the official size presets.

    All official presets share the ratios hidden = units = deter / 8 and
    depth = units / 16, so a config only needs ``d_h`` (deter) plus the latent
    and action sizes.
    """
    deter = int(getattr(cfg, "d_h", 512))
    hidden = max(8, deter // 8)
    encoder_type = getattr(cfg, "encoder_type", "cnn")
    d_obs = int(getattr(cfg, "d_obs", 0) or 0)
    arch = default_arch(
        deter=deter, hidden=hidden, units=hidden, depth=max(2, hidden // 16),
        stoch=int(getattr(cfg, "n_cat", 32)), classes=int(getattr(cfg, "n_cls", 32)),
        blocks=8 if deter % 8 == 0 else 1,
        action_dim=max(1, int(getattr(cfg, "d_action", 0) or 0)),
        img_shape=None if encoder_type != "cnn" else (64, 64, 3),
        vec_dim=d_obs if encoder_type != "cnn" else 0,
    )
    if arch["img_shape"] is None and not arch["vec_dim"]:
        arch["vec_dim"] = 1
    arch.update(overrides)
    return arch


def _count(params: Mapping[str, Any], pattern: str) -> int:
    rx = re.compile(pattern)
    return len({m.group(1) for k in params for m in [rx.fullmatch(k)] if m})


def arch_from_params(params: Mapping[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Infer the architecture from official parameter names and shapes.

    ``stoch`` vs ``classes`` cannot be told apart from shapes (only their
    product appears), so ``stoch`` defaults to 32, as in every official size
    preset; pass ``stoch=`` to override.
    """
    shp = {k: tuple(np.shape(v)) for k, v in params.items()}
    D, Hd = shp["dyn/dynin0/kernel"]
    Z = shp["dyn/obslogit/kernel"][1]
    stoch = int(overrides.pop("stoch", 32))
    assert Z % stoch == 0, (Z, stoch)
    arch = default_arch(
        deter=D, hidden=Hd, stoch=stoch, classes=Z // stoch,
        blocks=shp["dyn/dynhid0/kernel"][0] if "dyn/dynhid0/kernel" in shp else shp["dyn/dyngru/kernel"][0],
        action_dim=shp["dyn/dynin2/kernel"][0],
        dynlayers=_count(shp, r"dyn/dynhid(\d+)/kernel"),
        imglayers=_count(shp, r"dyn/prior(\d+)/kernel"),
        obslayers=_count(shp, r"dyn/obs(\d+)/kernel"),
    )
    ncnn = _count(shp, r"enc/cnn(\d+)/kernel")
    nmlp = _count(shp, r"enc/mlp(\d+)/kernel")
    arch["enc_layers"] = nmlp
    arch["vec_dim"] = shp["enc/mlp0/kernel"][0] if nmlp else 0
    obs_in = shp["dyn/obs0/kernel"][0]
    tokens = obs_in - (0 if arch["absolute"] else D)
    if ncnn:
        k, _, cin, _ = shp["enc/cnn0/kernel"]
        depths = [shp[f"enc/cnn{i}/kernel"][3] for i in range(ncnn)]
        base = math.gcd(*depths)
        arch.update(kernel=k, depth=base, mults=tuple(d // base for d in depths))
        img_tokens = tokens - (shp["enc/mlp0/kernel"][1] if nmlp else 0)
        minres = int(round(math.sqrt(img_tokens / depths[-1])))
        assert minres * minres * depths[-1] == img_tokens, (img_tokens, depths)
        res = minres * 2**ncnn
        arch["img_shape"] = (res, res, cin)
        arch["bspace"] = shp["dec/sp0/kernel"][0] if "dec/sp0/kernel" in shp else 0
    else:
        arch["img_shape"] = None
    if "rew/mlp/linear0/kernel" in shp:
        arch["units"] = shp["rew/mlp/linear0/kernel"][1]
    elif nmlp:
        arch["units"] = shp["enc/mlp0/kernel"][1]
    arch["dec_layers"] = _count(shp, r"dec/mlp/linear(\d+)/kernel")
    for name in ("rew", "con", "pol", "val"):
        arch[f"{name}_layers"] = _count(shp, rf"{name}/mlp/linear(\d+)/kernel")
    if "rew/head/logits/kernel" in shp:
        arch["bins"] = shp["rew/head/logits/kernel"][1]
    arch["discrete_actions"] = "pol/head/action/logits/kernel" in shp
    arch["slowval"] = "slowval/mlp/linear0/kernel" in shp
    arch.update(overrides)
    return arch


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint reading
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"Refusing to load '{module}.{name}' from checkpoint: only numpy arrays are allowed."
            )
        return getattr(__import__(module, fromlist=[name]), name)


def read_official_checkpoint(path: str | Path) -> Dict[str, np.ndarray]:
    """Read ``agent.pkl`` written by danijar/dreamerv3 (``{'params': {...}, 'counters': ...}``)."""
    with open(path, "rb") as f:
        data = _NumpyOnlyUnpickler(f).load()
    if not isinstance(data, dict) or "params" not in data:
        raise ValueError(f"{path} is not an official DreamerV3 agent checkpoint")
    return dict(data["params"])


NETWORK_PREFIXES = ("enc/", "dyn/", "dec/", "rew/", "con/", "pol/", "val/", "slowval/")


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class DreamerV3Adapter(BaseModelAdapter):
    """Faithful DreamerV3 world model + actor-critic.

    State convention: ``h`` is the deterministic state ``deter`` with shape
    ``[B, deter]``; ``z`` is the stochastic state ``stoch``, a one-hot tensor of
    shape ``[B, stoch, classes]`` (flat ``[B, stoch * classes]`` is accepted too).

    One environment step of the official ``RSSM._observe`` is::

        h_t = adapter.transition(h_{t-1}, z_{t-1}, a_{t-1})   # zeros on is_first
        post_logits, tokens = adapter.encode(obs_t, h_t)
        z_t = adapter.sample_z(post_logits)

    and one imagination step (``RSSM.imagine``) is::

        a_t = sample(adapter.actor_forward(h_t, z_t))
        h_{t+1} = adapter.transition(h_t, z_t, a_t)
        z_{t+1} = adapter.sample_z(adapter.dynamics(h_{t+1}))
    """

    def __init__(self, config: Any = None, arch: Optional[Dict[str, Any]] = None, **overrides: Any):
        if arch is None:
            arch = arch_from_adapter_config(config, **overrides) if config is not None else default_arch(**overrides)
        if config is None:
            config = AdapterConfig(
                d_h=arch["deter"], n_cat=arch["stoch"], n_cls=arch["classes"],
                d_action=arch["action_dim"], d_obs=arch["vec_dim"],
                encoder_type="cnn" if arch["img_shape"] else "mlp",
                name="dreamerv3", backend="dreamerv3",
            )
        super().__init__(config)
        self.config = config
        self.arch = arch
        S, C, D = arch["stoch"], arch["classes"], arch["deter"]
        F_in = D + S * C
        A = arch["action_dim"]

        self.enc = DV3Encoder(arch)
        self.dyn = DV3RSSM(arch, self.enc.out_dim)
        self.dec = DV3Decoder(arch)
        units, act = arch["units"], arch["act"]
        self.rew = DV3MLPHead(F_in, arch["rew_layers"], units, "symexp_twohot", act=act, bins=arch["bins"])
        self.con = DV3MLPHead(F_in, arch["con_layers"], units, "binary", act=act)
        pol_kind = "categorical" if arch["discrete_actions"] else "bounded_normal"
        self.pol = DV3MLPHead(
            F_in, arch["pol_layers"], units, pol_kind, size=A, key="action", act=act,
            minstd=arch["minstd"], maxstd=arch["maxstd"],
        )
        self.val = DV3MLPHead(F_in, arch["val_layers"], units, "symexp_twohot", act=act, bins=arch["bins"])
        if arch["slowval"]:
            self.slowval = DV3MLPHead(F_in, arch["val_layers"], units, "symexp_twohot", act=act, bins=arch["bins"])

        # Shapes only reveal stoch * classes, so record the split in the state dict.
        self.register_buffer("stoch_classes", torch.tensor([S, C]))
        self.extras: Dict[str, Any] = {}
        self.load_report: Optional[Dict[str, Any]] = None
        self._capabilities = WorldModelCapabilities(
            has_decoder=True, has_reward_head=True, has_continue_head=True,
            has_actor=True, has_critic=True, uses_actions=True, is_rl_trained=True,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def deter_dim(self) -> int:
        return self.arch["deter"]

    @property
    def stoch_shape(self) -> Tuple[int, int]:
        return (self.arch["stoch"], self.arch["classes"])

    def _z(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Return ``z`` flattened to ``[..., stoch * classes]`` with the batch dims of ``h``."""
        return z.reshape(*h.shape[:-1], -1)

    def feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``agent.feat2tensor``: ``concat(deter, stoch.flatten())``."""
        return torch.cat([h, self._z(z, h)], -1)

    def action_onehot(self, action: torch.Tensor) -> torch.Tensor:
        """Integer actions ``[B]`` -> one-hot ``[B, A]``; float vectors pass through."""
        if not torch.is_floating_point(action):
            return F.one_hot(action.long(), self.arch["action_dim"]).float()
        return action.float()

    # ── required API ─────────────────────────────────────────────────────────

    def initial_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``RSSM.initial``: zeros for both deter and stoch."""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, *self.stoch_shape, device=device)
        return h, z

    def transition(
        self, h: torch.Tensor, z: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``RSSM._core``: ``h_{t+1} = f(h_t, z_t, a_t)``."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if action is None:
            action = torch.zeros(*h.shape[:-1], self.arch["action_dim"], device=h.device)
        action = self.action_onehot(action)
        if action.dim() < h.dim():
            action = action.unsqueeze(0)
        return self.dyn.core(h, self._z(z, h), action)

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior logits ``[B, stoch, classes]`` from ``(h_t, obs_t)``, plus the encoder tokens.

        ``h_prev`` must be the deterministic state *after* this step's
        transition, as in ``RSSM._observe``. Images are ``[B, H, W, C]`` or
        ``[B, C, H, W]`` with pixel values in [0, 255].
        """
        tokens = self.enc(obs)
        if h_prev.dim() < tokens.dim():
            h_prev = h_prev.expand(*tokens.shape[:-1], h_prev.shape[-1])
        return self.dyn.post_logits(h_prev, tokens), tokens

    def dynamics(self, h: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Prior logits ``[B, stoch, classes]`` from ``h`` alone (``RSSM._prior``)."""
        return self.dyn.prior_logits(h)

    def latent_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        """Log-probabilities of the latent distribution, with the 1% unimix applied."""
        return unimix_logits(logits, self.arch["unimix"])

    def sample_z(
        self,
        logits_or_repr: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample one-hot ``z`` from prior or posterior logits (``outs.OneHot``).

        Applies the official unimix, then Gumbel-max sampling. Pass ``noise``
        (uniform samples in (0, 1) with the logits' shape) to reuse the same
        randomness across runs, or a ``generator``. ``sample=False`` returns
        the mode. Gradients flow straight through, as in the official code.
        """
        logp = self.latent_logprobs(logits_or_repr)
        if sample:
            if noise is None:
                noise = torch.rand(logp.shape, generator=generator, device=logp.device)
            u = noise.clamp(torch.finfo(torch.float32).tiny, 1.0 - 1e-7)
            idx = (logp / temperature - torch.log(-torch.log(u))).argmax(-1)
        else:
            idx = logp.argmax(-1)
        hard = F.one_hot(idx, logp.shape[-1]).float()
        probs = logp.exp()
        return hard + (probs - probs.detach())

    # ── heads ────────────────────────────────────────────────────────────────

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Image reconstruction ``[B, H, W, C]`` in [0, 1] (or the vector prediction)."""
        out = self.dec(h, self._z(z, h))
        return out["image"] if "image" in out else out["obs"]

    def reward_logits(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.rew(self.feat(h, z))

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predicted reward ``[B, 1]`` (two-hot expectation, symmetric sum)."""
        return twohot_pred(self.reward_logits(h, z), cast(torch.Tensor, self.rew.out.bins)).unsqueeze(-1)

    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Continuation logit ``[B, 1]``; ``sigmoid`` gives the probability."""
        return self.con(self.feat(h, z))

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Action logits ``[B, A]`` (no unimix, as in ``Head.categorical``).

        For continuous actions returns ``concat(mean, std)`` of the bounded normal.
        """
        out = self.pol(self.feat(h, z))
        if self.arch["discrete_actions"]:
            return out
        A = self.arch["action_dim"]
        mean, std = out[..., :A], out[..., A:]
        lo, hi = self.pol.minstd, self.pol.maxstd
        return torch.cat([torch.tanh(mean), (hi - lo) * torch.sigmoid(std + 2.0) + lo], -1)

    def value_logits(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.val(self.feat(h, z))

    def critic_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predicted value ``[B, 1]`` in return units (``val`` head)."""
        return twohot_pred(self.value_logits(h, z), cast(torch.Tensor, self.val.out.bins)).unsqueeze(-1)

    # ── loading ──────────────────────────────────────────────────────────────

    def load_official_params(self, params: Mapping[str, Any], strict: bool = True) -> Dict[str, Any]:
        """Copy official parameters in, checking that every key maps exactly once.

        Network keys (``enc/ dyn/ dec/ rew/ con/ pol/ val/ slowval/``) must
        match the adapter's parameters one-to-one with equal shapes. Optimizer
        state (``opt/``) is skipped; other entries (``retnorm/``,
        ``slowval_count``) are kept in ``self.extras``.
        """
        own = dict(self.named_parameters())
        net = {k: v for k, v in params.items() if k.startswith(NETWORK_PREFIXES)}
        skipped = sorted(k for k in params if k.startswith("opt/"))
        extras = {k: np.asarray(v) for k, v in params.items() if k not in net and not k.startswith("opt/")}
        mapped, shape_mismatch, unexpected = [], [], []
        with torch.no_grad():
            for key, value in net.items():
                name = key.replace("/", ".")
                if name not in own:
                    unexpected.append(key)
                    continue
                value = torch.as_tensor(np.asarray(value, dtype=np.float32))
                if tuple(value.shape) != tuple(own[name].shape):
                    shape_mismatch.append((key, tuple(value.shape), tuple(own[name].shape)))
                    continue
                own[name].copy_(value)
                mapped.append(key)
        missing = sorted(set(own) - {k.replace("/", ".") for k in mapped})
        report = dict(
            mapped=len(mapped), total_adapter_params=len(own),
            missing=missing, unexpected=sorted(unexpected), shape_mismatch=shape_mismatch,
            skipped_optimizer=len(skipped), extras=sorted(extras),
        )
        self.extras = extras
        self.load_report = report
        if strict and (missing or unexpected or shape_mismatch):
            raise RuntimeError(f"DreamerV3 checkpoint does not match the adapter: {report}")
        return report

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, config: Optional[AdapterConfig] = None, strict: bool = True, **overrides: Any
    ) -> "DreamerV3Adapter":
        """Load an official ``agent.pkl`` or a state dict saved from this adapter.

        The architecture is inferred from parameter shapes. ``overrides`` (for
        example ``stoch=32`` or ``act='silu'``) fix values that shapes cannot
        determine.
        """
        path = Path(path)
        if path.suffix == ".pkl":
            params = read_official_checkpoint(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if "stoch_classes" in state:
                overrides.setdefault("stoch", int(state.pop("stoch_classes")[0]))
            params = {k.replace(".", "/"): v.numpy() for k, v in state.items()}
        net = {k: v for k, v in params.items() if k.startswith(NETWORK_PREFIXES)}
        arch = arch_from_params(net, **overrides)
        adapter = cls(config, arch=arch)
        adapter.load_official_params(params, strict=strict)
        adapter.eval()
        return adapter

    @classmethod
    def infer_config(cls, state_dict: Mapping[str, Any]) -> Dict[str, Any]:
        """Architecture inferred from a state dict (official ``/`` or torch ``.`` names)."""
        params = {k.replace(".", "/"): v for k, v in state_dict.items()}
        return arch_from_params({k: v for k, v in params.items() if k.startswith(NETWORK_PREFIXES)})

    def to(self, *args: Any, **kwargs: Any) -> "DreamerV3Adapter":  # type: ignore[override]
        """Standard ``nn.Module.to`` (accepts a positional device or dtype)."""
        return cast("DreamerV3Adapter", nn.Module.to(self, *args, **kwargs))

    def list_hookable_points(self) -> List[str]:
        """Return semantic hook names plus concrete module paths."""
        points = list(self.hook_point_names)
        points.extend(name for name, _ in self.named_modules() if name)
        return list(dict.fromkeys(points))
