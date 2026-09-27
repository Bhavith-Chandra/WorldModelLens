"""V-JEPA (Video Joint-Embedding Predictive Architecture) Adapter.

Implements 3D spatiotemporal tubelet embedding, 3D Vision Transformer context encoder,
and 3D Predictor for video world model evaluation in World Model Lens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional, Tuple, List, Dict, Any, Union, cast

from world_model_lens.backends.base_adapter import BaseModelAdapter, WorldModelCapabilities
from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooked_root import HookedRootModule
from world_model_lens.core.hooks import HookContext, HookPoint, HookRegistry


class TubeletEmbed(nn.Module):
    """Spatiotemporal Video to 3D Tubelet Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.grid_t = num_frames // tubelet_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.n_patches = self.grid_t * self.grid_h * self.grid_w

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape [B, C, T, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(2) # add T dimension if 2D image passed
        # Conv3D projection: [B, C, T, H, W] -> [B, embed_dim, grid_t, grid_h, grid_w]
        x = self.proj(x)
        # Flatten spatiotemporal dimensions: [B, embed_dim, N] -> [B, N, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention3D(nn.Module):
    """3D Spatiotemporal Attention Module with Hook Registration."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.hook_query = nn.Identity()
        self.hook_key = nn.Identity()
        self.hook_value = nn.Identity()
        self.hook_pattern = nn.Identity()
        self.hook_z = nn.Identity()

        self.last_attn_weights = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.hook_query(q)
        k = self.hook_key(k)
        v = self.hook_value(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        self.last_attn_weights = attn.detach()
        attn = self.hook_pattern(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.hook_z(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block3D(nn.Module):
    """3D Spatiotemporal Transformer Block with Activation Hooks."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        self.hook_resid_mid = nn.Identity()
        self.hook_mlp_in = nn.Identity()
        self.hook_mlp_out = nn.Identity()
        self.hook_resid_post = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = self.hook_resid_mid(x)

        mlp_in = self.norm2(x)
        mlp_in = self.hook_mlp_in(mlp_in)
        mlp_out = self.mlp(mlp_in)
        mlp_out = self.hook_mlp_out(mlp_out)

        x = x + mlp_out
        x = self.hook_resid_post(x)
        return x


class VJEPAEncoder(nn.Module):
    """3D Vision Transformer Context Encoder for V-JEPA."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16
    ):
        super().__init__()
        self.patch_embed = TubeletEmbed(img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block3D(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.hook_resid_pre = nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, patch_ids: Optional[Any] = None) -> torch.Tensor:
        x = self.patch_embed(x)
        if patch_ids is not None:
            if isinstance(patch_ids, (list, tuple)):
                patch_ids = torch.tensor(patch_ids, device=x.device)
            if patch_ids.dim() == 1:
                pos_embed = self.pos_embed[:, patch_ids, :]
                x = x[:, patch_ids, :]
            else:
                B, N_full, C = x.shape
                pos_embed = self.pos_embed.expand(B, -1, -1)
                x = torch.gather(x, 1, patch_ids.unsqueeze(-1).expand(-1, -1, C))
                pos_embed = torch.gather(pos_embed, 1, patch_ids.unsqueeze(-1).expand(-1, -1, C))
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        return self.forward_blocks(x)

    def forward_blocks(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        return x


class VJEPAPredictor(HookedRootModule):
    """3D Predictor Transformer for V-JEPA."""

    def __init__(
        self,
        encoder_embed_dim: int = 1024,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 16,
        num_patches: int = 1568
    ):
        super().__init__()
        self.prefix = ""
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        # Meta's V-JEPA checkpoint stores one token for each mask configuration.
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for _ in range(2)]
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )

        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block3D(dim=predictor_embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.hook_resid_pre = nn.Identity()
        self.predictor_project_back = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        # The official zero-initialized mask tokens and fixed positional values
        # are supplied by the checkpoint when loading pretrained weights.

    def forward(
        self, context_latents: torch.Tensor, context_ids: Any,
        target_ids: Any, mask_index: int = 1,
    ) -> torch.Tensor:
        B = context_latents.shape[0]
        context_ids = torch.as_tensor(context_ids, device=context_latents.device, dtype=torch.long)
        target_ids = torch.as_tensor(target_ids, device=context_latents.device, dtype=torch.long)
        if context_ids.ndim == 1:
            context_ids = context_ids.expand(B, -1)
        if target_ids.ndim == 1:
            target_ids = target_ids.expand(B, -1)
        if context_ids.ndim != 2 or target_ids.ndim != 2 or context_ids.shape[0] != B or target_ids.shape[0] != B:
            raise ValueError("Context and target masks must have shape [B, N] or [N].")
        if context_latents.shape[1] != context_ids.shape[1]:
            raise ValueError("Context latent count must match context mask length.")
        context_pos = self.pos_embed.expand(B, -1, -1).gather(
            1, context_ids.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        target_pos = self.pos_embed.expand(B, -1, -1).gather(
            1, target_ids.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        context_inputs = self.predictor_embed(context_latents) + context_pos
        target_tokens = self.mask_tokens[mask_index % len(self.mask_tokens)].expand(B, target_ids.shape[1], -1)
        target_inputs = target_tokens + target_pos

        x = torch.cat([context_inputs, target_inputs], dim=1)
        x = self.hook_resid_pre(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if hasattr(self, "hooks") and self.hooks is not None:
                ctx = HookContext(timestep=getattr(self, "current_timestep", 0), component=f"predictor.blocks.{i}")
                x = self.hooks.apply(f"predictor.blocks.{i}.hook_resid_post", getattr(self, "current_timestep", 0), x, ctx)

        x = self.norm(x)

        target_preds = x[:, context_ids.shape[1]:, :]
        target_preds = self.predictor_project_back(target_preds)
        return target_preds

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(*args, **kwargs)

@register(
    "vjepa", WorldModelFamily.JEPA, "Video Joint-Embedding Predictive Architecture",
    supports_rl=False, supports_video=True,
)
class VJEPAAdapter(BaseModelAdapter, HookedRootModule):
    """Hookable V-JEPA v1 model with strict Meta checkpoint loading."""

    def __init__(self, config: Optional[WorldModelConfig] = None):
        if config is None:
            config = WorldModelConfig.vjepa_vitl16()
        BaseModelAdapter.__init__(self, config)
        HookedRootModule.__init__(self)
        self.config = config

        img_size = getattr(config, "img_size", 224)
        patch_size = getattr(config, "patch_size", 16)
        num_frames = getattr(config, "num_frames", 16)
        tubelet_size = getattr(config, "tubelet_size", 2)
        embed_dim = config.d_embed
        depth = config.n_layers
        num_heads = config.n_heads

        predictor_embed_dim = getattr(config, "predictor_embed_dim", 384)
        predictor_depth = config.predictor_depth
        predictor_num_heads = config.predictor_heads

        grid_t = num_frames // tubelet_size
        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        num_patches = grid_t * grid_h * grid_w

        self.context_encoder = VJEPAEncoder(
            img_size=img_size, patch_size=patch_size, num_frames=num_frames, tubelet_size=tubelet_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = VJEPAPredictor(
            encoder_embed_dim=embed_dim, predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth, num_heads=predictor_num_heads, num_patches=num_patches
        )

        self.last_context_ids: Optional[List[int]] = None
        self.last_target_ids: Optional[List[int]] = None
        self.hooks = HookRegistry()
        self.current_timestep = 0

        self._capabilities = WorldModelCapabilities(
            has_decoder=False, has_reward_head=False, has_continue_head=False,
            has_actor=False, has_critic=False, uses_actions=False, is_rl_trained=False
        )

    def add_hook(self, hook_point: str, hook_fn: Any) -> HookPoint:
        hp = HookPoint(name=hook_point, fn=hook_fn)
        self.hooks.register(hp)
        return hp

    def remove_hook(self, handle_or_name: Any) -> None:
        if isinstance(handle_or_name, HookPoint):
            self.hooks.remove(handle_or_name)
        else:
            self.hooks.clear(str(handle_or_name))

    def encode(self, obs: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.context_encoder.hooks = self.hooks
        self.context_encoder.current_timestep = self.current_timestep
        if self.last_context_ids is not None:
            ctx_latents = self.context_encoder(obs, patch_ids=self.last_context_ids)
        else:
            ctx_latents = self.context_encoder(obs)
        return ctx_latents, ctx_latents

    def target_encode(self, obs: torch.Tensor) -> torch.Tensor:
        self.target_encoder.hooks = self.hooks
        self.target_encoder.current_timestep = self.current_timestep
        with torch.no_grad():
            return self.target_encoder(obs)

    def dynamics(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_patches = self.context_encoder.patch_embed.n_patches
        if self.last_context_ids is None or self.last_target_ids is None:
            ctx_ids = list(range(int(n_patches * 0.5)))
            tgt_ids = list(range(int(n_patches * 0.5), n_patches))
        else:
            ctx_ids = self.last_context_ids
            tgt_ids = self.last_target_ids

        if state.ndim != 3:
            raise ValueError(f"Expected context latents [B, N, D], got shape {tuple(state.shape)}")
        context_count = len(ctx_ids) if isinstance(ctx_ids, list) else ctx_ids.shape[-1]
        if state.shape[1] == n_patches:
            # encode() returns every token when no context mask was supplied.
            # Select the context tokens before passing them to the predictor.
            if isinstance(ctx_ids, torch.Tensor) and ctx_ids.ndim == 2:
                state = state.gather(1, ctx_ids.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
            else:
                state = state[:, ctx_ids, :]
        elif state.shape[1] != context_count:
            raise ValueError(
                f"Expected {context_count} context tokens or {n_patches} full tokens, "
                f"got {state.shape[1]}"
            )

        self.predictor.hooks = self.hooks
        self.predictor.current_timestep = self.current_timestep
        return self.predictor(state, ctx_ids, tgt_ids)

    def predict_masked(
        self, video: torch.Tensor, context_ids: torch.Tensor,
        target_ids: torch.Tensor, mask_index: int = 1,
    ) -> torch.Tensor:
        """Predict target tokens using explicit per-video patch indices."""
        context = self.context_encoder(video, patch_ids=context_ids)
        return self.predictor(context, context_ids, target_ids, mask_index=mask_index)

    @classmethod
    def from_checkpoint(
        cls, path: str, config: Optional[WorldModelConfig] = None
    ) -> "VJEPAAdapter":
        """Load Meta's encoder, EMA target encoder and predictor without partial weights."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("V-JEPA checkpoint must be a state dictionary.")
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        if all(name in checkpoint for name in ("encoder", "target_encoder", "predictor")):
            def map_keys(state: Dict[str, torch.Tensor], predictor: bool = False) -> Dict[str, torch.Tensor]:
                mapped: Dict[str, torch.Tensor] = {}
                for key, value in state.items():
                    for prefix in ("module.", "backbone."):
                        if key.startswith(prefix):
                            key = key[len(prefix):]
                    if predictor:
                        for original, replacement in (
                            ("predictor_blocks.", "blocks."),
                            ("predictor_norm.", "norm."),
                            ("predictor_proj.", "predictor_project_back."),
                            ("predictor_pos_embed", "pos_embed"),
                        ):
                            if key.startswith(original):
                                key = replacement + key[len(original):]
                                break
                    key = key.replace(".mlp.fc1.", ".mlp.0.")
                    key = key.replace(".mlp.fc2.", ".mlp.2.")
                    if key in mapped:
                        raise RuntimeError(f"Duplicate V-JEPA checkpoint key after mapping: {key}")
                    mapped[key] = value
                return mapped

            encoder = map_keys(checkpoint["encoder"])
            target = map_keys(checkpoint["target_encoder"])
            predictor = map_keys(checkpoint["predictor"], predictor=True)
            if config is None:
                config = WorldModelConfig.vjepa_vitl16()
                patch_weight = encoder.get("patch_embed.proj.weight")
                if not isinstance(patch_weight, torch.Tensor) or patch_weight.ndim != 5:
                    raise RuntimeError("V-JEPA checkpoint lacks a 3D patch embedding weight.")
                config.d_embed = int(patch_weight.shape[0])
                config.tubelet_size = int(patch_weight.shape[2])
                config.patch_size = int(patch_weight.shape[3])
                config.n_layers = sum(k.endswith(".norm1.weight") and k.startswith("blocks.") for k in encoder)
                pred_weight = predictor.get("predictor_embed.weight")
                if not isinstance(pred_weight, torch.Tensor):
                    raise RuntimeError("V-JEPA checkpoint lacks predictor_embed.weight.")
                config.predictor_embed_dim = int(pred_weight.shape[0])
                config.predictor_depth = sum(
                    k.endswith(".norm1.weight") and k.startswith("blocks.") for k in predictor
                )
            adapter = cls(config)
            adapter.context_encoder.load_state_dict(encoder, strict=True)
            adapter.target_encoder.load_state_dict(target, strict=True)
            adapter.predictor.load_state_dict(predictor, strict=True)
        else:
            # Project-native checkpoints use the adapter's own state-dict keys.
            if any(name in checkpoint for name in ("encoder", "target_encoder", "predictor")):
                raise RuntimeError("V-JEPA checkpoint must contain encoder, target_encoder and predictor.")
            adapter = cls(config)
            adapter.load_state_dict(checkpoint, strict=True)
        adapter.eval()
        return adapter
