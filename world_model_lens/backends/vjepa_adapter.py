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

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
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

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
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
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12
    ):
        super().__init__()
        self.patch_embed = TubeletEmbed(img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block3D(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
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
        encoder_embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        num_patches: int = 1568
    ):
        super().__init__()
        self.prefix = ""
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim))

        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block3D(dim=predictor_embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.hook_resid_pre = nn.Identity()
        self.predictor_project_back = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context_latents: torch.Tensor, context_ids: List[int], target_ids: List[int]) -> torch.Tensor:
        B = context_latents.shape[0]
        context_inputs = self.predictor_embed(context_latents) + self.pos_embed[:, context_ids, :]

        target_tokens = self.mask_token.expand(B, len(target_ids), -1)
        target_inputs = target_tokens + self.pos_embed[:, target_ids, :]

        x = torch.cat([context_inputs, target_inputs], dim=1)
        x = self.hook_resid_pre(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if hasattr(self, "hooks") and self.hooks is not None:
                ctx = HookContext(timestep=getattr(self, "current_timestep", 0), component=f"predictor.blocks.{i}")
                x = self.hooks.apply(f"predictor.blocks.{i}.hook_resid_post", getattr(self, "current_timestep", 0), x, ctx)

        x = self.norm(x)

        target_preds = x[:, len(context_ids):, :]
        target_preds = self.predictor_project_back(target_preds)
        return target_preds

    def __call__(self, context_latents, context_ids, target_ids):
        return self.forward(context_latents, context_ids, target_ids)


@register("vjepa", WorldModelFamily.JEPA, "Video Joint-Embedding Predictive Architecture")
class VJEPAAdapter(BaseModelAdapter, HookedRootModule):
    """Architecturally correct adapter for V-JEPA."""

    def __init__(self, config: WorldModelConfig):
        BaseModelAdapter.__init__(self, config)
        HookedRootModule.__init__(self)
        self.config = config

        img_size = getattr(config, "img_size", 224)
        patch_size = getattr(config, "patch_size", 16)
        num_frames = getattr(config, "num_frames", 16)
        tubelet_size = getattr(config, "tubelet_size", 2)
        embed_dim = getattr(config, "d_embed", 768)
        depth = getattr(config, "n_layers", 12)
        num_heads = getattr(config, "n_heads", 12)

        predictor_embed_dim = getattr(config, "predictor_embed_dim", 384)
        predictor_depth = getattr(config, "predictor_depth", 6)
        predictor_num_heads = getattr(config, "predictor_num_heads", 6)

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
        if self.last_context_ids is None or self.last_target_ids is None:
            n_patches = self.context_encoder.patch_embed.n_patches
            ctx_ids = list(range(int(n_patches * 0.5)))
            tgt_ids = list(range(int(n_patches * 0.5), n_patches))
        else:
            ctx_ids = self.last_context_ids
            tgt_ids = self.last_target_ids

        self.predictor.hooks = self.hooks
        self.predictor.current_timestep = self.current_timestep
        return self.predictor(state, ctx_ids, tgt_ids)

    @classmethod
    def from_checkpoint(
        cls, path: str, config: Optional[WorldModelConfig] = None
    ) -> "VJEPAAdapter":
        """Loads V-JEPA from checkpoint file or state dict strictly."""
        sd = torch.load(path, map_location="cpu")

        if config is None:
            config = WorldModelConfig(backend="vjepa")

        if "encoder" in sd:
            enc_sd = sd["encoder"]
            mapped_enc = {}
            for k, v in enc_sd.items():
                clean_k = k.replace("module.", "")
                clean_k = clean_k.replace(".mlp.fc1.", ".mlp.0.")
                clean_k = clean_k.replace(".mlp.fc2.", ".mlp.2.")
                mapped_enc[clean_k] = v

            if "predictor" in sd:
                pred_sd = sd["predictor"]
                mapped_pred = {}
                max_pred_idx = 0
                for k, v in pred_sd.items():
                    clean_k = k.replace("module.predictor_blocks.", "blocks.")
                    clean_k = clean_k.replace("module.predictor_pos_embed", "pos_embed")
                    clean_k = clean_k.replace("module.predictor_norm.", "norm.")
                    clean_k = clean_k.replace("module.predictor_proj.", "predictor_project_back.")
                    clean_k = clean_k.replace("module.", "")
                    clean_k = clean_k.replace(".mlp.fc1.", ".mlp.0.")
                    clean_k = clean_k.replace(".mlp.fc2.", ".mlp.2.")
                    mapped_pred[clean_k] = v
                    if clean_k.startswith("blocks."):
                        try:
                            b_idx = int(clean_k.split(".")[1])
                            max_pred_idx = max(max_pred_idx, b_idx)
                        except ValueError:
                            pass
                if max_pred_idx > 0:
                    config.predictor_depth = max_pred_idx + 1

            adapter = cls(config)

            res_ctx = adapter.context_encoder.load_state_dict(mapped_enc, strict=True)
            res_tgt = adapter.target_encoder.load_state_dict(mapped_enc, strict=True)
            print(f"[VJEPAAdapter.from_checkpoint] Context/Target Encoder load status: {res_ctx}")

            if "predictor" in sd:
                res_pred = adapter.predictor.load_state_dict(mapped_pred, strict=True)
                print(f"[VJEPAAdapter.from_checkpoint] Predictor load status: {res_pred}")
        elif "model" in sd:
            adapter = cls(config)
            res = adapter.load_state_dict(sd["model"], strict=True)
            print(f"[VJEPAAdapter.from_checkpoint] Full model load status: {res}")
        else:
            adapter = cls(config)
            res = adapter.load_state_dict(sd, strict=True)
            print(f"[VJEPAAdapter.from_checkpoint] State dict load status: {res}")

        adapter.eval()
        return adapter
