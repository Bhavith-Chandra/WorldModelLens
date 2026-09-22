"""MAE (Masked Autoencoder) Adapter.

Implements Meta Masked Autoencoders (MAE) with Vision Transformer encoder and lightweight ViT
pixel reconstruction decoder for pixel-space vs feature-space comparative interpretability in World Model Lens.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, cast

from world_model_lens.backends.base_adapter import BaseModelAdapter, WorldModelCapabilities
from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooked_root import HookedRootModule
from world_model_lens.core.hooks import HookContext, HookPoint, HookRegistry
from world_model_lens.backends.ijepa_adapter import PatchEmbed, Block, VisionTransformer


class MAEDecoder(nn.Module):
    """Lightweight ViT Pixel Reconstruction Decoder for MAE."""

    def __init__(
        self,
        num_patches: int = 196,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16
    ):
        super().__init__()
        self.prefix = ""
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block(dim=decoder_embed_dim, num_heads=decoder_num_heads) for _ in range(decoder_depth)]
        )
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.hook_resid_pre = nn.Identity()

        # Final linear projection to RGB pixels per patch (16x16x3 = 768 values)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context_latents: torch.Tensor, context_ids: List[int], target_ids: List[int]) -> torch.Tensor:
        B = context_latents.shape[0]

        # 1. Project context latents to decoder embedding space
        context_inputs = self.decoder_embed(context_latents) + self.pos_embed[:, context_ids, :]

        # 2. Prepare target mask tokens
        target_tokens = self.mask_token.expand(B, len(target_ids), -1)
        target_inputs = target_tokens + self.pos_embed[:, target_ids, :]

        # 3. Concatenate and process through ViT decoder blocks
        x = torch.cat([context_inputs, target_inputs], dim=1)
        x = self.hook_resid_pre(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if hasattr(self, "hooks") and self.hooks is not None:
                ctx = HookContext(timestep=getattr(self, "current_timestep", 0), component=f"decoder.blocks.{i}")
                x = self.hooks.apply(f"decoder.blocks.{i}.hook_resid_post", getattr(self, "current_timestep", 0), x, ctx)

        x = self.norm(x)

        # 4. Extract target mask token outputs and project to RGB pixels
        target_tokens_out = x[:, len(context_ids):, :]
        pixel_preds = self.decoder_pred(target_tokens_out) # [B, N_tgt, 768]
        return pixel_preds


@register("mae", WorldModelFamily.JEPA, "Masked Autoencoder (Pixel Reconstruction Loss)")
class MAEAdapter(BaseModelAdapter, HookedRootModule):
    """Architecturally correct adapter for Masked Autoencoders (MAE)."""

    def __init__(self, config: WorldModelConfig):
        BaseModelAdapter.__init__(self, config)
        HookedRootModule.__init__(self)
        self.config = config

        img_size = getattr(config, "img_size", 224)
        patch_size = getattr(config, "patch_size", 16)
        embed_dim = getattr(config, "d_embed", 192)
        depth = getattr(config, "n_layers", 6)
        num_heads = getattr(config, "n_heads", 3)

        decoder_embed_dim = getattr(config, "decoder_embed_dim", 512)
        decoder_depth = getattr(config, "decoder_depth", 8)
        decoder_num_heads = getattr(config, "decoder_num_heads", 16)
        num_patches = (img_size // patch_size) ** 2

        self.context_encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )

        self.decoder = MAEDecoder(
            num_patches=num_patches, patch_size=patch_size, in_chans=3,
            encoder_embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads
        )

        self.last_context_ids: Optional[List[int]] = None
        self.last_target_ids: Optional[List[int]] = None
        self.hooks = HookRegistry()
        self.current_timestep = 0

        self._capabilities = WorldModelCapabilities(
            has_decoder=True, has_reward_head=False, has_continue_head=False,
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

    def target_encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract ground-truth raw RGB pixel patches [B, N_patches, 768]."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        B, C, H, W = obs.shape
        p_val = self.context_encoder.patch_embed.patch_size
        p = p_val[0] if isinstance(p_val, (tuple, list)) else p_val
        h_grid, w_grid = H // p, W // p
        x = obs.reshape(B, C, h_grid, p, w_grid, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        patches = x.reshape(B, h_grid * w_grid, p * p * C)
        return patches

    def encode(self, obs: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.context_encoder.hooks = self.hooks
        self.context_encoder.current_timestep = self.current_timestep
        if self.last_context_ids is not None:
            ctx_latents = self.context_encoder(obs, patch_ids=self.last_context_ids)
        else:
            ctx_latents = self.context_encoder(obs)
        return ctx_latents, ctx_latents

    def dynamics(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.last_context_ids is None or self.last_target_ids is None:
            n_patches = self.context_encoder.patch_embed.n_patches
            ctx_ids = list(range(int(n_patches * 0.85)))
            tgt_ids = list(range(int(n_patches * 0.85), n_patches))
        else:
            ctx_ids = self.last_context_ids
            tgt_ids = self.last_target_ids

        self.decoder.hooks = self.hooks
        self.decoder.current_timestep = self.current_timestep
        return self.decoder(state, ctx_ids, tgt_ids)
