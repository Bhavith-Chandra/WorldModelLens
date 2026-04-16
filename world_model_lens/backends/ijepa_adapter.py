import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import logging
from typing import Optional, Tuple, List, Dict, Any, Union

from world_model_lens.backends.base_adapter import BaseModelAdapter, WorldModelCapabilities
from world_model_lens.backends.registry import register
from world_model_lens.core.types import WorldModelFamily
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooked_root import HookedRootModule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I-JEPA Model Components (Vision Transformer & Predictor)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, C, H, W] -> [B, embed_dim, grid_size, grid_size] -> [B, N, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

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
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.hook_query(q)
        k = self.hook_key(k)
        v = self.hook_value(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask expected shape [B, num_heads, N, N]
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

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
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

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = self.hook_resid_mid(x)

        mlp_in = self.norm2(x)
        mlp_in = self.hook_mlp_in(mlp_in)
        mlp_out = self.mlp(mlp_in)
        mlp_out = self.hook_mlp_out(mlp_out)

        x = x + mlp_out
        x = self.hook_resid_post(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=6, num_heads=3
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        self.hook_resid_pre = nn.Identity()

        # Proper initialization for positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, patch_ids=None):
        """Processes (subset of) patches. Positional embeddings are sliced BEFORE addition."""
        # 1. Patchify
        x = self.patch_embed(x)

        # 2. Select specific patches if specified
        if patch_ids is not None:
            # patch_ids can be a single list or a tensor [B, N_subset]
            if isinstance(patch_ids, (list, tuple, torch.Tensor)):
                if isinstance(patch_ids, (list, tuple)):
                    patch_ids = torch.tensor(patch_ids, device=x.device)
                if patch_ids.dim() == 1:
                    # Same patches for all items in batch
                    pos_embed = self.pos_embed[:, patch_ids, :]
                    x = x[:, patch_ids, :]
                else:
                    # Different patches for each item in batch
                    # x is [B, N_full, C], pos_embed is [1, N_full, C]
                    # We need to extract [B, N_subset, C]
                    B, N_full, C = x.shape
                    pos_embed = self.pos_embed.expand(B, -1, -1)

                    # Gather selected patches
                    x = torch.gather(x, 1, patch_ids.unsqueeze(-1).expand(-1, -1, C))
                    pos_embed = torch.gather(
                        pos_embed, 1, patch_ids.unsqueeze(-1).expand(-1, -1, C)
                    )
        else:
            pos_embed = self.pos_embed

        # 3. Add positional embeddings to only the visible patches
        x = x + pos_embed
        return self.forward_blocks(x)

    def forward_blocks(self, x, mask=None):
        """Processes latent embeddings through the transformer blocks."""
        x = self.pos_drop(x)
        x = self.hook_resid_pre(x)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        return x


class IJEPAPredictor(nn.Module):
    """Predictor transformer that maps context embeddings to target embeddings."""

    def __init__(
        self, encoder_embed_dim=192, predictor_embed_dim=384, depth=4, num_heads=6, num_patches=196
    ):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        # Bottleneck: Project from encoder space to predictor space
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim))

        self.blocks = nn.ModuleList(
            [Block(dim=predictor_embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        
        self.hook_resid_pre = nn.Identity()

        # Project back to encoder representation space for MSE loss
        self.predictor_project_back = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        # Proper initialization matching I-JEPA paper
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context_latents, context_ids, target_ids):
        """
        Args:
            context_latents: [B, N_context, encoder_embed_dim] (post context encoder)
            context_ids: indices of context patches
            target_ids: indices of target patches (to be predicted)
        """
        B = context_latents.shape[0]

        # 1. Project context to predictor space
        context_inputs = self.predictor_embed(context_latents)
        # Add positions in predictor space
        context_inputs = context_inputs + self.pos_embed[:, context_ids, :]

        # 2. Prepare target tokens
        target_tokens = self.mask_token.expand(B, len(target_ids), -1)
        target_inputs = target_tokens + self.pos_embed[:, target_ids, :]

        # 3. Process concatenated sequence
        x = torch.cat([context_inputs, target_inputs], dim=1)
        x = self.hook_resid_pre(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 4. Extract target predictions and project back
        target_preds = x[:, len(context_ids) :, :]
        target_preds = self.predictor_project_back(target_preds)

        return target_preds

    def get_last_self_attention(self):
        return self.blocks[-1].attn.last_attn_weights


# ---------------------------------------------------------------------------
# I-JEPA Adapter Implementation
# ---------------------------------------------------------------------------


@register("ijepa", WorldModelFamily.JEPA, "Image Joint-Embedding Predictive Architecture")
class IJEPAAdapter(BaseModelAdapter, HookedRootModule):
    """Architecturally correct adapter for I-JEPA."""

    def __init__(self, config: WorldModelConfig):
        BaseModelAdapter.__init__(self, config)
        HookedRootModule.__init__(self)
        self.config = config

        # Parameters from config
        img_size = getattr(config, "img_size", 224)
        patch_size = getattr(config, "patch_size", 16)
        embed_dim = getattr(config, "d_embed", 192)
        depth = getattr(config, "n_layers", 6)
        num_heads = getattr(config, "n_heads", 3)

        predictor_embed_dim = getattr(config, "predictor_embed_dim", 384)
        predictor_depth = getattr(config, "predictor_depth", 4)
        predictor_heads = getattr(config, "predictor_heads", 6)
        num_patches = (img_size // patch_size) ** 2

        # Context Encoder (Trainable)
        self.context_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        # Target Encoder (EMA updated)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor (Trainable)
        self.predictor = IJEPAPredictor(
            encoder_embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=num_patches,
        )

        self._capabilities = WorldModelCapabilities(
            has_decoder=False, has_reward_head=False, uses_actions=False, is_rl_trained=False
        )

        # Last known masks for inference/interpretability
        self.last_context_ids = None
        self.last_target_ids = None
        
        self.setup_hooks()

    @property
    def capabilities(self) -> WorldModelCapabilities:
        return self._capabilities

    def update_target_encoder(self, momentum: float = 0.999):
        """EMA update for the target encoder weights."""
        with torch.no_grad():
            for p_target, p_context in zip(
                self.target_encoder.parameters(), self.context_encoder.parameters()
            ):
                p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)

    def _get_block(
        self, grid_size, min_scale, max_scale, aspect_ratio_min=0.75, aspect_ratio_max=1.5
    ):
        """Samples a contiguous rectangle in the patch grid directly by height and width."""
        n_patches = grid_size**2
        target_area = random.uniform(min_scale, max_scale) * n_patches
        aspect_ratio = random.uniform(aspect_ratio_min, aspect_ratio_max)

        h = max(1, min(grid_size, int((target_area * aspect_ratio) ** 0.5)))
        w = max(1, min(grid_size, int((target_area / aspect_ratio) ** 0.5)))

        # NOTE: Clamping to grid_size here can cause the actual area to be smaller
        # than intended for large target blocks samples near the boundaries.

        top = random.randint(0, grid_size - h)
        left = random.randint(0, grid_size - w)

        indices = []
        for r in range(top, top + h):
            for c in range(left, left + w):
                indices.append(r * grid_size + c)
        return set(indices)

    def _get_structured_masks(self, num_patches, grid_size, n_targets=4):
        """
        Implements I-JEPA rect-on-rect multi-block masking.
        Returns:
            context_ids: List[int]
            target_ids_list: List[List[int]] (one list per target block)
        """
        # 1. Sample target blocks (4 blocks, ~15-20% area each)
        target_ids_list = []
        all_target_indices = set()
        for _ in range(n_targets):
            block = self._get_block(grid_size, 0.15, 0.20)
            target_ids_list.append(list(block))
            all_target_indices.update(block)

        # 2. Sample context block (~85% area) with resampling fallback
        found_context = False
        attempts = 0
        while not found_context and attempts < 10:
            # Context rectangle covers ~85% of grid independently
            context_raw = self._get_block(grid_size, 0.80, 0.90)
            # Subtract target overlaps
            context_indices = context_raw - all_target_indices

            # Check for degeneracy: context must retain >= 50% of its original patches
            if len(context_indices) >= 0.5 * len(context_raw) and len(context_indices) > 0:
                found_context = True
            attempts += 1

        if not found_context:
            # Fallback to a simple complement if resampling fails persistently
            all_patches = set(range(num_patches))
            context_indices = all_patches - all_target_indices

        return list(context_indices), target_ids_list

    def compute_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes the I-JEPA MSE loss, predicting each target block independently."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]

        # 1. Generate Structured Masks
        grid_size = self.context_encoder.patch_embed.grid_size
        num_patches = self.context_encoder.patch_embed.n_patches
        context_ids, target_ids_list = self._get_structured_masks(num_patches, grid_size)

        # 2. Target Encoder: Full image pass (No Gradients)
        with torch.no_grad():
            target_reps = self.target_encoder(obs)  # [B, N_full, C]

        # 3. Context Encoder: Context-only pass
        context_latents = self.context_encoder(obs, patch_ids=context_ids)  # [B, N_context, C]

        # 4. Predict targets block-wise as per the original I-JEPA paper
        total_loss = 0.0
        for block_ids in target_ids_list:
            target_gt_block = target_reps[:, block_ids, :]
            predicted_block = self.predictor(context_latents, context_ids, block_ids)
            total_loss += F.mse_loss(predicted_block, target_gt_block)

        loss = total_loss / len(target_ids_list)

        # Update state trackers (flattened targets for backward compatibility in dynamics)
        self.last_context_ids = context_ids
        self.last_target_ids = sorted(list(set([i for block in target_ids_list for i in block])))

        return loss

    def encode(
        self, obs: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode visible context patches.

        WARNING: This implementation relies on internal state (last_context_ids).
        Always ensure encode() is called before dynamics() in the same masking session
        for structurally consistent predictions.
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        # Generate/refresh masks if needed
        if self.last_context_ids is None:
            grid_size = self.context_encoder.patch_embed.grid_size
            num_patches = self.context_encoder.patch_embed.n_patches
            ctx, targets = self._get_structured_masks(num_patches, grid_size)
            self.last_context_ids = ctx
            self.last_target_ids = sorted([i for block in targets for i in block])

        context_latents = self.context_encoder(obs, patch_ids=self.last_context_ids)
        return context_latents, context_latents

    def target_encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Expose target encoder as a separate hook point for ground-truth comparison."""
        with torch.no_grad():
            return self.target_encoder(obs)

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """Predict target patches from encoded context.

        NOTE: If the input hidden state 'h' does not match the currently active
        context masking session (generated during encode()), this method returns
        zeros to prevent crashing due to dimension mismatches.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)

        # Safety check: ensure h matches the context_ids being used
        if self.last_context_ids is None or self.last_target_ids is None:
            # If masks are uninitialized or inconsistent, return zero-prediction
            target_size = len(self.last_target_ids) if self.last_target_ids else 50
            return torch.zeros(h.shape[0], target_size, h.shape[-1], device=h.device)

        if h.shape[1] != len(self.last_context_ids):
            # We return zeros if the state dimension doesn't match the masking.
            # This handles the HookedWorldModel t=0 case or cross-session stale states.
            target_size = len(self.last_target_ids) if self.last_target_ids else 50
            return torch.zeros(h.shape[0], target_size, h.shape[-1], device=h.device)

        if self.last_target_ids is None:
            self.last_target_ids = list(range(10))

        pred_latents = self.predictor(h, self.last_context_ids, self.last_target_ids)
        return pred_latents

    def transition(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor = None
    ) -> torch.Tensor:
        return h

    def initial_state(
        self, batch_size: int = 1, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_size = self.context_encoder.patch_embed.grid_size
        num_patches = self.context_encoder.patch_embed.n_patches

        # Ensure we have consistent masks for the initial state
        if self.last_context_ids is None:
            ctx, targets = self._get_structured_masks(num_patches, grid_size)
            self.last_context_ids = ctx
            self.last_target_ids = sorted([i for block in targets for i in block])

        embed_dim = self.context_encoder.norm.normalized_shape[0]
        num_context = len(self.last_context_ids)
        num_target = len(self.last_target_ids)

        h = torch.zeros(batch_size, num_context, embed_dim, device=device)
        z = torch.zeros(batch_size, num_target, embed_dim, device=device)
        return h, z

    @classmethod
    def from_checkpoint(
        cls, path: str, config: Optional[WorldModelConfig] = None
    ) -> "IJEPAAdapter":
        """Loads I-JEPA from checkpoint. Supports native and Meta official weights."""
        if config is None:
            config = WorldModelConfig(backend="ijepa")
        adapter = cls(config)

        # Load state dict
        sd = torch.load(path, map_location="cpu")

        # Official Meta weights are typically nested under 'model' or 'encoder'
        if "model" in sd:
            sd = sd["model"]
        elif "encoder" in sd:
            sd = sd["encoder"]

        # Check if it's a Meta ViT-H checkpoint or a compatible one
        # Meta keys are top-level (e.g. 'patch_embed.proj.weight')
        # Our keys are nested under context_encoder and target_encoder
        is_meta = any(k.startswith("patch_embed") or k.startswith("blocks.0") for k in sd.keys())

        if is_meta:
            logger.info(
                f"Detected Meta-style ViT checkpoint from {path}. Mapping to context/target encoders."
            )
            # Load into both encoders
            adapter.context_encoder.load_state_dict(sd, strict=False)
            adapter.target_encoder.load_state_dict(sd, strict=False)
            # Predictor will remain initialized randomly unless also in checkpoint
        else:
            # Traditional checkpoint matching our adapter's full state dict
            adapter.load_state_dict(sd)

        adapter.eval()
        return adapter
