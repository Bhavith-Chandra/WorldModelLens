from __future__ import annotations

import contextlib
import uuid
from typing import Any, Optional, Set

import torch
from torch import Tensor

from world_model_lens.core.hooks import HookContext
from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.latent_trajectory import LatentTrajectory
from world_model_lens.core.types import WorldModelFamily


class ForwardRunner:
    """Isolated runner that performs the forward pass for a HookedWorldModel.

    This class delegates computations to the provided *hooked* object which is
    expected to implement the small helpers used by the forward loop (for
    example: _encode, _dynamics_and_prior, _posterior, _compute_kl_and_cache,
    _compute_heads, _build_state, _apply_and_cache, and _cache_prior_equivalent_at_t0).

    Keeping the loop here makes it easier to test the sequencing and to reuse
    the runner from different entry points (run_with_cache, run_with_hooks,
    etc.) without duplicating control-flow.
    """

    def __init__(self, hooked: Any):
        self.hooked = hooked

    def run_forward(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        cache: Optional[ActivationCache],
        names_filter: Optional[Set[str]],
        no_grad: bool = True,
    ) -> LatentTrajectory:
        T = obs_seq.shape[0]
        ctx_mgr = torch.no_grad() if no_grad else contextlib.nullcontext()

        states = []

        is_patch_axis = False
        if hasattr(self.hooked, "config"):
            is_patch_axis = getattr(self.hooked.config, "world_model_family", None) == WorldModelFamily.JEPA
            
        if is_patch_axis:
            return self._run_forward_patch_axis(obs_seq, cache, names_filter, ctx_mgr)

        adapter = getattr(self.hooked, "adapter", getattr(self.hooked, "_adapter", None))
        h: Tensor = adapter.initial_state()
        z: Tensor | None = None

        manager = getattr(self.hooked, "_hook_cache_manager", None)

        with ctx_mgr:
            for t in range(T):
                traj_so_far = (
                    LatentTrajectory(states=list(states), env_name="", episode_id=None)
                    if states
                    else None
                )
                ctx = (
                    HookContext(timestep=t, component="forward", trajectory_so_far=[])
                    if traj_so_far is None
                    else HookContext(
                        timestep=t, component="forward", trajectory_so_far=list(states)
                    )
                )

                if manager is not None:
                    adapter = getattr(self.hooked, "adapter", getattr(self.hooked, "_adapter", None))
                    obs_emb = adapter.encode(obs_seq[t])
                    obs_emb = manager.apply_and_cache(
                        "encoder.out", t, obs_emb, ctx, cache, names_filter
                    )
                else:
                    obs_emb = self.hooked._encode(obs_seq[t], t, ctx, cache, names_filter)

                if t == 0:
                    h = self.hooked._apply_and_cache("rnn.h", 0, h, ctx, cache, names_filter)
                    z_prior_logits = None
                else:
                    a_prev = action_seq[t - 1]
                    h, z_prior_raw, z_prior_prob = self.hooked._dynamics_and_prior(
                        h, z, a_prev, t, ctx, cache, names_filter
                    )
                    z_prior_logits = z_prior_raw

                z_post_raw, z_post_prob = self.hooked._posterior(
                    h, obs_emb, t, ctx, cache, names_filter
                )
                z = z_post_prob

                if t == 0:
                    z_prior_logits = z_post_raw
                    z_prior_prob = z_post_prob
                    # prefer HookCacheManager helper when available
                    manager = getattr(self.hooked, "_hook_cache_manager", None)
                    if manager is not None:
                        manager.set_prior_equivalent(
                            cache,
                            0,
                            z_post_raw.detach(),
                            z_post_prob.detach(),
                            names_filter,
                        )
                    else:
                        # fall back to cache helper
                        try:
                            if cache is not None:
                                cache.set_prior_equivalent(
                                    0,
                                    z_post_raw.detach(),
                                    z_post_prob.detach(),
                                    list(names_filter) if names_filter is not None else None,
                                )
                        except Exception:
                            # last-resort fallback to hooked shim
                            self.hooked._cache_prior_equivalent_at_t0(
                                z_post_raw, z_post_prob, 0, cache, names_filter
                            )

                kl = self.hooked._compute_kl_and_cache(
                    z_post_prob, z_prior_prob, t, ctx, cache, names_filter
                )

                reward_val, cont_val, actor_logits_out, value_val = self.hooked._compute_heads(
                    h, z_post_prob, t, ctx, cache, names_filter
                )

                state = self.hooked._build_state(
                    h,
                    z_post_prob,
                    z_prior_prob,
                    t,
                    action_seq,
                    reward_val,
                    cont_val,
                    actor_logits_out,
                    value_val,
                )
                states.append(state)

        return LatentTrajectory(
            states=states,
            env_name=self.hooked.name,
            episode_id=f"run_{uuid.uuid4().hex[:8]}",
            imagined=False,
        )

    def _run_forward_patch_axis(
        self,
        obs_batch: Tensor,
        cache: Optional[ActivationCache],
        names_filter: Optional[Set[str]],
        ctx_mgr: Any,
    ) -> LatentTrajectory:
        """Patch-Axis Mode: executes spatial masking loop concurrently.
        
        obs_batch is implicitly [B, C, H, W] for a single step (or unbatched).
        """
        states = []
        adapter = getattr(self.hooked, "adapter", getattr(self.hooked, "_adapter", None))
        manager = getattr(self.hooked, "_hook_cache_manager", None)
        
        with ctx_mgr:
            # 1. Context Encoder
            context_latents, _ = adapter.encode(obs_batch)
            if manager is not None:
                ctx = HookContext(timestep=0, component="forward", trajectory_so_far=[])
                manager.apply_and_cache("encoder.out", 0, context_latents, ctx, cache, names_filter)
            
            # 2. Target Encoder (EMA)
            target_reps = adapter.target_encode(obs_batch)
            if manager is not None:
                manager.apply_and_cache("target_encoder.out", 0, target_reps, ctx, cache, names_filter)
            
            # 3. Predictor 
            # IJEPA Predictor expects (context, context_ids, target_ids)
            # In validation/inference without explicit masks, we can predict all targets 
            # or use the last generated masks from encode()
            c_ids = adapter.last_context_ids
            t_ids = adapter.last_target_ids
            
            if c_ids is None or t_ids is None:
                # Fallback if encode() didn't cache spatial masks
                N = getattr(adapter.config, "num_patches", 196)
                c_ids = list(range(int(N * 0.85)))
                t_ids = list(range(int(N * 0.85), N))
            
            # Predictor manual loop to expose predictor.layer_N
            context_inputs = adapter.predictor.predictor_embed(context_latents)
            context_inputs = context_inputs + adapter.predictor.pos_embed[:, c_ids, :]
            
            B = obs_batch.shape[0] if obs_batch.dim() == 4 else 1
            target_tokens = adapter.predictor.mask_token.expand(B, len(t_ids), -1)
            target_inputs = target_tokens + adapter.predictor.pos_embed[:, t_ids, :]
            
            x = torch.cat([context_inputs, target_inputs], dim=1)
            
            # Manually run predictor layer loop to cache intermediate layer states
            for i, block in enumerate(adapter.predictor.blocks):
                x = block(x)
                if manager is not None:
                    manager.apply_and_cache(f"predictor.layer_{i}", 0, x, ctx, cache, names_filter)
                    
            x = adapter.predictor.norm(x)
            
            # Predictor final projections
            target_preds = x[:, len(c_ids):, :]
            target_preds = adapter.predictor.predictor_project_back(target_preds)
            if manager is not None:
                manager.apply_and_cache("predictor.final", 0, target_preds, ctx, cache, names_filter)

            # 4. Map output to LatentTrajectory across the spatial sequence (patches as timesteps)
            # We align patch sequences to timesteps. 
            num_targets = len(t_ids)
            for t in range(num_targets):
                h_patch = target_preds[:, t, :] if target_preds.dim() == 3 else target_preds[t]
                
                # Mock a LatentTrajectory spatial step
                state = self.hooked._build_state(
                    h=h_patch,
                    z_post_prob=torch.zeros_like(h_patch),
                    z_prior_prob=torch.zeros_like(h_patch),
                    t=t,
                    action_seq=None,
                    reward_val=None,
                    cont_val=None,
                    actor_logits_out=None,
                    value_val=None,
                )
                states.append(state)

        return LatentTrajectory(
            states=states,
            env_name=self.hooked.name,
            episode_id=f"run_spatial_{uuid.uuid4().hex[:8]}",
            imagined=False,
        )
