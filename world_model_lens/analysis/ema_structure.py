"""I-JEPA EMA-structure and controlled-ambiguity analyses.

The context and target encoders share an architecture, but their parameters
diverge during training because the target encoder follows the context encoder
through an exponential moving average (EMA).  This module measures that
representation-level divergence without confusing it with I-JEPA masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class EMAStructureDivergenceResult:
    """Context-versus-target encoder divergence for a batch of images."""

    context_representations: Tensor
    target_representations: Tensor
    delta: Tensor
    token_l2: Tensor
    token_cosine_distance: Tensor
    mean_l2: float
    mean_cosine_distance: float
    layer_mean_l2: Dict[int, float]


@dataclass
class InpaintingCandidateScore:
    """How one inpainted candidate aligns with I-JEPA's prediction."""

    candidate_index: int
    prediction_mse: float
    prediction_cosine_distance: float
    reference_mse: float
    reference_cosine_distance: float


class EMAStructureAnalyzer:
    """Measure EMA divergence and score controlled-ambiguity inpaintings.

    The analyzer intentionally does not generate images.  Callers supply
    candidate inpaintings from their chosen diffusion system, which keeps the
    evaluation reproducible and prevents a particular generator from becoming
    an implicit experimental confound.
    """

    def __init__(self, adapter: nn.Module):
        if not hasattr(adapter, "context_encoder") or not hasattr(adapter, "target_encoder"):
            raise TypeError("EMAStructureAnalyzer requires an I-JEPA-style adapter.")
        self.adapter = adapter

    @staticmethod
    def _cosine_distance(left: Tensor, right: Tensor) -> Tensor:
        return 1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-8)

    @staticmethod
    def _capture_block_outputs(encoder: nn.Module) -> tuple[List[Tensor], List[object]]:
        outputs: List[Tensor] = []
        handles = []

        def capture(_: nn.Module, __: tuple[object, ...], output: Tensor) -> None:
            outputs.append(output.detach().clone())

        for block in encoder.blocks:
            handles.append(block.register_forward_hook(capture))
        return outputs, handles

    @staticmethod
    def _restore_training_mode(modules: Sequence[nn.Module], modes: Sequence[bool]) -> None:
        for module, mode in zip(modules, modes):
            module.train(mode)

    def compare_encoders(self, images: Tensor) -> EMAStructureDivergenceResult:
        """Compare full-image context and target representations.

        Both encoders receive every patch.  Consequently, the reported
        ``delta = f_ctx(x) - f_tgt(x)`` is attributable to EMA parameter
        structure, not to distinct context/target masks.
        """
        if images.dim() != 4:
            raise ValueError("images must have shape [batch, channels, height, width].")

        context_encoder = self.adapter.context_encoder
        target_encoder = self.adapter.target_encoder
        encoders = (context_encoder, target_encoder)
        modes = [encoder.training for encoder in encoders]
        context_layers, context_handles = self._capture_block_outputs(context_encoder)
        target_layers, target_handles = self._capture_block_outputs(target_encoder)

        try:
            context_encoder.eval()
            target_encoder.eval()
            with torch.no_grad():
                context = context_encoder(images)
                target = target_encoder(images)
        finally:
            for handle in [*context_handles, *target_handles]:
                handle.remove()
            self._restore_training_mode(encoders, modes)

        delta = context - target
        token_l2 = delta.norm(dim=-1)
        token_cosine_distance = self._cosine_distance(context, target)
        layer_mean_l2 = {
            index: (context_layer - target_layer).norm(dim=-1).mean().item()
            for index, (context_layer, target_layer) in enumerate(zip(context_layers, target_layers))
        }
        return EMAStructureDivergenceResult(
            context_representations=context,
            target_representations=target,
            delta=delta,
            token_l2=token_l2,
            token_cosine_distance=token_cosine_distance,
            mean_l2=token_l2.mean().item(),
            mean_cosine_distance=token_cosine_distance.mean().item(),
            layer_mean_l2=layer_mean_l2,
        )

    def score_inpainting_candidates(
        self,
        source_images: Tensor,
        candidate_images: Tensor,
        context_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> List[InpaintingCandidateScore]:
        """Score candidate inpaintings against a fixed I-JEPA prediction.

        Args:
            source_images: Original image batch, shape ``[1, C, H, W]``.
            candidate_images: Candidate inpaintings, shape ``[K, C, H, W]``.
            context_ids: Visible patch IDs used by the context encoder.
            target_ids: Ambiguous/inpainted patch IDs predicted by I-JEPA.

        Lower prediction metrics indicate a candidate whose target-encoder
        representation agrees more closely with the prediction implied by the
        unmodified visible context.  Reference metrics quantify its drift from
        the original target representation.
        """
        if source_images.dim() != 4 or source_images.shape[0] != 1:
            raise ValueError("source_images must have shape [1, C, H, W].")
        if candidate_images.dim() != 4:
            raise ValueError("candidate_images must have shape [K, C, H, W].")
        if not context_ids or not target_ids:
            raise ValueError("context_ids and target_ids must both be non-empty.")

        context_encoder = self.adapter.context_encoder
        target_encoder = self.adapter.target_encoder
        predictor = self.adapter.predictor
        modules = (context_encoder, target_encoder, predictor)
        modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            with torch.no_grad():
                context = context_encoder(source_images, patch_ids=list(context_ids))
                prediction = predictor(context, list(context_ids), list(target_ids))
                reference = target_encoder(source_images)[:, list(target_ids), :]
                candidates = target_encoder(candidate_images)[:, list(target_ids), :]
        finally:
            self._restore_training_mode(modules, modes)

        prediction = prediction.expand_as(candidates)
        reference = reference.expand_as(candidates)
        prediction_mse = (prediction - candidates).square().mean(dim=(1, 2))
        reference_mse = (reference - candidates).square().mean(dim=(1, 2))
        prediction_cos = self._cosine_distance(prediction, candidates).mean(dim=1)
        reference_cos = self._cosine_distance(reference, candidates).mean(dim=1)
        return [
            InpaintingCandidateScore(
                candidate_index=index,
                prediction_mse=prediction_mse[index].item(),
                prediction_cosine_distance=prediction_cos[index].item(),
                reference_mse=reference_mse[index].item(),
                reference_cosine_distance=reference_cos[index].item(),
            )
            for index in range(candidate_images.shape[0])
        ]
