"""Cross-modal probing for vision-language world models.

This module solves the problem stated in Issue #11: probing a world-model
latent space for plain-English concepts is hard when you only have raw tensors.

The solution is a two-step pipeline:

1. **Alignment** -- train a lightweight CrossModalProjector (a single linear
   layer) that maps world-model hidden states into CLIP's shared vision-language
   embedding space.

2. **Querying** -- once projected, any latent vector can be compared with a
   CLIP text embedding, letting researchers ask in plain English:
   "Does this latent contain the concept of 'danger'?"

Example
-------
>>> from world_model_lens.probing.crossmodal import CrossModalProber
>>> prober = CrossModalProber()
>>> projector = prober.train_projector(latents, clip_image_features)
>>> result = prober.query_concept(latents, "danger", projector=projector)
>>> print(f"'danger' similarity: {result.similarity:.3f} -- present: {result.is_present}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    from transformers import CLIPModel, CLIPProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CrossModalResult:
    """Aggregated result of a cross-modal probing session.

    Attributes
    ----------
    alignment_score:
        Mean cosine similarity between projected latents and CLIP features.
    shared_concepts:
        Concept strings that scored above the detection threshold.
    retrieval_accuracy:
        Top-1 retrieval accuracy (fraction of queries with correct top result).
    concept_similarities:
        Per-concept mean similarity scores.
    projection_loss:
        Final MSE loss of the trained projector, or None if not trained.
    """

    alignment_score: float = 0.0
    shared_concepts: List[str] = field(default_factory=list)
    retrieval_accuracy: float = 0.0
    concept_similarities: Dict[str, float] = field(default_factory=dict)
    projection_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialise to a plain Python dictionary."""
        return {
            "alignment_score": self.alignment_score,
            "shared_concepts": self.shared_concepts,
            "retrieval_accuracy": self.retrieval_accuracy,
            "concept_similarities": self.concept_similarities,
            "projection_loss": self.projection_loss,
        }


@dataclass
class ConceptQueryResult:
    """Result of a single plain-English concept query.

    Attributes
    ----------
    concept:
        The queried concept string (e.g. "danger").
    similarity:
        Mean cosine similarity between projected latents and the CLIP text
        embedding for *concept*.
    is_present:
        True when *similarity* exceeds *threshold*.
    threshold:
        Detection threshold used to compute *is_present*.
    per_sample_similarities:
        Per-sample cosine similarities of shape [N] (on CPU).
    """

    concept: str
    similarity: float
    is_present: bool
    threshold: float = 0.0
    per_sample_similarities: Optional[Tensor] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialise to a plain Python dictionary (excludes tensor field)."""
        return {
            "concept": self.concept,
            "similarity": self.similarity,
            "is_present": self.is_present,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Learnable projector
# ---------------------------------------------------------------------------


class CrossModalProjector(nn.Module):
    """Learnable affine projection from latent space to CLIP embedding space.

    A single nn.Linear layer trained to align world-model hidden states with
    CLIP's vision-language embeddings.  After training via
    CrossModalProber.train_projector, any latent vector can be compared
    directly with CLIP text embeddings.

    Parameters
    ----------
    d_latent:
        Dimensionality of the world-model latent space.
    d_clip:
        Dimensionality of the CLIP embedding space (typically 512).
    """

    def __init__(self, d_latent: int, d_clip: int = 512) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_clip = d_clip
        self.projection = nn.Linear(d_latent, d_clip, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, latents: Tensor) -> Tensor:
        """Project latents into CLIP space and L2-normalise.

        Parameters
        ----------
        latents:
            Shape [N, d_latent].

        Returns
        -------
        Tensor
            L2-normalised embeddings of shape [N, d_clip].
        """
        return F.normalize(self.projection(latents), dim=-1)


# ---------------------------------------------------------------------------
# Main prober
# ---------------------------------------------------------------------------


class CrossModalProber:
    """Probe world-model latent states via CLIP's vision-language embedding.

    Enables researchers to ask plain-English concept questions about any latent
    state: "Does this latent contain the concept of 'danger'?"

    Typical workflow
    ----------------
    1. Collect latents [N, D] and matching clip_features [N, 512] from a CLIP
       image encoder on the same observations.
    2. Call train_projector to learn a latent -> CLIP mapping.
    3. Call query_concept with any English phrase to probe a concept.
    4. Call batch_query_concepts to rank a list of concepts at once.

    Parameters
    ----------
    device:
        Target device. Defaults to CUDA when available, else CPU.
    clip_model_name:
        HuggingFace model ID for CLIP (default openai/clip-vit-base-patch32).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.device = device or _get_device()
        self.clip_model_name = clip_model_name
        self._clip_model: Optional[object] = None
        self._clip_processor: Optional[object] = None

    # ------------------------------------------------------------------
    # CLIP loading (lazy)
    # ------------------------------------------------------------------

    def _load_clip(self) -> None:
        """Lazy-load the CLIP model and processor on first use."""
        if self._clip_model is not None:
            return
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The transformers package is required for CrossModalProber. "
                "Install it with: pip install transformers"
            )
        logger.info("Loading CLIP model '%s' ...", self.clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self._clip_model.eval()
        self._clip_model.to(self.device)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(self, texts: List[str]) -> Tensor:
        """Encode text strings to L2-normalised CLIP text embeddings.

        Parameters
        ----------
        texts:
            List of strings to encode.

        Returns
        -------
        Tensor
            Shape [len(texts), D_clip], L2-normalised.
        """
        self._load_clip()
        inputs = self._clip_processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._clip_model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

    # ------------------------------------------------------------------
    # Projector training
    # ------------------------------------------------------------------

    def train_projector(
        self,
        latents: Tensor,
        clip_features: Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> CrossModalProjector:
        """Train a linear projector that maps latents into CLIP embedding space.

        Minimises MSE between projector(latents) and L2-normalised CLIP image
        features so that, after training, dot-products with CLIP text embeddings
        yield meaningful cosine similarities.

        Parameters
        ----------
        latents:
            World-model hidden states [N, d_latent].
        clip_features:
            Corresponding CLIP image embeddings [N, d_clip].
        epochs:
            Number of full-dataset passes.
        lr:
            Adam learning rate.
        batch_size:
            Mini-batch size.

        Returns
        -------
        CrossModalProjector
            Trained projector in eval mode, ready for query_concept.
        """
        latents = latents.to(self.device)
        clip_features = F.normalize(clip_features.to(self.device), dim=-1)

        d_latent = latents.shape[-1]
        d_clip = clip_features.shape[-1]
        projector = CrossModalProjector(d_latent, d_clip).to(self.device)
        optimizer = torch.optim.Adam(projector.parameters(), lr=lr)

        n = latents.shape[0]
        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                loss = F.mse_loss(projector(latents[idx]), clip_features[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            if (epoch + 1) % 50 == 0:
                logger.debug(
                    "Projector epoch %d/%d -- loss %.5f",
                    epoch + 1,
                    epochs,
                    epoch_loss / max(steps, 1),
                )

        projector.eval()
        return projector

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def project_latents(
        self,
        latents: Tensor,
        projector: Optional[CrossModalProjector] = None,
    ) -> Tensor:
        """Project latents into CLIP embedding space.

        If no trained projector is provided the latents are L2-normalised
        directly (zero-shot mode; requires d_latent == d_clip).

        Parameters
        ----------
        latents:
            Shape [N, D].
        projector:
            Trained CrossModalProjector (optional).

        Returns
        -------
        Tensor
            L2-normalised embeddings [N, D_clip].
        """
        latents = latents.to(self.device)
        if projector is not None:
            projector = projector.to(self.device)
            with torch.no_grad():
                return projector(latents)
        return F.normalize(latents, dim=-1)

    # ------------------------------------------------------------------
    # Concept querying  (Issue #11 core feature)
    # ------------------------------------------------------------------

    def query_concept(
        self,
        latents: Tensor,
        concept: str,
        projector: Optional[CrossModalProjector] = None,
        threshold: float = 0.0,
        prompt_template: str = "a photo of {}",
    ) -> ConceptQueryResult:
        """Query whether a plain-English concept is present in latent states.

        Projects the world-model latents into CLIP space, then measures cosine
        similarity with the CLIP text embedding of *concept*.

        Parameters
        ----------
        latents:
            World-model hidden states [N, D].
        concept:
            Plain-English concept to probe, e.g. "danger", "a dog", "fire".
        projector:
            Trained latent -> CLIP projector.  If None, latents are used
            directly (requires d_latent == d_clip).
        threshold:
            Minimum mean similarity to classify the concept as present.
        prompt_template:
            Wraps the concept before CLIP encoding.  The {} placeholder is
            replaced with concept.  Prompt engineering can improve accuracy
            (e.g. "a scene containing {}").

        Returns
        -------
        ConceptQueryResult
            Contains the mean similarity score, a boolean is_present flag,
            and per-sample similarities.
        """
        prompt = prompt_template.format(concept)
        text_features = self.encode_text([prompt])            # [1, D_clip]
        projected = self.project_latents(latents, projector)  # [N, D_clip]

        per_sample = (projected @ text_features.T).squeeze(-1)  # [N]
        mean_sim = float(per_sample.mean().item())

        return ConceptQueryResult(
            concept=concept,
            similarity=mean_sim,
            is_present=mean_sim > threshold,
            threshold=threshold,
            per_sample_similarities=per_sample.cpu(),
        )

    def batch_query_concepts(
        self,
        latents: Tensor,
        concepts: List[str],
        projector: Optional[CrossModalProjector] = None,
        threshold: float = 0.0,
        prompt_template: str = "a photo of {}",
    ) -> List[ConceptQueryResult]:
        """Query multiple concepts at once and rank by similarity.

        More efficient than calling query_concept in a loop because all text
        embeddings are computed in a single CLIP forward pass.

        Parameters
        ----------
        latents:
            World-model hidden states [N, D].
        concepts:
            List of concept strings to probe.
        projector:
            Trained projector (optional).
        threshold:
            Detection threshold applied to all concepts.
        prompt_template:
            Prompt wrapper for each concept.

        Returns
        -------
        list[ConceptQueryResult]
            One result per concept, sorted by descending similarity.
        """
        prompts = [prompt_template.format(c) for c in concepts]
        text_features = self.encode_text(prompts)             # [C, D_clip]
        projected = self.project_latents(latents, projector)  # [N, D_clip]

        sims = projected @ text_features.T                    # [N, C]
        mean_sims = sims.mean(dim=0)                          # [C]

        results = [
            ConceptQueryResult(
                concept=concept,
                similarity=float(mean_sims[i].item()),
                is_present=float(mean_sims[i].item()) > threshold,
                threshold=threshold,
                per_sample_similarities=sims[:, i].cpu(),
            )
            for i, concept in enumerate(concepts)
        ]
        return sorted(results, key=lambda r: r.similarity, reverse=True)

    # ------------------------------------------------------------------
    # Vision-language alignment
    # ------------------------------------------------------------------

    def align_vision_language(
        self,
        vm_latents: Tensor,
        images: Tensor,
        captions: List[str],
        projector: Optional[CrossModalProjector] = None,
    ) -> CrossModalResult:
        """Compute alignment between world-model latents and CLIP features.

        Parameters
        ----------
        vm_latents:
            World-model latent tensor [N, D].
        images:
            Image tensor [N, C, H, W].
        captions:
            Caption strings for each image.
        projector:
            Trained projector (optional).

        Returns
        -------
        CrossModalResult
            Aggregated alignment metrics.
        """
        self._load_clip()
        with torch.no_grad():
            img_inputs = self._clip_processor(
                images=list(images), return_tensors="pt"
            )
            img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
            image_features = F.normalize(
                self._clip_model.get_image_features(**img_inputs), dim=-1
            )
        text_features = self.encode_text(captions)

        projected = self.project_latents(vm_latents, projector)
        img_text_sim = (image_features * text_features).sum(dim=-1).mean()
        vm_text_sim = (projected * text_features).sum(dim=-1).mean()
        alignment = float(((img_text_sim + vm_text_sim) / 2).item())

        return CrossModalResult(alignment_score=alignment)

    def find_shared_concepts(
        self,
        vm_latents: Tensor,
        concept_candidates: List[str],
        projector: Optional[CrossModalProjector] = None,
        prompt_template: str = "a photo of {}",
    ) -> List[Tuple[str, float]]:
        """Rank concept candidates by their presence in world-model latents.

        Parameters
        ----------
        vm_latents:
            World-model latents [N, D].
        concept_candidates:
            Candidate concept strings to rank.
        projector:
            Trained projector (optional).
        prompt_template:
            Prompt wrapper for encoding.

        Returns
        -------
        list[tuple[str, float]]
            (concept, similarity) pairs sorted by descending similarity.
        """
        results = self.batch_query_concepts(
            vm_latents,
            concept_candidates,
            projector=projector,
            prompt_template=prompt_template,
        )
        return [(r.concept, r.similarity) for r in results]

    def crossmodal_retrieval(
        self,
        vm_latents: Tensor,
        text_queries: List[str],
        projector: Optional[CrossModalProjector] = None,
        top_k: int = 1,
    ) -> List[List[int]]:
        """Retrieve the most relevant latent indices for each text query.

        Parameters
        ----------
        vm_latents:
            World-model latents [N, D].
        text_queries:
            Text query strings.
        projector:
            Trained projector (optional).
        top_k:
            Number of top-matching latent indices to return per query.

        Returns
        -------
        list[list[int]]
            top_k latent indices for each query, ranked by similarity.
        """
        text_features = self.encode_text(text_queries)          # [Q, D_clip]
        projected = self.project_latents(vm_latents, projector)  # [N, D_clip]

        sims = projected @ text_features.T                      # [N, Q]
        top_k = min(top_k, len(vm_latents))
        top_indices = sims.topk(top_k, dim=0).indices           # [top_k, Q]
        return [top_indices[:, q].tolist() for q in range(len(text_queries))]


# ---------------------------------------------------------------------------
# Standalone utility (no CLIP required)
# ---------------------------------------------------------------------------


def align_multimodal(
    vm_latents: Tensor,
    vlm_features: Tensor,
) -> Dict[str, float]:
    """Directly align world-model latents with VLM features (no CLIP needed).

    Computes cosine similarity and Euclidean distance between two embedding
    tensors.  Useful for quick sanity checks or when CLIP is unavailable.

    Parameters
    ----------
    vm_latents:
        World-model latents [N, D].
    vlm_features:
        VLM features [N, D].

    Returns
    -------
    dict
        {"cosine_similarity": float, "euclidean_distance": float}
    """
    vm_norm = F.normalize(vm_latents, dim=-1)
    vlm_norm = F.normalize(vlm_features, dim=-1)
    cosine_sim = (vm_norm * vlm_norm).sum(dim=-1).mean()
    euclidean_dist = (vm_latents - vlm_features).norm(dim=-1).mean()
    return {
        "cosine_similarity": float(cosine_sim.item()),
        "euclidean_distance": float(euclidean_dist.item()),
    }
