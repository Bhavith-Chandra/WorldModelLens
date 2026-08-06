"""Semantic probes using DINO/CLIP features for world model interpretability."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from transformers import CLIPModel, CLIPProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from torchvision import models

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SemanticProbeResult:
    """Result of semantic probing."""

    concept_name: str
    alignment_score: float
    dino_alignment: float
    clip_alignment: float
    semantic_direction: Tensor


class SemanticProber:
    """Semantic probing using DINO and CLIP features."""

    def __init__(
        self,
        model_name: str = "dino_vitb16",
        device: Optional[torch.device] = None,
    ):
        """Initialize semantic prober.

        Args:
            model_name: Model name ('dino_vitb16', 'dino_vits8', 'clip_vitb32').
            device: Device for computations.
        """
        self.device = device or _get_device()
        self.model_name = model_name
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load the feature extraction model."""
        if self.model is not None:
            return

        if "dino" in self.model_name.lower() and TORCHVISION_AVAILABLE:
            self.model = models.dino_vitb16(pretrained=True)
            self.model.eval()
            self.model.to(self.device)

        elif "clip" in self.model_name.lower() and TRANSFORMERS_AVAILABLE:
            model_id = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model.eval()
            self.model.to(self.device)

        else:
            raise ValueError(
                f"Model {self.model_name} not available. Install transformers or torchvision."
            )

    def extract_features(self, images: Tensor) -> Tensor:
        """Extract semantic features from images.

        Args:
            images: Image tensor [B, C, H, W].

        Returns:
            Feature tensor [B, D].
        """
        if self.model is None:
            self.load_model()

        images = images.to(self.device)

        with torch.no_grad():
            if "dino" in self.model_name.lower():
                features = self.model.get_intermediate_features(images)
                features = features[-1].flatten(1).mean(dim=1)
            elif "clip" in self.model_name.lower():
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
            else:
                features = self.model(images)

        return features

    def project_onto_dino(self, latents: Tensor, images: Optional[Tensor] = None) -> Tensor:
        """Project latent representations onto DINO feature space.

        Args:
            latents: World model latents [N, D_latent].
            images: Optional images for feature extraction.

        Returns:
            Projected features [N, D_dino].
        """
        if images is None:
            return torch.randn(len(latents), 768, device=self.device)

        dino_features = self.extract_features(images)

        if dino_features.shape[0] != latents.shape[0]:
            dino_features = dino_features[: latents.shape[0]]

        return dino_features

    def compute_alignment(
        self,
        latents: Tensor,
        concept_labels: Dict[str, Tensor],
        images: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute alignment between latents and semantic concepts.

        Args:
            latents: Latent representations [N, D].
            concept_labels: Dict of concept name -> binary labels.
            images: Optional images for DINO features.

        Returns:
            Dict mapping concept to alignment score.
        """
        if images is not None:
            dino_features = self.project_onto_dino(latents, images)
        else:
            return {concept: 0.0 for concept in concept_labels.keys()}

        latents_norm = F.normalize(latents, dim=1)
        dino_norm = F.normalize(dino_features, dim=1)

        alignment_scores = {}

        for concept_name, labels in concept_labels.items():
            if len(labels) != len(latents):
                alignment_scores[concept_name] = 0.0
                continue

            labels = labels.to(self.device)

            positive_mask = labels == 1
            negative_mask = labels == 0

            if positive_mask.sum() < 1 or negative_mask.sum() < 1:
                alignment_scores[concept_name] = 0.0
                continue

            positive_latents = latents_norm[positive_mask].mean(dim=0)
            negative_latents = latents_norm[negative_mask].mean(dim=0)

            concept_direction = positive_latents - negative_latents
            concept_direction = concept_direction / (concept_direction.norm() + 1e-8)

            similarity = (latents_norm @ concept_direction).abs().mean()
            alignment_scores[concept_name] = float(similarity.item())

        return alignment_scores

    def find_semantic_directions(
        self,
        latents: Tensor,
        labels: Tensor,
        n_directions: int = 10,
    ) -> List[Tensor]:
        """Find semantic directions in latent space.

        Args:
            latents: Latent tensor [N, D].
            labels: Concept labels [N].
            n_directions: Number of directions to find.

        Returns:
            List of semantic direction vectors.
        """
        unique_labels = torch.unique(labels)

        directions = []

        for label in unique_labels:
            mask = labels == label
            if mask.sum() < 2:
                continue

            pos_mean = latents[mask].mean(dim=0)
            neg_mean = latents[~mask].mean(dim=0)

            direction = pos_mean - neg_mean
            direction = direction / (direction.norm() + 1e-8)

            directions.append(direction)

        return directions[:n_directions]


class CLIPTextProber:
    """Use CLIP text features for concept alignment."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize CLIP prober."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for CLIPTextProber")

        self.device = device or _get_device()
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load CLIP model."""
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        self.model.to(self.device)

    def encode_text(self, texts: List[str]) -> Tensor:
        """Encode text prompts to features.

        Args:
            texts: List of text strings.

        Returns:
            Text features [N, D].
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

        return features

    def compute_text_alignment(
        self,
        latents: Tensor,
        concept_prompts: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Compute alignment between latents and text concepts.

        Args:
            latents: Latent tensor [N, D].
            concept_prompts: Dict mapping concept to text prompts.

        Returns:
            Dict mapping concept to alignment score.
        """
        latents_norm = F.normalize(latents, dim=1)

        alignment_scores = {}

        for concept_name, prompts in concept_prompts.items():
            text_features = self.encode_text(prompts)
            text_features = F.normalize(text_features, dim=1)

            text_mean = text_features.mean(dim=0)
            text_mean = text_mean / (text_mean.norm() + 1e-8)

            similarity = (latents_norm @ text_mean).abs().mean()
            alignment_scores[concept_name] = float(similarity.item())

        return alignment_scores

    def zero_shot_classify(
        self,
        latents: Tensor,
        class_prompts: List[str],
    ) -> Tuple[Tensor, Tensor]:
        """Zero-shot classification of latents using text prompts.

        Args:
            latents: Latent tensor [N, D].
            class_prompts: List of class prompt strings.

        Returns:
            Tuple of (predicted_class, probabilities).
        """
        text_features = self.encode_text(class_prompts)
        text_features = F.normalize(text_features, dim=1)

        latents_norm = F.normalize(latents, dim=1)

        similarities = latents_norm @ text_features.T

        probs = F.softmax(similarities, dim=-1)
        pred_class = probs.argmax(dim=-1)

        return pred_class, probs
