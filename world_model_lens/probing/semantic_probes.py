"""Semantic probes using DINO/CLIP features for world model interpretability."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
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


@dataclass
class PatchTextAlignmentResult:
    """Per-patch CLIP text alignment for one I-JEPA component."""

    component: str
    patch_ids: List[int]
    concept_scores: Dict[str, np.ndarray]
    positive_cosines: Dict[str, np.ndarray]
    negative_cosines: Dict[str, np.ndarray]


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

        if "dino" in self.model_name.lower():
            try:
                self.model = torch.hub.load(
                    "facebookresearch/dino:main", self.model_name, pretrained=True
                )
            except Exception as exc:
                raise ImportError(
                    f"Could not load DINO model '{self.model_name}' via torch.hub. "
                    f"Ensure internet access is available. Original error: {exc}"
                ) from exc
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
                features = self.model(images)
                if features.dim() == 3:
                    features = features[:, 0]  # [CLS] token
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

    def _extract_patch_clip_features(
        self,
        raw_image: Any,
        patch_ids: List[int],
        grid_size: int,
        image_size: int = 224,
    ) -> Tensor:
        """Crop each patch, resize to 224×224, return CLIP image embeddings [N, 512]."""
        from PIL import Image as _PILImage

        patch_extent = image_size // grid_size
        raw_224 = raw_image.resize((image_size, image_size))
        features = []
        for patch_id in patch_ids:
            row, col = divmod(int(patch_id), grid_size)
            x, y = col * patch_extent, row * patch_extent
            patch_crop = raw_224.crop((x, y, x + patch_extent, y + patch_extent))
            patch_resized = patch_crop.resize((224, 224), _PILImage.BILINEAR)
            inputs = self.processor(images=patch_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = self.model.get_image_features(**inputs)
            features.append(feat.squeeze(0).cpu())
        return torch.stack(features)  # [N, 512]

    def _train_projector(self, latents: Tensor, clip_features: Tensor, epochs: int = 300) -> "torch.nn.Module":
        """Train a linear projector: latent_dim → clip_dim."""
        from world_model_lens.probing.crossmodal import CrossModalProjector

        latents = latents.float().to(self.device)
        clip_features = F.normalize(clip_features.float().to(self.device), dim=-1)
        projector = CrossModalProjector(latents.shape[-1], clip_features.shape[-1]).to(self.device)
        optimizer = torch.optim.Adam(projector.parameters(), lr=1e-3)
        n = latents.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, 256):
                idx = perm[start: start + 256]
                loss = F.mse_loss(projector(latents[idx]), clip_features[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        projector.eval()
        return projector

    def compute_patch_text_alignment(
        self,
        latents: Tensor,
        concept_prompts: Dict[str, Tuple[str, str]],
        patch_ids: Optional[List[int]] = None,
        component_name: str = "",
        raw_image: Optional[Any] = None,
        grid_size: int = 14,
    ) -> PatchTextAlignmentResult:
        """Compute per-patch cosine similarities to CLIP text prompts.

        When raw_image is provided a CrossModalProjector is trained to map
        I-JEPA latents into CLIP image space before comparing with text
        embeddings.  Without raw_image latents must already be in CLIP space.
        """
        patch_ids = patch_ids or list(range(latents.shape[0]))

        if raw_image is not None:
            clip_patch_features = self._extract_patch_clip_features(
                raw_image, patch_ids, grid_size
            )
            projector = self._train_projector(latents, clip_patch_features)
            with torch.no_grad():
                latents_proj = projector(latents.float().to(self.device)).cpu()
        else:
            d_latent = latents.shape[-1]
            d_text = self.encode_text(["test"]).shape[-1]
            if d_latent != d_text:
                raise ValueError(
                    f"Latent dim ({d_latent}) != CLIP text dim ({d_text}). "
                    "Pass raw_image and grid_size so a projector can be trained."
                )
            latents_proj = F.normalize(latents.float(), dim=1)

        latents_norm = F.normalize(latents_proj, dim=1)

        concept_scores: Dict[str, np.ndarray] = {}
        positive_cosines: Dict[str, np.ndarray] = {}
        negative_cosines: Dict[str, np.ndarray] = {}

        for concept_name, prompts in concept_prompts.items():
            if len(prompts) < 2:
                raise ValueError(
                    f"Concept '{concept_name}' must provide positive and negative prompts."
                )

            text_features = self.encode_text([prompts[0], prompts[1]])
            text_features = F.normalize(text_features, dim=1).cpu()

            pos_scores = (latents_norm @ text_features[0]).detach().cpu().numpy()
            neg_scores = (latents_norm @ text_features[1]).detach().cpu().numpy()
            concept_scores[concept_name] = pos_scores - neg_scores
            positive_cosines[concept_name] = pos_scores
            negative_cosines[concept_name] = neg_scores

        return PatchTextAlignmentResult(
            component=component_name,
            patch_ids=list(patch_ids),
            concept_scores=concept_scores,
            positive_cosines=positive_cosines,
            negative_cosines=negative_cosines,
        )


# ---------------------------------------------------------------------------
# I-JEPA specific: patch-level CLIP alignment
# ---------------------------------------------------------------------------


@dataclass
class SemanticAlignmentResult:
    """Result of CLIP semantic alignment for one I-JEPA component."""

    component: str
    r2_projection: float
    concept_text_alignments: Dict[str, float]
    projection_loss: Optional[float] = None


class IJEPASemanticAligner:
    """Align I-JEPA patch latents to CLIP space and measure concept-text alignment.

    Pipeline:
    1. Crop each patch from the raw image, extract CLIP image features per patch.
    2. Train a CrossModalProjector: I-JEPA latents (192D) → CLIP image space (512D).
    3. For each spatial concept, compute the I-JEPA probe direction and project it.
    4. Measure cosine similarity with the CLIP text direction for that concept.
    """

    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, device: Optional[torch.device] = None) -> None:
        from world_model_lens.probing.crossmodal import CrossModalProber

        self.device = device or _get_device()
        self.prober = CrossModalProber(device=self.device, clip_model_name=self.CLIP_MODEL)

    def extract_patch_clip_features(
        self,
        raw_image: Any,
        patch_ids: List[int],
        grid_size: int,
        image_size: int = 224,
    ) -> Tensor:
        """Crop each patch region, resize to 224×224, return CLIP image embeddings [N, 512]."""
        from PIL import Image as _PILImage

        self.prober._load_clip()
        processor = self.prober._clip_processor
        model = self.prober._clip_model

        patch_extent = image_size // grid_size
        raw_224 = raw_image.resize((image_size, image_size))
        features = []

        for patch_id in patch_ids:
            row, col = divmod(int(patch_id), grid_size)
            x, y = col * patch_extent, row * patch_extent
            patch_crop = raw_224.crop((x, y, x + patch_extent, y + patch_extent))
            patch_resized = patch_crop.resize((224, 224), _PILImage.BILINEAR)

            inputs = processor(images=patch_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
            features.append(feat.squeeze(0).cpu())

        return torch.stack(features)  # [N, 512]

    def _compute_r2(self, latents: Tensor, clip_features: Tensor, projector: Any) -> float:
        """R² of trained projector on the training patches."""
        with torch.no_grad():
            predicted = projector(latents.to(self.device)).cpu()
        target = F.normalize(clip_features, dim=-1)
        ss_res = ((predicted - target) ** 2).sum()
        ss_tot = ((target - target.mean(0)) ** 2).sum()
        return float(1.0 - (ss_res / (ss_tot + 1e-8)).item())

    def run(
        self,
        raw_image: Any,
        activations: Tensor,
        patch_ids: List[int],
        grid_size: int,
        concept_labels: Dict[str, np.ndarray],
        concept_prompts: Dict[str, Tuple[str, str]],
        component_name: str,
        projector_epochs: int = 300,
    ) -> SemanticAlignmentResult:
        """Run full semantic alignment pipeline for one I-JEPA component.

        Args:
            raw_image: PIL Image of the input.
            activations: Patch latents [N, D_latent].
            patch_ids: Patch indices corresponding to each row of activations.
            grid_size: Spatial grid side length.
            concept_labels: Dict mapping concept name → integer label array [N].
            concept_prompts: Dict mapping concept name → (positive_text, negative_text).
            component_name: Label for this component (used in result).
            projector_epochs: Adam epochs for CrossModalProjector training.

        Returns:
            SemanticAlignmentResult with R² and per-concept CLIP alignment scores.
        """
        clip_features = self.extract_patch_clip_features(raw_image, patch_ids, grid_size)

        projector = self.prober.train_projector(
            activations.float(),
            clip_features,
            epochs=projector_epochs,
            lr=1e-3,
        )
        r2 = self._compute_r2(activations.float(), clip_features, projector)

        acts_np = activations.cpu().float().numpy()
        concept_text_alignments: Dict[str, float] = {}

        for concept_name, (pos_prompt, neg_prompt) in concept_prompts.items():
            labels = concept_labels.get(concept_name)
            if labels is None:
                continue

            pos_mask = labels == 1
            neg_mask = labels == 0
            if pos_mask.sum() < 2 or neg_mask.sum() < 2:
                continue

            # Compute I-JEPA concept direction in latent space
            concept_dir = torch.from_numpy(
                acts_np[pos_mask].mean(0) - acts_np[neg_mask].mean(0)
            ).float()
            concept_dir = F.normalize(concept_dir, dim=0)

            # Project direction into CLIP space
            with torch.no_grad():
                concept_dir_clip = projector(
                    concept_dir.unsqueeze(0).to(self.device)
                ).squeeze(0).cpu()

            # CLIP text direction
            text_feats = self.prober.encode_text([pos_prompt, neg_prompt]).cpu()  # [2, 512]
            text_dir = F.normalize(text_feats[0] - text_feats[1], dim=0)

            concept_text_alignments[concept_name] = float(
                torch.dot(concept_dir_clip, text_dir).item()
            )

        return SemanticAlignmentResult(
            component=component_name,
            r2_projection=r2,
            concept_text_alignments=concept_text_alignments,
        )
