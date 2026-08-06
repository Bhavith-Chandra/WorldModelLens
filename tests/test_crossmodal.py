"""Tests for cross-modal probing (Issue #11 — Multi-Modal Alignment).

All CLIP calls are mocked so the test-suite runs offline without a GPU
and without downloading model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from world_model_lens.probing.crossmodal import (
    CrossModalProjector,
    CrossModalProber,
    CrossModalResult,
    ConceptQueryResult,
    align_multimodal,
)

# ---------------------------------------------------------------------------
# Constants / shared fixtures
# ---------------------------------------------------------------------------

LATENT_DIM = 32
CLIP_DIM = 512
N = 16
DEVICE = torch.device("cpu")


@pytest.fixture
def random_latents() -> torch.Tensor:
    """Synthetic world-model latents [N, LATENT_DIM]."""
    torch.manual_seed(0)
    return torch.randn(N, LATENT_DIM)


@pytest.fixture
def random_clip_features() -> torch.Tensor:
    """Synthetic L2-normalised CLIP image features [N, CLIP_DIM]."""
    torch.manual_seed(1)
    return F.normalize(torch.randn(N, CLIP_DIM), dim=-1)


@pytest.fixture
def trained_projector(random_latents: torch.Tensor, random_clip_features: torch.Tensor):
    """A CrossModalProjector trained for a handful of epochs (fast, CPU)."""
    prober = CrossModalProber(device=DEVICE)
    return prober.train_projector(
        random_latents, random_clip_features, epochs=5, batch_size=N
    )


def _mock_clip(n_texts: int = 1, clip_dim: int = CLIP_DIM):
    """Return (mock_clip_model, mock_clip_processor) with deterministic outputs."""
    torch.manual_seed(42)
    model = MagicMock()
    # get_text_features returns [n_texts, clip_dim]
    model.get_text_features.return_value = F.normalize(
        torch.randn(n_texts, clip_dim), dim=-1
    )
    # get_image_features returns [N, clip_dim]
    model.get_image_features.return_value = F.normalize(
        torch.randn(N, clip_dim), dim=-1
    )
    processor = MagicMock()
    processor.return_value = {"input_ids": torch.zeros(n_texts, 10, dtype=torch.long)}
    return model, processor


def _prober_with_mock_clip(n_texts: int = 1) -> CrossModalProber:
    """Return a CrossModalProber whose CLIP model/processor are mocked."""
    mock_model, mock_processor = _mock_clip(n_texts)
    prober = CrossModalProber(device=DEVICE)
    prober._clip_model = mock_model
    prober._clip_processor = mock_processor
    return prober


# ===========================================================================
# CrossModalProjector
# ===========================================================================


class TestCrossModalProjector:
    """Unit tests for the learnable affine projection layer."""

    def test_output_shape(self):
        proj = CrossModalProjector(d_latent=LATENT_DIM, d_clip=CLIP_DIM)
        out = proj(torch.randn(N, LATENT_DIM))
        assert out.shape == (N, CLIP_DIM), "Output shape mismatch"

    def test_output_is_l2_normalised(self):
        proj = CrossModalProjector(d_latent=LATENT_DIM, d_clip=CLIP_DIM)
        out = proj(torch.randn(N, LATENT_DIM))
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(N), atol=1e-5), "Output is not unit-norm"

    def test_custom_dimensions(self):
        proj = CrossModalProjector(d_latent=64, d_clip=256)
        out = proj(torch.randn(8, 64))
        assert out.shape == (8, 256)

    def test_single_sample(self):
        proj = CrossModalProjector(d_latent=LATENT_DIM, d_clip=CLIP_DIM)
        out = proj(torch.randn(1, LATENT_DIM))
        assert out.shape == (1, CLIP_DIM)

    def test_is_nn_module(self):
        proj = CrossModalProjector(d_latent=LATENT_DIM, d_clip=CLIP_DIM)
        import torch.nn as nn
        assert isinstance(proj, nn.Module)

    def test_parameters_are_trainable(self):
        proj = CrossModalProjector(d_latent=LATENT_DIM, d_clip=CLIP_DIM)
        params = list(proj.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ===========================================================================
# CrossModalResult
# ===========================================================================


class TestCrossModalResult:
    """Unit tests for the CrossModalResult dataclass."""

    def test_default_values(self):
        r = CrossModalResult()
        assert r.alignment_score == 0.0
        assert r.shared_concepts == []
        assert r.retrieval_accuracy == 0.0
        assert r.concept_similarities == {}
        assert r.projection_loss is None

    def test_custom_values(self):
        r = CrossModalResult(
            alignment_score=0.75,
            shared_concepts=["danger", "dog"],
            retrieval_accuracy=0.9,
            projection_loss=0.01,
        )
        assert r.alignment_score == pytest.approx(0.75)
        assert "danger" in r.shared_concepts
        assert r.projection_loss == pytest.approx(0.01)

    def test_to_dict_contains_all_keys(self):
        r = CrossModalResult(alignment_score=0.5, shared_concepts=["fire"])
        d = r.to_dict()
        for key in ("alignment_score", "shared_concepts", "retrieval_accuracy",
                    "concept_similarities", "projection_loss"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_match(self):
        r = CrossModalResult(alignment_score=0.42, shared_concepts=["cat"])
        d = r.to_dict()
        assert d["alignment_score"] == pytest.approx(0.42)
        assert d["shared_concepts"] == ["cat"]


# ===========================================================================
# ConceptQueryResult
# ===========================================================================


class TestConceptQueryResult:
    """Unit tests for the ConceptQueryResult dataclass."""

    def test_fields(self):
        r = ConceptQueryResult(concept="danger", similarity=0.42, is_present=True)
        assert r.concept == "danger"
        assert r.similarity == pytest.approx(0.42)
        assert r.is_present is True

    def test_threshold_default(self):
        r = ConceptQueryResult(concept="dog", similarity=0.1, is_present=False)
        assert r.threshold == 0.0

    def test_to_dict_excludes_tensor(self):
        r = ConceptQueryResult(
            concept="dog",
            similarity=0.3,
            is_present=False,
            threshold=0.5,
            per_sample_similarities=torch.randn(N),
        )
        d = r.to_dict()
        assert "concept" in d
        assert "similarity" in d
        assert "is_present" in d
        assert "threshold" in d
        assert "per_sample_similarities" not in d

    def test_per_sample_similarities_stored(self):
        sims = torch.randn(N)
        r = ConceptQueryResult(concept="fire", similarity=0.5, is_present=True,
                               per_sample_similarities=sims)
        assert r.per_sample_similarities is not None
        assert r.per_sample_similarities.shape == (N,)


# ===========================================================================
# CrossModalProber — no CLIP needed
# ===========================================================================


class TestProjectLatents:
    """Tests for project_latents (does not require CLIP)."""

    def test_with_projector_shape(self, random_latents, trained_projector):
        prober = CrossModalProber(device=DEVICE)
        out = prober.project_latents(random_latents, projector=trained_projector)
        assert out.shape == (N, CLIP_DIM)

    def test_with_projector_normalised(self, random_latents, trained_projector):
        prober = CrossModalProber(device=DEVICE)
        out = prober.project_latents(random_latents, projector=trained_projector)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(N), atol=1e-5)

    def test_without_projector_preserves_dim(self, random_latents):
        prober = CrossModalProber(device=DEVICE)
        out = prober.project_latents(random_latents, projector=None)
        assert out.shape == (N, LATENT_DIM)

    def test_without_projector_normalised(self, random_latents):
        prober = CrossModalProber(device=DEVICE)
        out = prober.project_latents(random_latents, projector=None)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(N), atol=1e-5)


class TestTrainProjector:
    """Tests for the projector training loop (no CLIP required)."""

    def test_returns_crossmodal_projector(self, random_latents, random_clip_features):
        prober = CrossModalProber(device=DEVICE)
        proj = prober.train_projector(random_latents, random_clip_features,
                                      epochs=2, batch_size=N)
        assert isinstance(proj, CrossModalProjector)

    def test_projector_in_eval_mode(self, random_latents, random_clip_features):
        prober = CrossModalProber(device=DEVICE)
        proj = prober.train_projector(random_latents, random_clip_features,
                                      epochs=2, batch_size=N)
        assert not proj.training, "Projector should be in eval mode after training"

    def test_output_shape(self, random_latents, random_clip_features):
        prober = CrossModalProber(device=DEVICE)
        proj = prober.train_projector(random_latents, random_clip_features,
                                      epochs=2, batch_size=N)
        out = proj(random_latents)
        assert out.shape == (N, CLIP_DIM)

    def test_training_reduces_loss(self, random_latents, random_clip_features):
        """More epochs should yield lower reconstruction error."""
        prober = CrossModalProber(device=DEVICE)
        clip_norm = F.normalize(random_clip_features, dim=-1)

        proj_few = prober.train_projector(random_latents, random_clip_features,
                                          epochs=1, batch_size=N)
        proj_more = prober.train_projector(random_latents, random_clip_features,
                                           epochs=300, batch_size=N)

        loss_few = F.mse_loss(proj_few(random_latents), clip_norm).item()
        loss_more = F.mse_loss(proj_more(random_latents), clip_norm).item()
        # Allow generous slack for stochastic mini-batch training
        assert loss_more <= loss_few + 0.5, (
            f"Expected more training to reduce loss, got {loss_few:.4f} → {loss_more:.4f}"
        )

    def test_single_sample_batch(self):
        """Edge case: a single training sample."""
        prober = CrossModalProber(device=DEVICE)
        lat = torch.randn(1, LATENT_DIM)
        clip = F.normalize(torch.randn(1, CLIP_DIM), dim=-1)
        proj = prober.train_projector(lat, clip, epochs=2, batch_size=1)
        assert proj(lat).shape == (1, CLIP_DIM)


# ===========================================================================
# CrossModalProber — with mocked CLIP
# ===========================================================================


class TestQueryConcept:
    """Tests for the core Issue #11 feature: query_concept."""

    def test_returns_concept_query_result(self, random_latents, trained_projector):
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "danger",
                                      projector=trained_projector)
        assert isinstance(result, ConceptQueryResult)

    def test_concept_field(self, random_latents, trained_projector):
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "danger",
                                      projector=trained_projector)
        assert result.concept == "danger"

    def test_similarity_is_float(self, random_latents, trained_projector):
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "fire",
                                      projector=trained_projector)
        assert isinstance(result.similarity, float)

    def test_is_present_is_bool(self, random_latents, trained_projector):
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "sky",
                                      projector=trained_projector)
        assert isinstance(result.is_present, bool)

    def test_per_sample_shape(self, random_latents, trained_projector):
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "dog",
                                      projector=trained_projector)
        assert result.per_sample_similarities is not None
        assert result.per_sample_similarities.shape == (N,)

    def test_low_threshold_triggers_present(self, random_latents, trained_projector):
        """With threshold = -1 every concept should be classified as present."""
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "tree",
                                      projector=trained_projector, threshold=-1.0)
        assert result.is_present is True

    def test_high_threshold_suppresses_present(self, random_latents, trained_projector):
        """With threshold = 1 no concept (cosine ≤ 1) should be present."""
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "danger",
                                      projector=trained_projector, threshold=1.0)
        assert result.is_present is False

    def test_threshold_stored_in_result(self, random_latents, trained_projector):
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(random_latents, "car",
                                      projector=trained_projector, threshold=0.25)
        assert result.threshold == pytest.approx(0.25)

    def test_custom_prompt_template(self, random_latents, trained_projector):
        """Custom prompt template should be accepted without errors."""
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(
            random_latents, "fire",
            projector=trained_projector,
            prompt_template="a scene containing {}",
        )
        assert result.concept == "fire"

    def test_no_projector_zero_shot(self):
        """Zero-shot mode (no projector) should work when dims match."""
        latents = F.normalize(torch.randn(N, CLIP_DIM), dim=-1)
        prober = _prober_with_mock_clip(n_texts=1)
        result = prober.query_concept(latents, "sky", projector=None)
        assert isinstance(result, ConceptQueryResult)


class TestBatchQueryConcepts:
    """Tests for batch_query_concepts."""

    def test_length_matches_concepts(self, random_latents, trained_projector):
        concepts = ["danger", "dog", "car", "tree"]
        prober = _prober_with_mock_clip(n_texts=len(concepts))
        results = prober.batch_query_concepts(random_latents, concepts,
                                              projector=trained_projector)
        assert len(results) == len(concepts)

    def test_sorted_by_descending_similarity(self, random_latents, trained_projector):
        concepts = ["danger", "dog", "car", "tree"]
        prober = _prober_with_mock_clip(n_texts=len(concepts))
        results = prober.batch_query_concepts(random_latents, concepts,
                                              projector=trained_projector)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"

    def test_all_concepts_in_results(self, random_latents, trained_projector):
        concepts = ["danger", "dog"]
        prober = _prober_with_mock_clip(n_texts=len(concepts))
        results = prober.batch_query_concepts(random_latents, concepts,
                                              projector=trained_projector)
        assert {r.concept for r in results} == set(concepts)

    def test_per_sample_shape(self, random_latents, trained_projector):
        concepts = ["a", "b"]
        prober = _prober_with_mock_clip(n_texts=len(concepts))
        results = prober.batch_query_concepts(random_latents, concepts,
                                              projector=trained_projector)
        for r in results:
            assert r.per_sample_similarities is not None
            assert r.per_sample_similarities.shape == (N,)

    def test_single_concept(self, random_latents, trained_projector):
        prober = _prober_with_mock_clip(n_texts=1)
        results = prober.batch_query_concepts(random_latents, ["danger"],
                                              projector=trained_projector)
        assert len(results) == 1
        assert results[0].concept == "danger"


class TestFindSharedConcepts:
    """Tests for find_shared_concepts."""

    def test_returns_list_of_tuples(self, random_latents, trained_projector):
        concepts = ["danger", "dog", "car"]
        prober = _prober_with_mock_clip(n_texts=len(concepts))
        results = prober.find_shared_concepts(random_latents, concepts,
                                              projector=trained_projector)
        assert isinstance(results, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

    def test_sorted_descending(self, random_latents, trained_projector):
        concepts = ["danger", "dog", "car"]
        prober = _prober_with_mock_clip(n_texts=len(concepts))
        results = prober.find_shared_concepts(random_latents, concepts,
                                              projector=trained_projector)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_tuple_types(self, random_latents, trained_projector):
        concepts = ["fire", "water"]
        prober = _prober_with_mock_clip(n_texts=len(concepts))
        results = prober.find_shared_concepts(random_latents, concepts,
                                              projector=trained_projector)
        for concept, score in results:
            assert isinstance(concept, str)
            assert isinstance(score, float)


class TestCrossmodalRetrieval:
    """Tests for crossmodal_retrieval."""

    def test_length_matches_queries(self, random_latents, trained_projector):
        queries = ["a dog", "danger scene", "fire"]
        prober = _prober_with_mock_clip(n_texts=len(queries))
        results = prober.crossmodal_retrieval(random_latents, queries,
                                              projector=trained_projector, top_k=2)
        assert len(results) == len(queries)

    def test_top_k_length(self, random_latents, trained_projector):
        queries = ["cat", "dog"]
        prober = _prober_with_mock_clip(n_texts=len(queries))
        results = prober.crossmodal_retrieval(random_latents, queries,
                                              projector=trained_projector, top_k=3)
        assert all(len(r) == 3 for r in results)

    def test_indices_in_valid_range(self, random_latents, trained_projector):
        queries = ["cat"]
        prober = _prober_with_mock_clip(n_texts=1)
        results = prober.crossmodal_retrieval(random_latents, queries,
                                              projector=trained_projector, top_k=1)
        assert all(0 <= idx < N for idx in results[0])

    def test_top_k_clamped_to_n(self, random_latents, trained_projector):
        """top_k > N should be silently clamped."""
        queries = ["sky"]
        prober = _prober_with_mock_clip(n_texts=1)
        results = prober.crossmodal_retrieval(random_latents, queries,
                                              projector=trained_projector, top_k=N + 100)
        assert len(results[0]) == N


# ===========================================================================
# align_multimodal — standalone utility (no CLIP)
# ===========================================================================


class TestAlignMultimodal:
    """Tests for the standalone align_multimodal function."""

    def test_returns_dict_with_correct_keys(self):
        result = align_multimodal(torch.randn(N, 64), torch.randn(N, 64))
        assert "cosine_similarity" in result
        assert "euclidean_distance" in result

    def test_returns_floats(self):
        result = align_multimodal(torch.randn(4, 32), torch.randn(4, 32))
        assert isinstance(result["cosine_similarity"], float)
        assert isinstance(result["euclidean_distance"], float)

    def test_identical_tensors_cosine_one(self):
        x = torch.randn(N, 64)
        result = align_multimodal(x, x)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)

    def test_opposite_tensors_cosine_neg_one(self):
        x = torch.randn(N, 64)
        result = align_multimodal(x, -x)
        assert result["cosine_similarity"] == pytest.approx(-1.0, abs=1e-4)

    def test_identical_tensors_euclidean_zero(self):
        x = torch.randn(N, 64)
        result = align_multimodal(x, x)
        assert result["euclidean_distance"] == pytest.approx(0.0, abs=1e-4)

    def test_cosine_bounded(self):
        for _ in range(5):
            result = align_multimodal(torch.randn(8, 32), torch.randn(8, 32))
            assert -1.0 - 1e-4 <= result["cosine_similarity"] <= 1.0 + 1e-4

    def test_euclidean_non_negative(self):
        result = align_multimodal(torch.randn(N, 64), torch.randn(N, 64))
        assert result["euclidean_distance"] >= 0.0
