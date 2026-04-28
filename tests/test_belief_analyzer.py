"""Tests for the analysis module."""

import torch
import pytest

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.analysis.uncertainty import UncertaintyQuantifier
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from tests.conftest import SimpleTestAdapter


class TestBeliefAnalyzer:
    """Tests for BeliefAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a belief analyzer."""
        config = WorldModelConfig(d_h=32, d_action=4, d_obs=64)
        adapter = SimpleTestAdapter(config)
        wm = HookedWorldModel(adapter=adapter, config=config)
        return BeliefAnalyzer(wm)

    def test_analyzer_creation(self, analyzer):
        """Test analyzer creates correctly."""
        assert analyzer is not None

    def test_capabilities_property(self, analyzer):
        """Test capabilities property."""
        caps = analyzer.capabilities
        assert caps is not None

    def test_has_surprise_timeline_method(self, analyzer):
        """Test surprise_timeline method exists."""
        assert hasattr(analyzer, "surprise_timeline")
        assert callable(analyzer.surprise_timeline)

    def test_has_concept_search_method(self, analyzer):
        """Test concept_search method exists."""
        assert hasattr(analyzer, "concept_search")
        assert callable(analyzer.concept_search)

    def test_has_latent_saliency_method(self, analyzer):
        """Test latent_saliency method exists."""
        assert hasattr(analyzer, "latent_saliency")
        assert callable(analyzer.latent_saliency)

    def test_has_disentanglement_score_method(self, analyzer):
        """Test disentanglement_score method exists."""
        assert hasattr(analyzer, "disentanglement_score")
        assert callable(analyzer.disentanglement_score)

    def test_ijepa_surprise_timeline(self):
        """Test surprise_timeline with I-JEPA cosine distance."""

        config = WorldModelConfig(backend="ijepa", d_embed=128, predictor_embed_dim=192)
        adapter = IJEPAAdapter(config)
        wm = HookedWorldModel(adapter=adapter, config=config)
        analyzer = BeliefAnalyzer(wm)

        # Create dummy input for IJEPA (images)
        batch_size = 2
        obs = torch.randn(batch_size, 3, 224, 224)  # RGB images

        # Run with cache
        with torch.no_grad():
            _, cache = wm.run_with_cache(obs)

        result = analyzer.surprise_timeline(cache)

        assert result.kl_sequence is not None
        assert len(result.kl_sequence) >= 1  # At least one timestep
        assert result.mean_surprise >= 0  # cosine distance is 0-2

    def test_uncertainty_fit_and_score_ood(self):
        """Test OOD fitting and scoring."""

        quantifier = UncertaintyQuantifier()

        # Mock in-distribution latents
        id_latents = [torch.randn(4, 32) for _ in range(10)]  # 10 samples, 4 patches, 32 dim

        quantifier.fit_latent_distribution(id_latents)
        assert quantifier.is_ood_fitted

        # Mock OOD latents
        ood_latents = [torch.randn(4, 32) for _ in range(5)]

        scores = quantifier.score_ood_latents(ood_latents)
        assert scores.shape == (5,)
        assert (scores >= 0).all()
