"""Tests for LayerCKAAnalyzer."""

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.analysis.layer_cka import LayerCKAAnalyzer, LayerCKAResult
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.types import WorldModelFamily


class MockVisionAdapter(IJEPAAdapter):
    """Simplified vision adapter for testing."""

    def __init__(self, config):
        # Just use the parent constructor with config
        super().__init__(config)


@pytest.fixture
def vision_config():
    """Config for vision model testing."""
    return WorldModelConfig(
        world_model_family="ijepa",
        img_size=32,
        d_embed=64,
        n_heads=4,
        n_layers=3,
        patch_size=8,
        backend="ijepa",
        encoder_type="vit",
    )


@pytest.fixture
def vision_adapter(vision_config):
    """Vision adapter for testing."""
    return MockVisionAdapter(vision_config)


@pytest.fixture
def hooked_vision_wm(vision_adapter, vision_config):
    """Hooked world model with vision encoder."""
    return HookedWorldModel(adapter=vision_adapter, config=vision_config)


@pytest.fixture
def layer_cka_analyzer(hooked_vision_wm):
    """LayerCKAAnalyzer instance."""
    return LayerCKAAnalyzer(hooked_vision_wm)


@pytest.fixture
def fake_images():
    """Fake images for testing."""
    return torch.randn(2, 3, 32, 32)  # Batch of 2 small images


class TestLayerCKAAnalyzer:
    """Tests for LayerCKAAnalyzer class."""

    def test_analyzer_creation(self, layer_cka_analyzer):
        """Test analyzer creates correctly."""
        assert layer_cka_analyzer is not None
        assert hasattr(layer_cka_analyzer, "wm")
        assert hasattr(layer_cka_analyzer, "_centered_kernel_alignment")

    def test_cka_computation(self, layer_cka_analyzer):
        """Test CKA computation between two tensors."""
        a = torch.randn(10, 32)
        b = torch.randn(10, 32)

        cka = layer_cka_analyzer._centered_kernel_alignment(a, b)

        assert isinstance(cka, float)
        assert 0.0 <= cka <= 1.0

        # Test identical tensors should have CKA = 1
        cka_identical = layer_cka_analyzer._centered_kernel_alignment(a, a)
        assert abs(cka_identical - 1.0) < 1e-6

    def test_layer_extraction(self, layer_cka_analyzer, fake_images):
        """Test layer representation extraction."""
        layer_reps = layer_cka_analyzer._extract_layer_representations(
            fake_images, "target_encoder.blocks.{}.hook_resid_post", max_layers=2
        )

        assert isinstance(layer_reps, dict)
        assert len(layer_reps) == 2  # Should extract 2 layers

        for key, tensor in layer_reps.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape[0] == fake_images.shape[0]  # Batch size
            assert tensor.shape[1] == 16  # num_patches
            assert tensor.shape[2] == 64  # embed_dim

    def test_analyze_layers_basic(self, layer_cka_analyzer, fake_images):
        """Test basic layer analysis."""
        result = layer_cka_analyzer.analyze_layers(fake_images, max_layers=2)

        assert isinstance(result, LayerCKAResult)
        assert len(result.layer_names) == 2
        assert result.cka_matrix.shape == (1, 16)  # 1 transition, 16 patches
        assert result.avg_cka_per_layer.shape == (1,)  # 1 transition
        assert result.patch_convergence.shape == (16,)  # 16 patches
        assert isinstance(result.semantic_convergence_score, float)

    def test_analyze_layers_all_layers(self, layer_cka_analyzer, fake_images):
        """Test analysis with all available layers."""
        result = layer_cka_analyzer.analyze_layers(fake_images)

        # Should have 3 layers (depth=3)
        assert len(result.layer_names) == 3
        assert result.cka_matrix.shape == (2, 16)  # 2 transitions, 16 patches
        assert result.avg_cka_per_layer.shape == (2,)  # 2 transitions

    def test_patch_convergence(self, layer_cka_analyzer, fake_images):
        """Test patch convergence computation."""
        result = layer_cka_analyzer.analyze_layers(fake_images, max_layers=3)

        convergence = layer_cka_analyzer._compute_patch_convergence(result.cka_matrix)

        assert convergence.shape == (16,)  # 16 patches
        assert np.all(convergence >= 0)  # Should be non-negative
        assert np.all(convergence <= 1 + 1e-6)  # Should be <= 1 (allow small tolerance)

    def test_semantic_convergence(self, layer_cka_analyzer, fake_images):
        """Test semantic convergence scoring."""
        # Create fake CKA values that increase (converging)
        avg_cka = np.array([0.1, 0.3, 0.6])
        score = layer_cka_analyzer._compute_semantic_convergence(avg_cka)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        # Test with decreasing values (diverging)
        avg_cka_decreasing = np.array([0.6, 0.3, 0.1])
        score_decreasing = layer_cka_analyzer._compute_semantic_convergence(avg_cka_decreasing)

        # Should be lower than converging case
        assert score_decreasing < score

    def test_plot_convergence(self, layer_cka_analyzer, fake_images):
        """Test convergence plotting."""
        result = layer_cka_analyzer.analyze_layers(fake_images, max_layers=3)

        fig = layer_cka_analyzer.plot_convergence(result)

        assert fig is not None
        # Should have 2x2 subplots
        assert len(fig.axes) in (4, 5)  # May have colorbar

        plt.close(fig)  # Clean up

    def test_plot_patch_embeddings_pca_2d(self, layer_cka_analyzer, fake_images):
        """Test PCA plotting in 2D."""
        result = layer_cka_analyzer.analyze_layers(fake_images, max_layers=3)

        fig = layer_cka_analyzer.plot_patch_embeddings_pca(result, fake_images, n_components=2)

        assert fig is not None
        # Should have 1x2 subplots for 2D
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_plot_patch_embeddings_pca_3d(self, layer_cka_analyzer, fake_images):
        """Test PCA plotting in 3D."""
        result = layer_cka_analyzer.analyze_layers(fake_images, max_layers=3)

        fig = layer_cka_analyzer.plot_patch_embeddings_pca(result, fake_images, n_components=3)

        assert fig is not None
        # Should have 1 subplot for 3D
        assert len(fig.axes) in (1, 2)  # May have 2 for first+last layers

        plt.close(fig)

    def test_invalid_n_components(self, layer_cka_analyzer, fake_images):
        """Test error handling for invalid n_components."""
        result = layer_cka_analyzer.analyze_layers(fake_images, max_layers=2)

        with pytest.raises((ValueError, RuntimeError), match="n_components must be 2 or 3|memory"):
            layer_cka_analyzer.plot_patch_embeddings_pca(result, fake_images, n_components=4)

    def test_insufficient_layers(self, layer_cka_analyzer, fake_images):
        """Test error when only one layer is available."""
        with pytest.raises(ValueError, match="Need at least 2 layers"):
            layer_cka_analyzer.analyze_layers(fake_images, max_layers=1)

    def test_missing_encoder(self):
        """Test error when encoder is not found."""
        config = WorldModelConfig(d_h=32, d_action=4, d_obs=64)
        from tests.conftest import SimpleTestAdapter

        adapter = SimpleTestAdapter(config)
        wm = HookedWorldModel(adapter=adapter, config=config)
        analyzer = LayerCKAAnalyzer(wm)

        fake_images = torch.randn(1, 3, 32, 32)

        with pytest.raises(ValueError, match="Could not find encoder"):
            analyzer.analyze_layers(fake_images)

    def test_missing_blocks(self, hooked_vision_wm):
        """Test error when encoder has no blocks."""
        # Temporarily remove blocks
        original_blocks = hooked_vision_wm.adapter.context_encoder.blocks
        hooked_vision_wm.adapter.context_encoder.blocks = None

        analyzer = LayerCKAAnalyzer(hooked_vision_wm)
        fake_images = torch.randn(1, 3, 32, 32)

        try:
            with pytest.raises(ValueError, match="Encoder does not have transformer blocks"):
                analyzer.analyze_layers(fake_images)
        finally:
            # Restore blocks
            hooked_vision_wm.adapter.context_encoder.blocks = original_blocks


class TestLayerCKAResult:
    """Tests for LayerCKAResult dataclass."""

    def test_result_creation(self):
        """Test result dataclass creation."""
        layer_names = ["layer_0", "layer_1", "layer_2"]
        cka_matrix = np.random.rand(2, 16)
        avg_cka = np.mean(cka_matrix, axis=1)
        patch_conv = np.random.rand(16)
        semantic_score = 0.75

        result = LayerCKAResult(
            layer_names=layer_names,
            cka_matrix=cka_matrix,
            avg_cka_per_layer=avg_cka,
            patch_convergence=patch_conv,
            semantic_convergence_score=semantic_score,
        )

        assert result.layer_names == layer_names
        assert np.array_equal(result.cka_matrix, cka_matrix)
        assert np.array_equal(result.avg_cka_per_layer, avg_cka)
        assert np.array_equal(result.patch_convergence, patch_conv)
        assert result.semantic_convergence_score == semantic_score

    def test_result_attributes(self):
        """Test result has all expected attributes."""
        layer_names = ["layer_0", "layer_1"]
        cka_matrix = np.random.rand(1, 16)
        avg_cka = np.mean(cka_matrix, axis=1)
        patch_conv = np.random.rand(16)
        semantic_score = 0.5

        result = LayerCKAResult(
            layer_names=layer_names,
            cka_matrix=cka_matrix,
            avg_cka_per_layer=avg_cka,
            patch_convergence=patch_conv,
            semantic_convergence_score=semantic_score,
        )

        assert hasattr(result, "layer_names")
        assert hasattr(result, "cka_matrix")
        assert hasattr(result, "avg_cka_per_layer")
        assert hasattr(result, "patch_convergence")
        assert hasattr(result, "semantic_convergence_score")
