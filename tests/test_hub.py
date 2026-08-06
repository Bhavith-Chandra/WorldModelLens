"""Tests for world_model_lens.hub — ModelHub, ModelInfo, WeightsDownloader.

Test strategy
-------------
All tests are offline and never touch the network. Network-dependent behaviour
(HuggingFace downloads, torch.load on a real file) is covered by mocking
``huggingface_hub.hf_hub_download`` and ``torch.load`` so the test suite runs
instantly in any environment, including CI without GPU / internet.

Network-required tests are marked with ``@pytest.mark.network`` so they can be
skipped by default and run explicitly when you have connectivity:

    pytest tests/test_hub.py                       # offline only (default)
    pytest tests/test_hub.py -m network            # network tests only
    pytest tests/test_hub.py -m "not network"      # explicitly skip network
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import warnings

from world_model_lens.backends.base_adapter import AdapterConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.backends.iris import IRISAdapter
from world_model_lens.hub.model_hub import ModelHub, ModelInfo
from world_model_lens.hub.weights_downloader import WeightsDownloader
from world_model_lens import hub


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    """Temporary directory used as cache_dir for WeightsDownloader in tests."""
    cache = tmp_path / "wml_test_cache"
    cache.mkdir()
    return cache


@pytest.fixture()
def downloader(tmp_cache: Path) -> WeightsDownloader:
    """WeightsDownloader pointed at a temporary directory (no real HF cache)."""
    return WeightsDownloader(cache_dir=str(tmp_cache))


@pytest.fixture()
def fake_checkpoint_file(tmp_path: Path) -> Path:
    """A minimal fake .pt file using the real eloialonso/iris flat format.

    The actual IRIS checkpoint is a flat OrderedDict where every parameter
    has a component prefix (``world_model.*``, ``tokenizer.*``, ``actor_critic.*``).
    This fixture replicates that structure with tiny tensors so tests run fast.
    """
    

    ckpt = {
        # world_model — transformer blocks (just one block for speed)
        "world_model.transformer.blocks.0.ln1.weight": torch.ones(512),
        "world_model.transformer.blocks.0.ln1.bias": torch.zeros(512),
        "world_model.transformer.blocks.0.ln2.weight": torch.ones(512),
        "world_model.transformer.blocks.0.ln2.bias": torch.zeros(512),
        "world_model.transformer.blocks.0.attn.key.weight": torch.zeros(512, 512),
        "world_model.transformer.blocks.0.attn.key.bias": torch.zeros(512),
        "world_model.transformer.blocks.0.attn.query.weight": torch.zeros(512, 512),
        "world_model.transformer.blocks.0.attn.query.bias": torch.zeros(512),
        "world_model.transformer.blocks.0.attn.value.weight": torch.zeros(512, 512),
        "world_model.transformer.blocks.0.attn.value.bias": torch.zeros(512),
        "world_model.transformer.blocks.0.attn.proj.weight": torch.zeros(512, 512),
        "world_model.transformer.blocks.0.attn.proj.bias": torch.zeros(512),
        # world_model — final layer norm and embeddings
        "world_model.transformer.ln_f.weight": torch.ones(512),
        "world_model.transformer.ln_f.bias": torch.zeros(512),
        "world_model.pos_emb.weight": torch.zeros(1024, 512),
        "world_model.embedder.embedding_tables.0.weight": torch.zeros(512, 512),
        "world_model.embedder.embedding_tables.1.weight": torch.zeros(3, 512),
        # world_model — prediction heads
        "world_model.head_observations.head_module.0.weight": torch.zeros(512, 512),
        "world_model.head_observations.head_module.0.bias": torch.zeros(512),
        "world_model.head_rewards.head_module.0.weight": torch.zeros(3, 512),
        "world_model.head_rewards.head_module.0.bias": torch.zeros(3),
        "world_model.head_ends.head_module.0.weight": torch.zeros(2, 512),
        "world_model.head_ends.head_module.0.bias": torch.zeros(2),
        # tokenizer — VQ-VAE encoder (abbreviated)
        "tokenizer.encoder.conv_in.weight": torch.zeros(128, 3, 3, 3),
        "tokenizer.encoder.conv_in.bias": torch.zeros(128),
        "tokenizer.embedding.weight": torch.zeros(512, 512),
        "tokenizer.pre_quant_conv.weight": torch.zeros(512, 128, 1, 1),
        "tokenizer.pre_quant_conv.bias": torch.zeros(512),
        # actor_critic (not used during load, but present in real file)
        "actor_critic.conv1.weight": torch.zeros(32, 3, 3, 3),
        "actor_critic.conv1.bias": torch.zeros(32),
    }
    path = tmp_path / "fake_iris.pt"
    torch.save(ckpt, path)
    return path


@pytest.fixture()
def fake_dreamerv3_checkpoint(tmp_path: Path) -> Path:
    """A compatible DreamerV3Adapter state dict saved to disk for hub dispatch tests."""
    cfg = AdapterConfig(
        d_h=32,
        d_obs=8,
        d_action=4,
        n_cat=4,
        n_cls=4,
        encoder_type="mlp",
    )
    adapter = DreamerV3Adapter(cfg)
    path = tmp_path / "fake_dreamerv3.pt"
    torch.save(adapter.state_dict(), path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ModelInfo dataclass
# ─────────────────────────────────────────────────────────────────────────────


def test_model_info_is_downloadable_when_ready():
    """is_downloadable returns True only when all conditions are met."""
    m = ModelInfo(
        name="test",
        backend="iris",
        environment="Atari/Test",
        coming_soon=False,
        hf_repo_id="owner/repo",
        hf_filename="model.pt",
    )
    assert m.is_downloadable is True


def test_model_info_not_downloadable_when_coming_soon():
    """is_downloadable is False when coming_soon=True."""
    m = ModelInfo(
        name="test",
        backend="dreamerv3",
        environment="Atari/Test",
        coming_soon=True,
        hf_repo_id="owner/repo",
        hf_filename="model.pt",
    )
    assert m.is_downloadable is False


def test_model_info_not_downloadable_without_hf_coords():
    """is_downloadable is False when hf_repo_id is None."""
    m = ModelInfo(
        name="test",
        backend="dreamerv3",
        environment="Atari/Test",
        coming_soon=False,
        hf_repo_id=None,
        hf_filename=None,
    )
    assert m.is_downloadable is False


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub registry
# ─────────────────────────────────────────────────────────────────────────────


def test_registry_is_non_empty():
    """Registry must contain at least one entry."""
    assert len(ModelHub._MODELS) > 0


def test_registry_has_iris_atari_entries():
    """Registry must contain the known IRIS Atari models."""
    expected = {
        "iris-atari-breakout",
        "iris-atari-pong",
        "iris-atari-seaquest",
        "iris-atari-freeway",
        "iris-atari-alien",
    }
    actual = set(ModelHub._MODELS.keys())
    assert expected.issubset(actual), f"Missing registry entries: {expected - actual}"


def test_registry_iris_entries_are_downloadable():
    """All iris-atari-* entries must be marked ready (coming_soon=False)."""
    iris_entries = [m for name, m in ModelHub._MODELS.items() if name.startswith("iris-atari-")]
    assert len(iris_entries) >= 5, "Expected at least 5 IRIS Atari entries"
    for m in iris_entries:
        assert not m.coming_soon, f"{m.name} should not be coming_soon"
        assert m.hf_repo_id == "eloialonso/iris", f"{m.name}: wrong repo_id {m.hf_repo_id!r}"
        assert m.hf_filename is not None


def test_registry_dreamerv3_entries_are_coming_soon():
    """All dreamerv3-* entries must be coming_soon=True (JAX-only weights)."""
    dreamer_entries = [m for name, m in ModelHub._MODELS.items() if name.startswith("dreamerv3-")]
    assert len(dreamer_entries) >= 2, "Expected at least 2 DreamerV3 entries"
    for m in dreamer_entries:
        assert m.coming_soon, f"{m.name} should be coming_soon (no public PyTorch weights)"


def test_registry_tdmpc2_entries_are_coming_soon():
    """All tdmpc2-* entries must be coming_soon=True."""
    tdmpc2_entries = [m for name, m in ModelHub._MODELS.items() if name.startswith("tdmpc2-")]
    assert len(tdmpc2_entries) >= 1, "Expected at least 1 TD-MPC2 entry"
    for m in tdmpc2_entries:
        assert m.coming_soon, f"{m.name} should be coming_soon"


def test_registry_all_entries_have_required_fields():
    """Every registry entry must have name, backend, environment, description."""
    for name, m in ModelHub._MODELS.items():
        assert m.name == name, f"ModelInfo.name mismatch for key '{name}'"
        assert m.backend, f"Missing backend for '{name}'"
        assert m.environment, f"Missing environment for '{name}'"
        assert m.description, f"Missing description for '{name}'"


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub.list_available()
# ─────────────────────────────────────────────────────────────────────────────


def test_list_available_default_excludes_coming_soon():
    """list_available() without args should only return downloadable models."""
    models = ModelHub.list_available()
    for m in models:
        assert not m.coming_soon, f"list_available() returned coming_soon model: {m.name}"


def test_list_available_include_coming_soon_returns_all():
    """list_available(include_coming_soon=True) returns the full registry."""
    all_models = ModelHub.list_available(include_coming_soon=True)
    assert len(all_models) == len(ModelHub._MODELS)


def test_list_available_default_is_subset_of_all():
    """Ready models must be a strict subset of all models."""
    ready = set(m.name for m in ModelHub.list_available())
    all_ = set(m.name for m in ModelHub.list_available(include_coming_soon=True))
    assert ready.issubset(all_)
    assert len(ready) < len(all_), "Expected some coming_soon entries"


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub.info()
# ─────────────────────────────────────────────────────────────────────────────


def test_info_returns_correct_model():
    """info() returns the correct ModelInfo for known keys."""
    m = ModelHub.info("iris-atari-pong")
    assert m.name == "iris-atari-pong"
    assert m.backend == "iris"
    assert m.environment == "Atari/Pong"


def test_info_raises_key_error_for_unknown_model():
    """info() raises KeyError for an unrecognised model name."""
    with pytest.raises(KeyError, match="not found in registry"):
        ModelHub.info("this-model-does-not-exist")


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub.pull() — offline (mocked)
# ─────────────────────────────────────────────────────────────────────────────


def test_pull_raises_not_implemented_for_coming_soon():
    """pull() raises NotImplementedError for coming_soon models."""
    with pytest.raises(NotImplementedError, match="not yet available"):
        ModelHub.pull("dreamerv3-atari-pong")


def test_pull_raises_key_error_for_unknown_model():
    """pull() raises KeyError for unrecognised model names."""
    with pytest.raises(KeyError):
        ModelHub.pull("not-a-real-model")


def test_pull_coming_soon_error_mentions_reason():
    """NotImplementedError message includes notes about why it's not available."""
    with pytest.raises(NotImplementedError) as exc_info:
        ModelHub.pull("dreamerv3-atari-pong")
    msg = str(exc_info.value)
    assert "JAX" in msg or "coming soon" in msg.lower() or "list_available" in msg


def test_pull_downloads_via_hf_hub(tmp_path: Path):
    """pull() calls hf_hub_download with the correct repo_id and filename."""
    fake_path = str(tmp_path / "Pong.pt")
    # Create a dummy file so the path exists
    Path(fake_path).write_bytes(b"fake")

    with patch("world_model_lens.hub.model_hub.hf_hub_download", create=True) as mock_dl:
        # Patch the import inside pull()
        mock_dl.return_value = fake_path
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
            # We need to patch at the call site
            with patch(
                "world_model_lens.hub.model_hub.ModelHub.pull",
                return_value=fake_path,
            ) as mock_pull:
                result = ModelHub.pull("iris-atari-pong")
                mock_pull.assert_called_once_with("iris-atari-pong")
                assert result == fake_path


def test_pull_returns_string_path(tmp_path: Path):
    """pull() returns a string (local file path) on success."""
    fake_path = str(tmp_path / "Breakout.pt")
    with patch.object(ModelHub, "pull", return_value=fake_path):
        result = ModelHub.pull("iris-atari-breakout")
    assert isinstance(result, str)
    assert "Breakout" in result or fake_path == result


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub.push()
# ─────────────────────────────────────────────────────────────────────────────


def test_push_raises_not_implemented():
    """push() must raise NotImplementedError (not yet implemented)."""
    with pytest.raises(NotImplementedError):
        ModelHub.push(object(), "test", "Atari/Test", "iris")


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub._verify_sha256()
# ─────────────────────────────────────────────────────────────────────────────


def test_sha256_passes_for_correct_digest(tmp_path: Path):
    """_verify_sha256 does not raise when digest matches."""
    content = b"world model lens test content"
    expected = hashlib.sha256(content).hexdigest()
    f = tmp_path / "model.pt"
    f.write_bytes(content)
    ModelHub._verify_sha256(str(f), expected, "test-model")  # should not raise


def test_sha256_raises_for_wrong_digest(tmp_path: Path):
    """_verify_sha256 raises RuntimeError when digest does not match."""
    f = tmp_path / "model.pt"
    f.write_bytes(b"corrupted data")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        ModelHub._verify_sha256(str(f), "a" * 64, "test-model")


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub._map_iris_keys()
# ─────────────────────────────────────────────────────────────────────────────


def test_map_iris_keys_renames_ln_f():
    """transformer.ln_f.* in world_model keys should become transformer.ln.*"""
    # In the real flat format, wm_state already has the world_model. prefix stripped
    wm_state = {"transformer.ln_f.weight": "dummy_tensor"}
    mapped, skipped = ModelHub._map_iris_keys(wm_state, {})
    assert "transformer.ln.weight" in mapped


def test_map_iris_keys_renames_pos_emb():
    """pos_emb.weight in world_model keys should become transformer.pos_embedding.weight."""
    wm_state = {"pos_emb.weight": "dummy_tensor"}
    mapped, skipped = ModelHub._map_iris_keys(wm_state, {})
    assert "transformer.pos_embedding.weight" in mapped


def test_map_iris_keys_renames_token_embedding():
    """embedder.embedding_tables.0.weight should map to transformer.token_embedding.weight."""
    wm_state = {"embedder.embedding_tables.0.weight": "dummy_tensor"}
    mapped, skipped = ModelHub._map_iris_keys(wm_state, {})
    assert "transformer.token_embedding.weight" in mapped


def test_map_iris_keys_transformer_blocks_pass_through():
    """transformer.blocks.* keys should be passed through unchanged."""
    wm_state = {
        "transformer.blocks.0.ln1.weight": "t1",
        "transformer.blocks.0.attn.key.weight": "t2",
    }
    mapped, _ = ModelHub._map_iris_keys(wm_state, {})
    assert "transformer.blocks.0.ln1.weight" in mapped
    assert "transformer.blocks.0.attn.key.weight" in mapped


def test_map_iris_keys_head_observations_mapped():
    """head_observations.* should map to head.*"""
    wm_state = {"head_observations.head_module.0.weight": "t1"}
    mapped, skipped = ModelHub._map_iris_keys(wm_state, {})
    assert "head.head_module.0.weight" in mapped
    assert len(skipped) == 0


def test_map_iris_keys_head_rewards_mapped():
    """head_rewards.* should map to reward_head.*"""
    wm_state = {"head_rewards.head_module.0.weight": "t1"}
    mapped, skipped = ModelHub._map_iris_keys(wm_state, {})
    assert "reward_head.head_module.0.weight" in mapped


def test_map_iris_keys_head_ends_mapped():
    """head_ends.* should map to continue_head.*"""
    wm_state = {"head_ends.head_module.0.weight": "t1"}
    mapped, skipped = ModelHub._map_iris_keys(wm_state, {})
    assert "continue_head.head_module.0.weight" in mapped


def test_map_iris_keys_unknown_tokenizer_key_goes_to_skipped():
    """Unrecognised tokenizer keys should end up in the skipped list."""
    tokenizer_state = {"some_random_key": "t1"}
    _, skipped = ModelHub._map_iris_keys({}, tokenizer_state)
    assert any("some_random_key" in s for s in skipped)


# ─────────────────────────────────────────────────────────────────────────────
# ModelHub._load_iris() — offline (mocked torch.load)
# ─────────────────────────────────────────────────────────────────────────────


def test_load_iris_raises_on_wrong_checkpoint_type(tmp_path: Path):
    """_load_iris raises RuntimeError if checkpoint is not a dict."""
    

    bad_path = tmp_path / "bad.pt"
    torch.save([1, 2, 3], bad_path)  # save a list instead of dict
    with pytest.raises(RuntimeError, match="Unexpected checkpoint type"):
        ModelHub._load_iris(str(bad_path), device="cpu")


def test_load_iris_raises_when_world_model_key_missing(tmp_path: Path):
    """_load_iris raises RuntimeError when no 'world_model.*' keys are present."""


    # A dict that has no world_model.* keys at all
    bad_ckpt = {
        "totally.unrelated.key": torch.zeros(4),
        "another.key": torch.zeros(4),
    }
    path = tmp_path / "no_wm.pt"
    torch.save(bad_ckpt, path)
    with pytest.raises(RuntimeError, match="Unrecognised checkpoint format"):
        ModelHub._load_iris(str(path), device="cpu")


def test_load_iris_returns_iris_adapter(fake_checkpoint_file: Path):
    """_load_iris returns an IRISAdapter instance from a minimal checkpoint."""
    

    # A partial load is expected (fake checkpoint has tiny/wrong shapes)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        adapter = ModelHub._load_iris(str(fake_checkpoint_file), device="cpu")

    assert isinstance(adapter, IRISAdapter)


def test_load_iris_returns_eval_mode(tmp_path: Path):
    """Adapter returned by _load_iris must be in eval() mode."""

    # Build an isolated checkpoint in this test's own tmp_path
    ckpt = {
        "world_model.transformer.blocks.0.ln1.weight": torch.ones(512),
        "world_model.transformer.blocks.0.ln1.bias": torch.zeros(512),
        "world_model.transformer.ln_f.weight": torch.ones(512),
        "world_model.transformer.ln_f.bias": torch.zeros(512),
        "world_model.pos_emb.weight": torch.zeros(1024, 512),
        "world_model.embedder.embedding_tables.0.weight": torch.zeros(512, 512),
        "world_model.embedder.embedding_tables.1.weight": torch.zeros(3, 512),
    }
    path = tmp_path / "eval_mode_test.pt"
    torch.save(ckpt, path)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        adapter = ModelHub._load_iris(str(path), device="cpu")

    assert not adapter.training, "Adapter should be in eval mode"


def test_load_dreamerv3_branch_returns_adapter(fake_dreamerv3_checkpoint: Path):
    """ModelHub.load() should dispatch to DreamerV3Adapter for dreamerv3 backends."""
    model_info = ModelInfo(
        name="dreamerv3-local-test",
        backend="dreamerv3",
        environment="TestEnv",
        description="test checkpoint",
        coming_soon=False,
        hf_repo_id="dummy/repo",
        hf_filename="dummy.pt",
    )

    with patch.object(ModelHub, "info", return_value=model_info):
        with patch.object(ModelHub, "pull", return_value=str(fake_dreamerv3_checkpoint)):
            adapter = ModelHub.load("dreamerv3-local-test", device="cpu")

    assert isinstance(adapter, DreamerV3Adapter)
    assert not adapter.training


def test_load_dreamerv3_registered_entry_still_raises_coming_soon():
    """Registered dreamerv3-* keys should still fail until a real HF source is wired up."""
    with pytest.raises(NotImplementedError, match="not yet available"):
        ModelHub.load("dreamerv3-atari-pong", device="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# WeightsDownloader — offline (no network)
# ─────────────────────────────────────────────────────────────────────────────


def test_downloader_initialises_cache_dir(tmp_cache: Path):
    """WeightsDownloader creates the cache directory on init."""
    assert tmp_cache.exists()


def test_downloader_list_ready_matches_model_hub(downloader: WeightsDownloader):
    """list_ready() must match ModelHub.list_available()."""
    assert downloader.list_ready() == ModelHub.list_available()


def test_downloader_list_all_matches_full_registry(downloader: WeightsDownloader):
    """list_all() must return all models including coming_soon."""
    assert downloader.list_all() == ModelHub.list_available(include_coming_soon=True)


def test_downloader_is_cached_false_before_download(downloader: WeightsDownloader):
    """is_cached() returns False when no pointer file exists."""
    assert downloader.is_cached("iris-atari-pong") is False


def test_downloader_cache_info_structure(downloader: WeightsDownloader):
    """cache_info() returns a dict with the expected structure for every model."""
    info = downloader.cache_info()
    assert set(info.keys()) == set(ModelHub._MODELS.keys())
    for name, entry in info.items():
        assert "cached" in entry
        assert "path" in entry
        assert "size_mb" in entry
        assert entry["cached"] is False  # nothing downloaded yet


def test_downloader_is_cached_true_after_pointer_written(
    downloader: WeightsDownloader, tmp_path: Path
):
    """is_cached() returns True after a valid pointer file is written."""
    # Simulate a download by writing a pointer file manually
    fake_file = tmp_path / "Pong.pt"
    fake_file.write_bytes(b"fake")
    pointer = downloader._pointer_path("iris-atari-pong")
    pointer.write_text(str(fake_file), encoding="utf-8")

    assert downloader.is_cached("iris-atari-pong") is True


def test_downloader_clear_cache_removes_pointer(downloader: WeightsDownloader, tmp_path: Path):
    """clear_cache(name) removes the pointer file for that model."""
    fake_file = tmp_path / "Pong.pt"
    fake_file.write_bytes(b"fake")
    pointer = downloader._pointer_path("iris-atari-pong")
    pointer.write_text(str(fake_file), encoding="utf-8")
    assert downloader.is_cached("iris-atari-pong") is True

    downloader.clear_cache("iris-atari-pong")
    assert downloader.is_cached("iris-atari-pong") is False


def test_downloader_clear_cache_all(downloader: WeightsDownloader, tmp_path: Path):
    """clear_cache() with no argument clears all pointer files."""
    # Populate pointers for two models
    for model_name in ("iris-atari-pong", "iris-atari-breakout"):
        fake_file = tmp_path / f"{model_name}.pt"
        fake_file.write_bytes(b"fake")
        pointer = downloader._pointer_path(model_name)
        pointer.write_text(str(fake_file), encoding="utf-8")

    downloader.clear_cache()  # clear all

    assert downloader.is_cached("iris-atari-pong") is False
    assert downloader.is_cached("iris-atari-breakout") is False


def test_downloader_download_skips_when_cached(downloader: WeightsDownloader, tmp_path: Path):
    """download() returns the cached path without calling ModelHub.pull()."""
    fake_file = tmp_path / "Pong.pt"
    fake_file.write_bytes(b"fake")
    pointer = downloader._pointer_path("iris-atari-pong")
    pointer.write_text(str(fake_file), encoding="utf-8")

    with patch.object(ModelHub, "pull") as mock_pull:
        result = downloader.download("iris-atari-pong", verbose=False)
        mock_pull.assert_not_called()

    assert result == fake_file


def test_downloader_download_calls_pull_when_not_cached(
    downloader: WeightsDownloader, tmp_path: Path
):
    """download() calls ModelHub.pull() when item is not cached."""
    fake_file = tmp_path / "Pong.pt"
    fake_file.write_bytes(b"fake")  # the "downloaded" file

    with patch.object(ModelHub, "pull", return_value=str(fake_file)) as mock_pull:
        result = downloader.download("iris-atari-pong", verbose=False)
        mock_pull.assert_called_once_with("iris-atari-pong", cache_dir=None, force=False)

    assert result == fake_file
    assert downloader.is_cached("iris-atari-pong") is True


def test_downloader_download_force_re_downloads(downloader: WeightsDownloader, tmp_path: Path):
    """download(force=True) calls ModelHub.pull() even when already cached."""
    fake_file = tmp_path / "Pong.pt"
    fake_file.write_bytes(b"fake")
    pointer = downloader._pointer_path("iris-atari-pong")
    pointer.write_text(str(fake_file), encoding="utf-8")

    with patch.object(ModelHub, "pull", return_value=str(fake_file)) as mock_pull:
        downloader.download("iris-atari-pong", force=True, verbose=False)
        mock_pull.assert_called_once()


def test_downloader_download_all_returns_dict(downloader: WeightsDownloader, tmp_path: Path):
    """download_all() returns a dict mapping name → path for all ready models."""
    ready = downloader.list_ready()
    fake_path = str(tmp_path / "model.pt")
    Path(fake_path).write_bytes(b"fake")

    with patch.object(ModelHub, "pull", return_value=fake_path):
        results = downloader.download_all(verbose=False)

    assert isinstance(results, dict)
    assert len(results) == len(ready)
    for name in results:
        assert name in {m.name for m in ready}


def test_downloader_download_all_handles_individual_failures(
    downloader: WeightsDownloader,
):
    """download_all() continues even when one download fails."""

    def selective_fail(name, **kwargs):
        if name == "iris-atari-pong":
            raise RuntimeError("Simulated network error")
        return "/fake/path.pt"

    with patch.object(ModelHub, "pull", side_effect=selective_fail):
        with patch("pathlib.Path.write_bytes"):
            with patch("pathlib.Path.write_text"):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024 * 1024 * 127
                    results = downloader.download_all(verbose=False)

    # Should have results for all models except the failing one
    assert "iris-atari-pong" not in results


# ─────────────────────────────────────────────────────────────────────────────
# Hub package exports
# ─────────────────────────────────────────────────────────────────────────────


def test_hub_package_exports_model_hub():
    """world_model_lens.hub must export ModelHub."""

    assert hasattr(hub, "ModelHub")


def test_hub_package_exports_model_info():
    """world_model_lens.hub must export ModelInfo (not the old ModelCard name)."""
    

    assert hasattr(hub, "ModelInfo")
    assert not hasattr(
        hub, "ModelCard"
    ), "ModelCard was renamed to ModelInfo; old name must not be re-exported"


def test_hub_package_exports_weights_downloader():
    """world_model_lens.hub must export WeightsDownloader."""

    assert hasattr(hub, "WeightsDownloader")


def test_hub_package_exports_trajectory_hub():
    """world_model_lens.hub must export TrajectoryHub and TrajectoryDataset."""

    assert hasattr(hub, "TrajectoryHub")
    assert hasattr(hub, "TrajectoryDataset")


# ─────────────────────────────────────────────────────────────────────────────
# Network-dependent tests (skipped by default — run with: pytest -m network)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.network
def test_pull_iris_atari_pong_downloads_file(tmp_path: Path):
    """[NETWORK] pull() for iris-atari-pong downloads a real .pt file."""
    path = ModelHub.pull("iris-atari-pong", cache_dir=str(tmp_path))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1_000_000, "File too small — likely failed"


@pytest.mark.network
def test_pull_returns_path_ending_in_pt(tmp_path: Path):
    """[NETWORK] Downloaded checkpoint file has a .pt extension."""
    path = ModelHub.pull("iris-atari-pong", cache_dir=str(tmp_path))
    assert path.endswith(".pt")


@pytest.mark.network
def test_load_iris_atari_pong_returns_adapter(tmp_path: Path):
    """[NETWORK] load() returns an IRISAdapter for iris-atari-pong."""

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        adapter = ModelHub.load("iris-atari-pong", cache_dir=str(tmp_path))

    assert isinstance(adapter, IRISAdapter)
    assert not adapter.training
