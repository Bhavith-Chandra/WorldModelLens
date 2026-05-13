"""Model hub for downloading and managing pretrained world models.

Known-good checkpoints
----------------------
IRIS (Atari 100k)
    Official PyTorch ``.pt`` files published by the IRIS authors at
    ``eloialonso/iris`` on HuggingFace. 26 games are available.

DreamerV3 (Atari / DMC)
    The official DreamerV3 implementation is in JAX. No public PyTorch
    checkpoints are available. All entries are marked ``coming_soon=True``.

TD-MPC2 (DMC / Meta-World / ManiSkill2 / MyoSuite)
    Official PyTorch checkpoints exist at ``nicklashansen/tdmpc2``.
    Weight-key mapping from their model class to our TDMPC2Adapter
    is in progress; entries are ``coming_soon=True`` for now.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

import torch

from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.backends.iris import IRISAdapter
from world_model_lens.core.config import WorldModelConfig, WorldModelFamily


@dataclass
class ModelInfo:
    """Information about a pretrained model."""

    name: str
    backend: str
    environment: str
    description: str = ""
    coming_soon: bool = True
    # HuggingFace coordinates; None means no public checkpoint exists yet.
    hf_repo_id: Optional[str] = None
    hf_filename: Optional[str] = None
    source_url: Optional[str] = None
    sha256: Optional[str] = None
    notes: str = ""

    @property
    def is_downloadable(self) -> bool:
        """True when a real checkpoint is available and can be pulled."""
        has_hf = self.hf_repo_id is not None and self.hf_filename is not None
        has_url = self.source_url is not None
        return not self.coming_soon and (has_hf or has_url)


class ModelHub:
    """Hub for downloading and managing pretrained world models.

    Example:
        hub = ModelHub()

        # Discover what is ready to download
        hub.list_available()

        # Download a checkpoint to the HuggingFace cache
        path = hub.pull("iris-atari-breakout")

        # Download + load into an IRISAdapter
        adapter = hub.load("iris-atari-breakout")

    Currently available (real PyTorch weights on HuggingFace):

    - ``iris-atari-*`` — IRIS world model trained on 26 Atari 100k games.
      Source: `eloialonso/iris <https://huggingface.co/eloialonso/iris>`_

    Coming soon:

    - ``dreamerv3-*`` — JAX-only implementation; no public PyTorch weights.
    - ``tdmpc2-*``    — Real HF weights at ``nicklashansen/tdmpc2``;
      adapter key-mapping in progress.
    """

    _MODELS: Dict[str, ModelInfo] = {
        # ──────────────────────────────────────────────────────────────────────
        # IRIS — Atari 100k  (eloialonso/iris on HuggingFace, native PyTorch)
        # These are the official checkpoints from the paper authors.
        # 127 MB each — tokenizer + world model + actor-critic bundled together.
        # ──────────────────────────────────────────────────────────────────────
        "iris-atari-breakout": ModelInfo(
            name="iris-atari-breakout",
            backend="iris",
            environment="Atari/Breakout",
            description="IRIS transformer world model trained on Atari Breakout (100k steps).",
            coming_soon=False,
            hf_repo_id="eloialonso/iris",
            hf_filename="pretrained_models/Breakout.pt",
            notes=(
                "Official checkpoint from the IRIS paper (ICLR 2023, top 5%). "
                "Bundles tokenizer + world model + actor-critic. "
                "pull() downloads the raw file; load() maps weights onto IRISAdapter. "
                "See https://github.com/eloialonso/iris for the original codebase."
            ),
        ),
        "iris-atari-pong": ModelInfo(
            name="iris-atari-pong",
            backend="iris",
            environment="Atari/Pong",
            description="IRIS transformer world model trained on Atari Pong (100k steps).",
            coming_soon=False,
            hf_repo_id="eloialonso/iris",
            hf_filename="pretrained_models/Pong.pt",
            notes=(
                "Official checkpoint from the IRIS paper (ICLR 2023). "
                "Source: eloialonso/iris on HuggingFace."
            ),
        ),
        "iris-atari-seaquest": ModelInfo(
            name="iris-atari-seaquest",
            backend="iris",
            environment="Atari/Seaquest",
            description="IRIS transformer world model trained on Atari Seaquest (100k steps).",
            coming_soon=False,
            hf_repo_id="eloialonso/iris",
            hf_filename="pretrained_models/Seaquest.pt",
            notes="Official checkpoint. Source: eloialonso/iris on HuggingFace.",
        ),
        "iris-atari-freeway": ModelInfo(
            name="iris-atari-freeway",
            backend="iris",
            environment="Atari/Freeway",
            description="IRIS transformer world model trained on Atari Freeway (100k steps).",
            coming_soon=False,
            hf_repo_id="eloialonso/iris",
            hf_filename="pretrained_models/Freeway.pt",
            notes="Official checkpoint. Source: eloialonso/iris on HuggingFace.",
        ),
        "iris-atari-alien": ModelInfo(
            name="iris-atari-alien",
            backend="iris",
            environment="Atari/Alien",
            description="IRIS transformer world model trained on Atari Alien (100k steps).",
            coming_soon=False,
            hf_repo_id="eloialonso/iris",
            hf_filename="pretrained_models/Alien.pt",
            notes="Official checkpoint. Source: eloialonso/iris on HuggingFace.",
        ),
        "iris-atari-qbert": ModelInfo(
            name="iris-atari-qbert",
            backend="iris",
            environment="Atari/Qbert",
            description="IRIS transformer world model trained on Atari Q*bert (100k steps).",
            coming_soon=False,
            hf_repo_id="eloialonso/iris",
            hf_filename="pretrained_models/Qbert.pt",
            notes="Official checkpoint. Source: eloialonso/iris on HuggingFace.",
        ),
        "iris-atari-ms-pacman": ModelInfo(
            name="iris-atari-ms-pacman",
            backend="iris",
            environment="Atari/MsPacman",
            description="IRIS transformer world model trained on Atari Ms. Pac-Man (100k steps).",
            coming_soon=False,
            hf_repo_id="eloialonso/iris",
            hf_filename="pretrained_models/MsPacman.pt",
            notes="Official checkpoint. Source: eloialonso/iris on HuggingFace.",
        ),
        # ──────────────────────────────────────────────────────────────────────
        # IJEPA — META checkpoints.
        # ──────────────────────────────────────────────────────────────────────
        "ijepa-vit-h-in1k": ModelInfo(
            name="ijepa-vit-h-in1k",
            backend="ijepa",
            environment="ImageNet-1K",
            description="I-JEPA ViT-H/14 pretrained on ImageNet-1K for 300 epochs.",
            coming_soon=False,
            source_url="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
            hf_filename="vith14_in1k_ep300.pth.tar",
            notes=(
                "Official Meta I-JEPA checkpoint. The downloaded file is a PyTorch "
                "pickle even though it uses a .pth.tar suffix."
            ),
        ),
        "ijepa-vit-h-in22k": ModelInfo(
            name="ijepa-vit-h-in22k",
            backend="ijepa",
            environment="ImageNet-22K",
            description="I-JEPA ViT-H/14 pretrained on ImageNet-22K for 900 epochs.",
            coming_soon=False,
            source_url="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar",
            hf_filename="vith14_in22k_ep900.pth.tar",
            notes=(
                "Official Meta I-JEPA checkpoint. The downloaded file is a PyTorch "
                "pickle even though it uses a .pth.tar suffix."
            ),
        ),
        "dreamerv3-atari-breakout": ModelInfo(
            name="dreamerv3-atari-breakout",
            backend="dreamerv3",
            environment="Atari/Breakout",
            coming_soon=True,
            description="DreamerV3 trained on Atari Breakout.",
            notes=(
                "The official DreamerV3 implementation (danijar/dreamerv3) is written "
                "in JAX. No converted PyTorch checkpoints are publicly available. "
                "To use DreamerV3, train from scratch using a PyTorch port such as "
                "NM512/dreamerv3-torch or sheeprl."
            ),
        ),
        "dreamerv3-atari-pong": ModelInfo(
            name="dreamerv3-atari-pong",
            backend="dreamerv3",
            environment="Atari/Pong",
            coming_soon=True,
            description="DreamerV3 trained on Atari Pong.",
            notes="JAX-only implementation. No public PyTorch checkpoint available.",
        ),
        "dreamerv3-dmcontrol-cheetah-run": ModelInfo(
            name="dreamerv3-dmcontrol-cheetah-run",
            backend="dreamerv3",
            environment="DeepMind Control/Cheetah-run",
            coming_soon=True,
            description="DreamerV3 trained on DM Control Cheetah Run.",
            notes="JAX-only implementation. No public PyTorch checkpoint available.",
        ),
        "dreamerv3-dmcontrol-walker-walk": ModelInfo(
            name="dreamerv3-dmcontrol-walker-walk",
            backend="dreamerv3",
            environment="DeepMind Control/Walker-walk",
            coming_soon=True,
            description="DreamerV3 trained on DM Control Walker Walk.",
            notes="JAX-only implementation. No public PyTorch checkpoint available.",
        ),
        # ──────────────────────────────────────────────────────────────────────
        # TD-MPC2 — coming soon
        # Real PyTorch weights exist at nicklashansen/tdmpc2 on HuggingFace.
        # Adapter key-mapping from their TDMPC2 class to ours is in progress.
        # ──────────────────────────────────────────────────────────────────────
        "tdmpc2-dmcontrol-cheetah-run": ModelInfo(
            name="tdmpc2-dmcontrol-cheetah-run",
            backend="tdmpc2",
            environment="DeepMind Control/Cheetah-run",
            coming_soon=True,
            description="TD-MPC2 trained on DM Control Cheetah Run.",
            notes=(
                "Official PyTorch checkpoints exist at nicklashansen/tdmpc2 on "
                "HuggingFace (324 checkpoints, MIT license). Key mapping from "
                "nicklashansen's TDMPC2 model class to our TDMPC2Adapter "
                "is in progress. Coming soon."
            ),
        ),
        "tdmpc2-dmcontrol-humanoid": ModelInfo(
            name="tdmpc2-dmcontrol-humanoid",
            backend="tdmpc2",
            environment="DeepMind Control/Humanoid",
            coming_soon=True,
            description="TD-MPC2 trained on DM Control Humanoid.",
            notes=("Real checkpoints at nicklashansen/tdmpc2. Adapter key-mapping in progress."),
        ),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def list_available(cls, include_coming_soon: bool = False) -> List[ModelInfo]:
        """List registered models.

        Args:
            include_coming_soon: When ``True``, include models that are not
                yet downloadable. Defaults to ``False`` (only show models
                with real, downloadable weights).

        Returns:
            List of :class:`ModelInfo` entries.
        """
        if include_coming_soon:
            return list(cls._MODELS.values())
        return [m for m in cls._MODELS.values() if m.is_downloadable]

    @classmethod
    def info(cls, name: str) -> ModelInfo:
        """Return metadata for a named model.

        Args:
            name: Registry key, e.g. ``"iris-atari-breakout"``.

        Returns:
            :class:`ModelInfo` dataclass.

        Raises:
            KeyError: If ``name`` is not in the registry.
        """
        if name not in cls._MODELS:
            available = list(cls._MODELS.keys())
            raise KeyError(f"Model '{name}' not found in registry.\nAvailable keys: {available}")
        return cls._MODELS[name]

    # ──────────────────────────────────────────────────────────────────────────
    # Download
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def pull(
        cls,
        name: str,
        cache_dir: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Download a checkpoint and return the local file path.

        Uses ``huggingface_hub`` for HF-backed models and direct HTTP
        downloads for models registered with ``source_url``.

        Args:
            name: Registry key, e.g. ``"iris-atari-breakout"``.
            cache_dir: Override the cache directory.
            force: Re-download even if already cached.

        Returns:
            Absolute path to the local ``.pt`` file.

        Raises:
            KeyError: Model not in registry.
            NotImplementedError: Model is marked ``coming_soon``.
            RuntimeError: Download failed or ``huggingface_hub`` not installed.
        """
        model_info = cls.info(name)

        if model_info.coming_soon:
            raise NotImplementedError(
                f"Model '{name}' is not yet available for download.\n\n"
                f"Reason: {model_info.notes or 'Coming soon.'}\n\n"
                f"To see currently downloadable models run:\n"
                f"  ModelHub.list_available()\n"
                f"  # or from the CLI: wml download --list"
            )

        if model_info.source_url is not None:
            return cls._download_direct_url(
                model_info.source_url,
                model_info.hf_filename or Path(model_info.source_url).name,
                cache_dir=cache_dir,
                force=force,
            )

        if model_info.hf_repo_id is None or model_info.hf_filename is None:
            raise RuntimeError(
                f"Model '{name}' has incomplete HuggingFace coordinates in the registry. "
                "This is a bug — please open an issue."
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for model downloads.\n"
                "Install it with:  pip install huggingface_hub"
            )

        try:
            local_path = hf_hub_download(
                repo_id=model_info.hf_repo_id,
                filename=model_info.hf_filename,
                cache_dir=cache_dir,
                force_download=force,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{name}' from "
                f"{model_info.hf_repo_id}/{model_info.hf_filename}.\n"
                f"Cause: {exc}"
            ) from exc

        if model_info.sha256:
            cls._verify_sha256(local_path, model_info.sha256, name)

        return local_path

    # ──────────────────────────────────────────────────────────────────────────
    # Load (download + instantiate adapter)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        name: str,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        force_download: bool = False,
    ) -> Any:
        """Download (if needed) and load a model into a WorldModelAdapter.

        For IRIS checkpoints, extracts the world model sub-state-dict and
        loads it into an :class:`~world_model_lens.backends.iris.IRISAdapter`.

        For backends where adapter loading is not yet implemented, raises
        :exc:`NotImplementedError` with a helpful message. You can still
        obtain the raw file via :meth:`pull`.

        Args:
            name: Registry key, e.g. ``"iris-atari-breakout"``.
            cache_dir: Optional download cache directory.
            device: PyTorch device string (``"cpu"``, ``"cuda"``, etc.).
            force_download: Re-download even if cached.

        Returns:
            Loaded adapter in ``eval()`` mode.

        Raises:
            KeyError: Model not in registry.
            NotImplementedError: Model is ``coming_soon`` or adapter loading
                not yet implemented for that backend.
            RuntimeError: Download or loading failed.
        """
        model_info = cls.info(name)
        local_path = cls.pull(name, cache_dir=cache_dir, force=force_download)

        if model_info.backend == "iris":
            return cls._load_iris(local_path, device=device)
        if model_info.backend == "ijepa":
            return cls._load_ijepa(local_path, device=device)

        raise NotImplementedError(
            f"Adapter loading for backend '{model_info.backend}' is not yet wired up.\n"
            f"You can still obtain the raw checkpoint file:\n"
            f"  path = ModelHub.pull('{name}')\n"
            f"  import torch; ckpt = torch.load(path, map_location='{device}')"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Push (stub)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def push(cls, model: Any, name: str, environment: str, backend: str) -> None:
        """Upload a model to the hub (not yet implemented).

        Args:
            model: World model to upload.
            name: Name for the model.
            environment: Environment the model was trained on.
            backend: Backend architecture name.

        Raises:
            NotImplementedError: Always. Contribute via a pull request.
        """
        raise NotImplementedError(
            "Model upload not yet implemented. "
            "Please open an issue or PR to contribute checkpoints."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def _download_direct_url(
        cls,
        url: str,
        filename: str,
        cache_dir: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Download a checkpoint from a direct URL into the local cache."""
        if cache_dir is None:
            cache_base = Path.home() / ".cache" / "world_model_lens" / "models"
        else:
            cache_base = Path(cache_dir)
        cache_base.mkdir(parents=True, exist_ok=True)

        local_path = cache_base / filename
        if local_path.exists() and not force:
            return str(local_path)

        try:
            with urlopen(url) as response, open(local_path, "wb") as fh:
                shutil.copyfileobj(response, fh)
        except Exception as exc:
            raise RuntimeError(f"Failed to download '{url}'. Cause: {exc}") from exc

        return str(local_path)

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a leading DDP 'module.' prefix from checkpoint keys."""
        if not state_dict:
            return state_dict
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {k[len("module.") :]: v for k, v in state_dict.items() if k.startswith("module.")}

    @staticmethod
    def _infer_tensor_dim(state_dict: Dict[str, Any], key: str, fallback: int) -> int:
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor):
            return int(tensor.shape[0])
        return fallback

    @classmethod
    def _load_ijepa(cls, checkpoint_path: str, device: str = "cpu") -> Any:
        """Load an official Meta I-JEPA checkpoint into our IJEPAAdapter."""
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"torch.load failed on '{checkpoint_path}'.\nCause: {exc}") from exc

        if not isinstance(ckpt, dict):
            raise RuntimeError(
                f"Unexpected checkpoint type {type(ckpt).__name__}. "
                "Expected a dict with encoder/target_encoder/predictor keys."
            )

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        encoder_state = ckpt.get("encoder") or ckpt.get("context_encoder")
        target_state = ckpt.get("target_encoder") or encoder_state
        predictor_state = ckpt.get("predictor")

        if encoder_state is None or predictor_state is None:
            raise RuntimeError(
                "Unrecognised I-JEPA checkpoint layout. Expected top-level "
                "'encoder' and 'predictor' state dicts."
            )

        encoder_state = cls._strip_module_prefix(encoder_state)
        target_state = cls._strip_module_prefix(target_state or {})
        predictor_state = cls._strip_module_prefix(predictor_state)

        patch_weight = encoder_state.get("patch_embed.proj.weight")
        if not isinstance(patch_weight, torch.Tensor):
            raise RuntimeError(
                "I-JEPA encoder checkpoint is missing 'patch_embed.proj.weight'. "
                "Cannot infer model dimensions."
            )

        d_embed = int(patch_weight.shape[0])
        patch_size = int(patch_weight.shape[2])
        n_layers = sum(
            1
            for k in encoder_state.keys()
            if k.startswith("blocks.") and k.endswith(".norm1.weight")
        )
        pos_embed = encoder_state.get("pos_embed")
        img_size = 224
        if isinstance(pos_embed, torch.Tensor) and pos_embed.dim() == 3:
            num_patches = int(pos_embed.shape[1])
            grid = int(math.sqrt(num_patches))
            if grid * grid == num_patches:
                img_size = grid * patch_size

        predictor_embed = predictor_state.get("predictor_embed.weight")
        predictor_embed_dim = (
            int(predictor_embed.shape[0]) if isinstance(predictor_embed, torch.Tensor) else 384
        )
        predictor_depth = (
            sum(
                1
                for k in predictor_state.keys()
                if k.startswith("blocks.") and k.endswith(".norm1.weight")
            )
            or 4
        )

        cfg = WorldModelConfig(
            d_h=d_embed,
            d_obs=img_size * img_size * 3,
            d_action=0,
            encoder_type="vit",
            backend="ijepa",
            world_model_family=WorldModelFamily.JEPA,
            n_layers=n_layers,
            n_heads=16,
            d_embed=d_embed,
            patch_size=patch_size,
            img_size=img_size,
            predictor_embed_dim=predictor_embed_dim,
            predictor_depth=predictor_depth,
            predictor_heads=16,
        )

        adapter = IJEPAAdapter(cfg)
        encoder_missing, encoder_unexpected = adapter.context_encoder.load_state_dict(
            encoder_state, strict=False
        )
        target_missing, target_unexpected = adapter.target_encoder.load_state_dict(
            target_state, strict=False
        )
        predictor_missing, predictor_unexpected = adapter.predictor.load_state_dict(
            predictor_state, strict=False
        )

        if (
            encoder_missing
            or encoder_unexpected
            or target_missing
            or target_unexpected
            or predictor_missing
            or predictor_unexpected
        ):
            warnings.warn(
                "I-JEPA checkpoint loaded with partial key mismatches; "
                "the adapter is usable but some weights were not matched exactly.",
                UserWarning,
                stacklevel=2,
            )

        adapter = adapter.to(torch.device(device))
        adapter.eval()
        return adapter

    @classmethod
    def _load_iris(cls, checkpoint_path: str, device: str = "cpu") -> Any:
        """Load an official IRIS checkpoint into our IRISAdapter.

        The official IRIS checkpoint (from ``eloialonso/iris``) is saved as
        a Python dict with top-level keys::

            {
                "epoch": int,
                "tokenizer": {state_dict of VQVAE tokenizer},
                "world_model": {state_dict of GPT-style transformer},
                "actor_critic": {state_dict},
            }

        We extract ``tokenizer`` and ``world_model``, remap keys to match
        our :class:`~world_model_lens.backends.iris.IRISAdapter` layout,
        and perform a non-strict ``load_state_dict`` (logging any unmapped
        keys as a warning rather than raising an error).

        Args:
            checkpoint_path: Local path to the ``.pt`` file.
            device: Target PyTorch device.

        Returns:
            :class:`~world_model_lens.backends.iris.IRISAdapter` in eval mode.
        """

        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
        except Exception as exc:
            raise RuntimeError(f"torch.load failed on '{checkpoint_path}'.\nCause: {exc}") from exc

        if not isinstance(ckpt, dict):
            raise RuntimeError(
                f"Unexpected checkpoint type {type(ckpt).__name__}. "
                "Expected a flat state dict with 'world_model.*' prefixed keys."
            )

        # ── Detect checkpoint format ────────────────────────────────────────
        # Format A (actual eloialonso/iris): flat OrderedDict where every key
        # carries a component prefix, e.g.
        #   "world_model.transformer.blocks.0.ln1.weight"
        #   "tokenizer.encoder.conv_in.weight"
        #   "actor_critic.conv1.weight"
        #
        # Format B (assumed in earlier versions): nested dict
        #   ckpt["world_model"] = {state_dict}, ckpt["tokenizer"] = {state_dict}
        #
        # We auto-detect by checking whether any key starts with "world_model."

        flat_format = any(k.startswith("world_model.") for k in ckpt)

        if flat_format:
            # Split the flat dict by prefix — strip the leading component name.
            wm_state = {
                k[len("world_model.") :]: v for k, v in ckpt.items() if k.startswith("world_model.")
            }
            tokenizer_state = {
                k[len("tokenizer.") :]: v for k, v in ckpt.items() if k.startswith("tokenizer.")
            }
        elif "world_model" in ckpt and isinstance(ckpt.get("world_model"), dict):
            # Format B — nested dict (kept for forward-compatibility)
            wm_state = ckpt["world_model"]
            tokenizer_state = ckpt.get("tokenizer", {})
        else:
            raise RuntimeError(
                "Unrecognised checkpoint format. Expected either:\n"
                "  (a) flat state dict with 'world_model.*' prefixed keys, or\n"
                "  (b) nested dict with a 'world_model' sub-dict.\n"
                f"Found top-level keys (first 5): {list(ckpt.keys())[:5]}"
            )

        if not wm_state:
            raise RuntimeError(
                "No 'world_model.*' keys found in checkpoint. "
                f"Top-level keys (first 10): {list(ckpt.keys())[:10]}"
            )

        # ── Infer hyperparameters from checkpoint tensor shapes ──────────────
        # This avoids hardcoding dims that differ across games / training runs.
        # The IRIS Pong checkpoint uses d_model=256; Breakout uses 512; etc.
        # Try a few common keys for d_model inference
        d_model = cls._infer_dim(wm_state, "transformer.blocks.0.ln1.weight", dim=0, fallback=None)
        if d_model is None:
            d_model = cls._infer_dim(wm_state, "transformer.ln.weight", dim=0, fallback=256) or 256

        n_layers = (
            sum(
                1
                for k in wm_state
                if k.startswith("transformer.blocks.") and k.endswith(".ln1.weight")
            )
            or 10
        )

        max_seq_len = cls._infer_dim(wm_state, "pos_emb.weight", dim=0, fallback=1024) or 1024

        # token_embedding tells us the vocab size of the first embedder table
        emb0_vocab = cls._infer_dim(
            wm_state, "embedder.embedding_tables.0.weight", dim=0, fallback=None
        )
        if emb0_vocab is None:
            emb0_vocab = (
                cls._infer_dim(wm_state, "transformer.token_embedding.weight", dim=0, fallback=512)
                or 512
            )

        # n_head: inferred from attention key weight shape [n_head*head_dim, d_model]
        # or just fallback to paper defaults based on d_model.
        n_head = 8 if d_model >= 512 else 4

        print(
            f"[ModelHub] Inferred for '{checkpoint_path.split('/')[-1]}': "
            f"d_model={d_model}, n_layers={n_layers}, vocab={emb0_vocab}, seq={max_seq_len}"
        )

        cfg = WorldModelConfig(
            d_h=d_model,
            d_obs=64 * 64 * 3,  # Atari 64×64 pixel observations (flattened)
            d_action=18,  # max Atari action space
            n_cat=1,
            n_cls=emb0_vocab,
            backend="iris",
            d_embed=d_model,
            n_layers=n_layers,
            n_heads=n_head,
            vocab_size=emb0_vocab,
        )
        adapter = IRISAdapter(
            cfg,
            d_model=d_model,
            n_layers=n_layers,
            n_head=n_head,
            vocab_size=emb0_vocab,
        )

        # Patch the positional embedding size to match the checkpoint exactly.
        # IRISTransformer hardcodes Embedding(1024, d_model) but real checkpoints
        # may use a different sequence length (e.g. 340 for Pong).
        if max_seq_len != 1024 and hasattr(adapter.transformer, "pos_embedding"):
            adapter.transformer.pos_embedding = torch.nn.Embedding(int(max_seq_len), int(d_model))

        mapped, skipped = cls._map_iris_keys(wm_state, tokenizer_state)

        # ── Shape-safe load ─────────────────────────────────────────────────
        # torch's load_state_dict(strict=False) ignores missing/extra keys but
        # still raises on shape mismatches. We manually filter those out so
        # load() never crashes — mismatched tensors are logged as warnings.
        current_state = adapter.state_dict()
        safe_mapped: Dict = {}
        shape_skipped: List[str] = []
        for k, v in mapped.items():
            if k not in current_state:
                # Will become an unexpected_keys entry — let strict=False handle it.
                safe_mapped[k] = v
            elif v.shape != current_state[k].shape:
                shape_skipped.append(
                    f"{k}  [ckpt {tuple(v.shape)} vs adapter {tuple(current_state[k].shape)}]"
                )
            else:
                safe_mapped[k] = v

        missing_keys, unexpected_keys = adapter.load_state_dict(safe_mapped, strict=False)

        all_issues: List[str] = (
            skipped + shape_skipped + [f"unexpected:{k}" for k in unexpected_keys]
        )
        if all_issues:
            warnings.warn(
                f"IRIS checkpoint: {len(all_issues)} key(s) were skipped during load "
                f"(shape mismatches, unmapped keys, or unexpected keys). "
                f"First 5: {all_issues[:5]!r}. "
                "The adapter is partially loaded; outputs may be approximate. "
                "Full key mapping is tracked in model_hub.py::_map_iris_keys.",
                UserWarning,
                stacklevel=3,
            )

        adapter = adapter.to(torch.device(device))
        adapter.eval()
        return adapter

    @classmethod
    def _map_iris_keys(
        cls,
        wm_state: Dict,
        tokenizer_state: Dict,
    ) -> tuple:
        """Map real IRIS checkpoint keys to our IRISAdapter parameter names.

        The actual ``eloialonso/iris`` checkpoint (flat OrderedDict format)
        uses the following world_model sub-key structure after stripping
        the ``world_model.`` prefix::

            transformer.blocks.N.ln1.weight/bias
            transformer.blocks.N.ln2.weight/bias
            transformer.blocks.N.attn.key/query/value.weight/bias
            transformer.blocks.N.attn.proj.weight/bias
            transformer.blocks.N.mlp.0.weight/bias
            transformer.blocks.N.mlp.2.weight/bias
            transformer.ln_f.weight/bias          # final layer-norm
            pos_emb.weight                         # positional embedding
            embedder.embedding_tables.0.weight     # token embedding (obs tokens)
            embedder.embedding_tables.1.weight     # token embedding (reward/done)
            head_observations.head_module.*        # reconstruction head
            head_rewards.head_module.*             # reward prediction head
            head_ends.head_module.*                # episode-end head

        We map these onto the IRISAdapter layout as closely as possible.
        Keys that have no matching target are collected in ``skipped``.

        Args:
            wm_state: World-model sub-dict (keys already stripped of
                ``world_model.`` prefix).
            tokenizer_state: Tokenizer sub-dict (keys stripped of
                ``tokenizer.`` prefix).

        Returns:
            Tuple of ``(mapped_dict, skipped_keys)``.
        """
        mapped: Dict = {}
        skipped: List[str] = []

        # ── world_model keys ────────────────────────────────────────────────
        for k, v in wm_state.items():
            # transformer.ln_f.* → transformer.ln.*
            if k.startswith("transformer.ln_f."):
                new_k = k.replace("transformer.ln_f.", "transformer.ln.")
                mapped[new_k] = v

            # pos_emb.weight → transformer.pos_embedding.weight
            elif k == "pos_emb.weight":
                mapped["transformer.pos_embedding.weight"] = v

            # embedder.embedding_tables.0.weight → transformer.token_embedding.weight
            # (table 0 = observation tokens; table 1 = special tokens)
            elif k == "embedder.embedding_tables.0.weight":
                mapped["transformer.token_embedding.weight"] = v
            elif k == "embedder.embedding_tables.1.weight":
                mapped["transformer.reward_embedding.weight"] = v

            # head_observations → head (reconstruction head)
            elif k.startswith("head_observations."):
                suffix = k[len("head_observations.") :]
                mapped[f"head.{suffix}"] = v

            # head_rewards → reward_head
            elif k.startswith("head_rewards."):
                suffix = k[len("head_rewards.") :]
                mapped[f"reward_head.{suffix}"] = v

            # head_ends → continue_head
            elif k.startswith("head_ends."):
                suffix = k[len("head_ends.") :]
                mapped[f"continue_head.{suffix}"] = v

            # transformer.blocks.* — pass through as-is (same naming)
            elif k.startswith("transformer."):
                mapped[k] = v

            else:
                skipped.append(f"world_model:{k}")

        # ── tokenizer (VQ-VAE encoder / decoder) keys ───────────────────────
        for k, v in tokenizer_state.items():
            if k.startswith("encoder."):
                # Our IRISAdapter has self.encoder = VQVAEEncoder; its children
                # follow the same sub-structure as the original IRIS tokenizer.
                mapped[f"encoder.{k}"] = v
            elif k == "embedding.weight":
                # VQ codebook
                mapped["encoder.embedding.weight"] = v
            elif k.startswith("pre_quant_conv.") or k.startswith("post_quant_conv."):
                mapped[f"encoder.{k}"] = v
            elif k.startswith("decoder."):
                # We expose the decoder separately if needed; skip for now.
                skipped.append(f"tokenizer:{k}")
            elif k.startswith("lpips."):
                # LPIPS perceptual loss network — not needed for inference.
                skipped.append(f"tokenizer:{k}")
            else:
                skipped.append(f"tokenizer:{k}")

        return mapped, skipped

    @staticmethod
    def _infer_dim(
        state: Dict,
        key: str,
        dim: int = 0,
        fallback: Optional[int] = 256,
    ) -> Optional[int]:
        """Read a tensor dimension from a state dict without crashing.

        Args:
            state: State dict to inspect.
            key: Parameter name to look up.
            dim: Which axis of the tensor shape to read.
            fallback: Value to return if the key is missing.

        Returns:
            Integer dimension value.
        """
        if key not in state:
            return fallback
        shape = state[key].shape
        if dim >= len(shape):
            return fallback
        return int(shape[dim])

    @staticmethod
    def _verify_sha256(path: str, expected: str, name: str) -> None:
        """Verify the SHA-256 digest of a downloaded file.

        Args:
            path: Path to the downloaded file.
            expected: Expected hex digest string.
            name: Model name (for error messages).

        Raises:
            RuntimeError: If the digest does not match.
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for '{name}'.\n"
                f"  Expected: {expected}\n"
                f"  Got:      {actual}\n"
                "The file may be corrupted. Delete the cache entry and retry."
            )
