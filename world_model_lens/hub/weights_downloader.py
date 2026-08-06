"""Weights downloader — download and cache known-good world model checkpoints.

Provides a standalone cache manager that wraps :class:`ModelHub` with
progress reporting, structured cache introspection and batch download support.

Example::

    dl = WeightsDownloader()
    dl.list_ready()                           # show downloadable models
    path = dl.download("iris-atari-breakout") # download with progress output
    dl.cache_info()                           # inspect what is cached locally
    dl.clear_cache("iris-atari-breakout")     # remove a single cache pointer

CLI equivalents (after ``pip install -e .[dev]``)::

    wml download --list
    wml download iris-atari-breakout
    wml download --all

Post-clone script (works without installing the package)::

    python download_weights.py --list
    python download_weights.py iris-atari-breakout
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from world_model_lens.hub.model_hub import ModelHub, ModelInfo


class WeightsDownloader:
    """Download and cache known-good pretrained world model checkpoints.

    This is a thin utility layer on top of :class:`ModelHub` that adds:

    - Human-readable progress output
    - Structured local cache via pointer files
    - Batch download (all ready models at once)
    - Cache introspection (size, paths, hit/miss status)

    The underlying checkpoint files live in the HuggingFace Hub cache
    (``~/.cache/huggingface/`` by default). This class writes small
    *pointer files* alongside them so :meth:`is_cached` can check
    availability without re-downloading.

    Args:
        cache_dir: Directory for pointer files and download metadata.
            Defaults to ``~/.cache/world_model_lens/weights/``.
    """

    DEFAULT_CACHE: Path = Path("~/.cache/world_model_lens/weights")

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = (
            Path(cache_dir).expanduser() if cache_dir else self.DEFAULT_CACHE.expanduser()
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────────────────────────────────

    def list_ready(self) -> List[ModelInfo]:
        """Return only models with downloadable (non-coming_soon) weights."""
        return ModelHub.list_available(include_coming_soon=False)

    def list_all(self) -> List[ModelInfo]:
        """Return all registered models, including coming-soon entries."""
        return ModelHub.list_available(include_coming_soon=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Download
    # ──────────────────────────────────────────────────────────────────────────

    def download(
        self,
        name: str,
        force: bool = False,
        verbose: bool = True,
    ) -> Path:
        """Download a checkpoint and return the local path.

        Skips the network request if the file is already cached
        (unless ``force=True``).

        Args:
            name: Registry key, e.g. ``"iris-atari-breakout"``.
            force: Re-download even if already cached.
            verbose: Print progress lines to stdout.

        Returns:
            :class:`~pathlib.Path` pointing to the local ``.pt`` file.

        Raises:
            KeyError: Model not found in registry.
            NotImplementedError: Model is marked ``coming_soon``.
            RuntimeError: Network or file-system error.
        """
        model_info = ModelHub.info(name)

        if self.is_cached(name) and not force:
            cached = self._resolve_pointer(name)
            if verbose:
                print(f"  ✓ {name}  [already cached]  {cached}")
            return cached

        if verbose:
            src = (
                f"{model_info.hf_repo_id}/{model_info.hf_filename}"
                if model_info.hf_repo_id
                else "?"
            )
            print(f"  ↓ {name}  ({src}) ...")

        t0 = time.time()
        raw_path = ModelHub.pull(name, cache_dir=None, force=force)

        # Record the resolved path in a pointer file so is_cached() works fast.
        pointer = self._pointer_path(name)
        pointer.write_text(raw_path, encoding="utf-8")

        elapsed = time.time() - t0
        if verbose:
            size_mb = Path(raw_path).stat().st_size / (1024**2)
            print(f"  ✓ {name}  [{size_mb:.0f} MB, {elapsed:.1f}s]  →  {raw_path}")

        return Path(raw_path)

    def download_all(
        self,
        force: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Path]:
        """Download all currently ready (non-coming_soon) models.

        Failures are caught and reported individually; the method always
        returns whatever succeeded.

        Args:
            force: Re-download even if cached.
            verbose: Print progress lines.

        Returns:
            Dict mapping model name → local :class:`~pathlib.Path`.
        """
        ready = self.list_ready()
        results: Dict[str, Path] = {}
        failed: List[str] = []

        if verbose:
            print(f"Downloading {len(ready)} model(s)...\n")

        for info in ready:
            try:
                path = self.download(info.name, force=force, verbose=verbose)
                results[info.name] = path
            except Exception as exc:
                failed.append(info.name)
                if verbose:
                    print(f"  ✗ {info.name}  — {exc}")

        if verbose:
            print(f"\n{len(results)}/{len(ready)} model(s) downloaded successfully.")
            if failed:
                print(f"Failed: {failed}")

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Cache management
    # ──────────────────────────────────────────────────────────────────────────

    def is_cached(self, name: str) -> bool:
        """Return ``True`` if the model is already downloaded and the file exists.

        Args:
            name: Registry key.

        Returns:
            ``True`` when the pointer file exists *and* points to a real file.
        """
        pointer = self._pointer_path(name)
        if not pointer.exists():
            return False
        target = Path(pointer.read_text(encoding="utf-8").strip())
        return target.exists()

    def cache_info(self) -> Dict[str, dict]:
        """Return cache status for every registered model.

        Returns:
            Dict mapping model name → a dict with keys:

            - ``cached``  (bool)
            - ``path``    (str or ``None``)
            - ``size_mb`` (float or ``None``)
        """
        result: Dict[str, dict] = {}
        for name in ModelHub._MODELS:
            entry: dict = {"cached": False, "path": None, "size_mb": None}
            pointer = self._pointer_path(name)
            if pointer.exists():
                target = Path(pointer.read_text(encoding="utf-8").strip())
                if target.exists():
                    entry["cached"] = True
                    entry["path"] = str(target)
                    entry["size_mb"] = round(target.stat().st_size / (1024**2), 1)
            result[name] = entry
        return result

    def clear_cache(self, name: Optional[str] = None) -> None:
        """Remove pointer file(s) from the WorldModelLens cache.

        .. note::
            This does **not** delete the underlying HuggingFace download to
            avoid wiping a cache shared with other tools. To fully remove
            the files, delete ``~/.cache/huggingface/hub/`` manually.

        Args:
            name: If given, clear only that model. If ``None``, clear all.
        """
        if name is not None:
            pointer = self._pointer_path(name)
            if pointer.exists():
                pointer.unlink()
        else:
            for n in ModelHub._MODELS:
                self.clear_cache(n)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _pointer_path(self, name: str) -> Path:
        """Return the path to the pointer file for ``name``."""
        return self.cache_dir / f"{name}.pointer"

    def _resolve_pointer(self, name: str) -> Path:
        """Return the target path stored in the pointer file."""
        return Path(self._pointer_path(name).read_text(encoding="utf-8").strip())
