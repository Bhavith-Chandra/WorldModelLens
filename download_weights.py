#!/usr/bin/env python3
"""download_weights.py — Pull known-good world model checkpoints from HuggingFace.

Run this right after cloning WorldModelLens to pre-download pretrained
checkpoints so demo scripts work immediately:

    python download_weights.py                   # download ALL ready models
    python download_weights.py iris-atari-pong   # download ONE specific model
    python download_weights.py --list            # list downloadable models
    python download_weights.py --list-all        # include coming-soon models
    python download_weights.py --cache-info      # inspect what is cached locally
    python download_weights.py --force           # force re-download

Checkpoints are cached at ~/.cache/world_model_lens/weights/ by default.
Override with --cache-dir <path>.

Currently available (real PyTorch weights, official authors):
  iris-atari-breakout   IRIS on Atari Breakout  (eloialonso/iris @ HuggingFace)
  iris-atari-pong       IRIS on Atari Pong
  iris-atari-seaquest   IRIS on Atari Seaquest
  iris-atari-freeway    IRIS on Atari Freeway
  iris-atari-alien      IRIS on Atari Alien
  iris-atari-qbert      IRIS on Atari Q*bert
  iris-atari-ms-pacman  IRIS on Atari Ms. Pac-Man

Coming soon:
  dreamerv3-*  (JAX-only; no public PyTorch weights available)
  tdmpc2-*     (real HF weights exist; adapter key-mapping in progress)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table


# Allow running from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))


def _build_table_rich(models, title: str) -> None:
    """Print a Rich-formatted table."""
    

    console = Console()
    table = Table(title=title, show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Backend", style="magenta")
    table.add_column("Environment", style="green")
    table.add_column("Status")
    table.add_column("Description")

    for m in models:
        if m.coming_soon:
            status = "[dim]coming soon[/dim]"
        else:
            status = "[bold green]✓ ready[/bold green]"
        table.add_row(
            m.name,
            m.backend,
            m.environment,
            status,
            m.description[:60],
        )
    console.print(table)


def _build_table_plain(models) -> None:
    """Print a plain-text table (fallback when Rich is unavailable)."""
    col_w = 42
    print(f"\n{'Name':<{col_w}}  {'Backend':<12}  {'Env':<32}  Status")
    print("-" * (col_w + 52))
    for m in models:
        status = "coming soon" if m.coming_soon else "✓ ready"
        env = m.environment[:30]
        print(f"{m.name:<{col_w}}  {m.backend:<12}  {env:<32}  {status}")
    print()


def _print_table(models, title: str = "WorldModelLens — Known-Good Checkpoints") -> None:
    try:
        _build_table_rich(models, title)
    except ImportError:
        _build_table_plain(models)


def _print_cache(info: dict) -> None:
    try:
        console = Console()
        table = Table(title="WorldModelLens — Local Cache", show_lines=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Cached", style="green")
        table.add_column("Size (MB)")
        table.add_column("Path", overflow="fold")

        for name, entry in info.items():
            cached_str = "[green]yes[/green]" if entry["cached"] else "[dim]no[/dim]"
            size_str = str(entry["size_mb"]) if entry["size_mb"] else "—"
            path_str = entry["path"] or "—"
            table.add_row(name, cached_str, size_str, path_str)
        console.print(table)
    except ImportError:
        print(f"\n{'Model':<42}  {'Cached':<8}  {'Size MB':<10}  Path")
        print("-" * 100)
        for name, entry in info.items():
            cached = "yes" if entry["cached"] else "no"
            size = str(entry["size_mb"]) if entry["size_mb"] else "—"
            path = entry["path"] or "—"
            print(f"{name:<42}  {cached:<8}  {size:<10}  {path}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_weights.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        metavar="MODEL_KEY",
        help=(
            "Registry key of the model to download "
            "(e.g. iris-atari-breakout). "
            "Omit to download all currently available models."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all downloadable (ready) models and exit.",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all registered models, including coming-soon, and exit.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Override default cache directory (~/.cache/world_model_lens/weights/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the checkpoint is already cached.",
    )
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Show local cache status for all registered models and exit.",
    )
    args = parser.parse_args()

    # ── Import the package ────────────────────────────────────────────────────
    try:
        from world_model_lens.hub.model_hub import ModelHub  # noqa: F401
        from world_model_lens.hub.weights_downloader import WeightsDownloader
    except ImportError as exc:
        print(f"\n[ERROR] Could not import WorldModelLens: {exc}")
        print(
            "Make sure you have installed the package:\n"
            "  pip install -e .\n"
            "or run from the repository root."
        )
        sys.exit(1)

    dl = WeightsDownloader(cache_dir=args.cache_dir)

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        _print_table(dl.list_ready(), title="WorldModelLens — Ready for Download")
        return

    # ── --list-all ────────────────────────────────────────────────────────────
    if args.list_all:
        _print_table(dl.list_all(), title="WorldModelLens — All Registered Models")
        return

    # ── --cache-info ──────────────────────────────────────────────────────────
    if args.cache_info:
        _print_cache(dl.cache_info())
        return

    # ── Download a specific model ─────────────────────────────────────────────
    if args.model:
        print(f"\nDownloading: {args.model}\n")
        try:
            path = dl.download(args.model, force=args.force)
            print(f"\n✓ Checkpoint ready at: {path}\n")
        except NotImplementedError as exc:
            print(f"\n[NOT AVAILABLE]\n{exc}\n")
            sys.exit(1)
        except KeyError as exc:
            print(f"\n[ERROR] {exc}")
            print(
                "\nRun  python download_weights.py --list-all  "
                "to see all registered keys.\n"
            )
            sys.exit(1)
        except Exception as exc:
            print(f"\n[ERROR] Download failed: {exc}\n")
            sys.exit(1)
        return

    # ── Default: download all ready models ────────────────────────────────────
    ready = dl.list_ready()
    if not ready:
        print(
            "\nNo models are currently available for download.\n"
            "Run  python download_weights.py --list-all  for the full registry.\n"
        )
        return

    print(f"\nDownloading {len(ready)} model(s)...\n")
    results = dl.download_all(force=args.force)
    print(
        f"\n✓ Done. {len(results)}/{len(ready)} model(s) downloaded successfully."
    )
    for name, path in results.items():
        print(f"  {name}:  {path}")
    print()


if __name__ == "__main__":
    main()
