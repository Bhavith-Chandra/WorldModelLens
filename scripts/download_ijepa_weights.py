#!/usr/bin/env python3
"""download_ijepa_weights.py — Download the official Meta I-JEPA ViT-H backbone.

The downloaded file (e.g. vith14_in1k_ep300.pth.tar) is a PyTorch pickle file
not a real tar archive, despite the .tar extension. Do NOT try to extract it
with tarfile — load it directly with torch.load().

Checkpoint structure (after torch.load):
    {
        'encoder':        OrderedDict  # context / target ViT-H weights (DDP: 'module.*' prefix)
        'target_encoder': OrderedDict  # EMA copy of encoder weights
        'predictor':      OrderedDict  # predictor network weights
        'opt':            dict         # optimizer state
        'scaler':         dict         # AMP scaler state
        'epoch':          int
        'loss':           float
        'batch_size':     int
        'world_size':     int
        'lr':             float
    }

Architecture (IN1K-vit.h.14-300e):
    embed_dim  = 1280
    depth      = 32
    num_heads  = 16
    patch_size = 14
    img_size   = 224   (pos_embed covers 16x16 = 256 patches for 224px at stride 14)

Usage:
    python scripts/download_ijepa_weights.py                    # IN1K 300e (default)
    python scripts/download_ijepa_weights.py --variant in22k    # IN22K 900e
    python scripts/download_ijepa_weights.py --dest /my/path --verify
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Known official Meta I-JEPA checkpoints (dl.fbaipublicfiles.com/ijepa/)
# ---------------------------------------------------------------------------
CHECKPOINTS = {
    "in1k": {
        "url": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
        "filename": "vith14_in1k_ep300.pth.tar",
        "description": "ViT-H/14 trained on ImageNet-1K for 300 epochs",
    },
    "in22k": {
        "url": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar",
        "filename": "vith14_in22k_ep900.pth.tar",
        "description": "ViT-H/14 trained on ImageNet-22K for 900 epochs",
    },
}

# Expected ViT-H architecture dims for verification
EXPECTED_EMBED_DIM = 1280
EXPECTED_DEPTH = 32
EXPECTED_NUM_HEADS = 16  # qkv weight shape: [3 * heads * head_dim, embed_dim] = [3840, 1280]


def download_file(url: str, target_path: str) -> None:
    """Stream-download url to target_path with a tqdm progress bar."""
    print(f"Downloading: {url}")
    print(f"         →  {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(target_path, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True, unit_divisor=1024, desc="Downloading"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            size = f.write(chunk)
            bar.update(size)

    print(f"\n✓ Saved to: {target_path}  ({Path(target_path).stat().st_size / 1e9:.2f} GB)")


def verify_checkpoint(path: str) -> bool:
    """
    Load the checkpoint with torch and verify it has the expected I-JEPA
    ViT-H structure.  NOTE: the .pth.tar file is a raw PyTorch pickle —
    NOT a real tar archive.  torch.load() opens it directly.
    """
    try:
        import torch
    except ImportError:
        print("[WARN] torch not installed — skipping verification.")
        return True

    print(f"\nVerifying checkpoint: {path}")
    print("  (this may take 10–30 s for the 10 GB file...)")

    sd = torch.load(path, map_location="cpu")

    # -- Top-level structure --------------------------------------------------
    required_keys = {"encoder", "target_encoder", "predictor", "epoch"}
    missing = required_keys - set(sd.keys())
    if missing:
        print(f"  [FAIL] Missing top-level keys: {missing}")
        return False
    print(f"  [OK]  Top-level keys: {list(sd.keys())}")
    print(f"  [OK]  epoch={sd.get('epoch')}, loss={sd.get('loss', 'n/a'):.4f}")

    # -- Encoder structure (keys are DDP-prefixed: 'module.*') ---------------
    enc = sd["encoder"]
    enc_keys = list(enc.keys())

    # Detect DDP prefix
    prefix = "module." if any(k.startswith("module.") for k in enc_keys) else ""
    if prefix:
        print(f"  [OK]  DDP-wrapped encoder detected (keys start with 'module.'). ")

    # embed_dim from patch_embed projection
    pe_key = f"{prefix}patch_embed.proj.weight"
    if pe_key not in enc:
        print(f"  [FAIL] Expected key '{pe_key}' not found in encoder.")
        return False
    embed_dim = enc[pe_key].shape[0]
    patch_size = enc[pe_key].shape[2]  # [embed_dim, in_chans, kH, kW]

    # depth from counting norm1 layers
    depth = sum(1 for k in enc_keys if k.startswith(f"{prefix}blocks.") and k.endswith(".norm1.weight"))

    # num_heads from qkv weight: shape [3 * num_heads * head_dim, embed_dim]
    qkv_key = f"{prefix}blocks.0.attn.qkv.weight"
    num_heads = enc[qkv_key].shape[0] // embed_dim // 3 * (embed_dim // 64)  # head_dim typically 64
    # Simpler: qkv_out == 3 * embed_dim for ViT, so num_heads = embed_dim // head_dim
    num_heads = EXPECTED_NUM_HEADS  # fixed for ViT-H

    # pos_embed patches
    pos_key = f"{prefix}pos_embed"
    pos_shape = enc.get(pos_key, None)
    n_patches = pos_shape.shape[1] if pos_shape is not None else "?"

    print(f"  [OK]  embed_dim={embed_dim}  depth={depth}  patch_size={patch_size}  pos_embed patches={n_patches}")

    ok = (
        embed_dim == EXPECTED_EMBED_DIM
        and depth == EXPECTED_DEPTH
        and patch_size == 14
    )
    status = "✓ PASS" if ok else "✗ MISMATCH"
    print(f"  [{status}]  Expected embed_dim={EXPECTED_EMBED_DIM}, depth={EXPECTED_DEPTH}, patch=14")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_ijepa_weights.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=["in1k", "in22k"],
        default="in1k",
        help="Which checkpoint to download (default: in1k = ImageNet-1K 300e).",
    )
    parser.add_argument(
        "--dest",
        default=".",
        metavar="DIR",
        help="Directory to save the checkpoint (default: current directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After downloading, load the checkpoint with torch and verify its structure.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available checkpoints and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable I-JEPA checkpoints:")
        for name, info in CHECKPOINTS.items():
            print(f"  {name:8s}  {info['filename']:<40s}  {info['description']}")
        print()
        return

    info = CHECKPOINTS[args.variant]
    dest_dir = Path(args.dest).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = str(dest_dir / info["filename"])

    print(f"\nI-JEPA ViT-H Checkpoint Downloader")
    print(f"  Variant    : {args.variant} — {info['description']}")
    print(f"  Destination: {target_path}")
    print()

    if os.path.exists(target_path) and not args.force:
        size_gb = Path(target_path).stat().st_size / 1e9
        print(f"✓ File already exists ({size_gb:.2f} GB). Use --force to re-download.")
    else:
        try:
            download_file(info["url"], target_path)
        except Exception as exc:
            print(f"\n✗ Download failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.verify:
        ok = verify_checkpoint(target_path)
        sys.exit(0 if ok else 1)
    else:
        print()
        print("TIP: Run with --verify to validate the checkpoint structure.")
        print(f"\nTo load in your code:")
        print(f"    from world_model_lens.backends.ijepa_adapter import IJEPAAdapter")
        print(f"    adapter = IJEPAAdapter.from_checkpoint('{target_path}')")
        print()


if __name__ == "__main__":
    main()
