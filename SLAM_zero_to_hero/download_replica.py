#!/usr/bin/env python3
"""
Download Replica dataset to ~/data/replica/
using the official multi-part archive from GitHub releases.

Reference: https://github.com/facebookresearch/Replica-Dataset
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


BASE_URL = "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"

# 17 parts: partaa through partaq
PARTS = [f"replica_v1_0.tar.gz.part{chr(ord('a'))}{chr(ord('a') + i)}" for i in range(17)]

DEST_DIR = Path.home() / "data" / "replica"


class TqdmDownloadHook:
    """Hook class for urllib.request.urlretrieve with tqdm progress."""

    def __init__(self, filename: str):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  ↓ {self.filename}",
                ncols=100,
                miniters=1,
                mininterval=0.5,
                position=0,
                leave=True,
            )
        if total_size > 0:
            self.pbar.update(min(block_size, total_size - self.pbar.n))
        else:
            self.pbar.update(block_size)

    def close(self):
        if self.pbar:
            self.pbar.close()


def download_file(url: str, dest: Path, filename: str) -> Path:
    """Download a file with tqdm progress bar. Skips if already exists."""
    filepath = dest / filename

    if filepath.exists():
        print(f"  ⏭  {filepath.name} already exists, skipping download.")
        return filepath

    hook = TqdmDownloadHook(filepath.name)
    try:
        urllib.request.urlretrieve(url, str(filepath), reporthook=hook)
    finally:
        hook.close()

    return filepath


def main():
    print("=" * 60)
    print("  Replica Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Parts:       {len(PARTS)}\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    part_files: list[Path] = []

    # --- Download ---
    print("📥 Downloading parts...\n")
    for part in PARTS:
        url = f"{BASE_URL}/{part}"
        try:
            pf = download_file(url, DEST_DIR, part)
            part_files.append(pf)
        except Exception as e:
            print(f"  ❌ Failed to download {part}: {e}")
            continue

    # --- Reassemble and extract ---
    print("\n📦 Reassembling and extracting...\n")
    combined = DEST_DIR / "replica_v1_0.tar.gz"
    try:
        # Concatenate all parts
        with open(combined, "wb") as out:
            for pf in sorted(part_files):
                if pf.exists():
                    print(f"  Merging {pf.name}...")
                    with open(pf, "rb") as inp:
                        while True:
                            chunk = inp.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)

        # Extract using tar (pigz if available, else gzip)
        print("\n  Extracting (this may take a while)...")
        subprocess.run(
            ["tar", "-xzf", str(combined), "-C", str(DEST_DIR)],
            check=True,
        )
        print("  ✅ Extraction complete.")
    except Exception as e:
        print(f"  ❌ Failed to extract: {e}")
        print("  Try manually: cat replica_v1_0.tar.gz.part* | tar -xz")

    # --- Cleanup ---
    print("\n🧹 Removing archive files...\n")
    for pf in part_files:
        if pf.exists():
            pf.unlink()
            print(f"  🗑  Removed {pf.name}")
    if combined.exists():
        combined.unlink()
        print(f"  🗑  Removed {combined.name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  ✅ Done! Dataset installed at:")
    print(f"     {DEST_DIR}")
    print("=" * 60)

    if DEST_DIR.exists():
        print("\n  Contents:")
        for item in sorted(DEST_DIR.iterdir()):
            kind = "📁" if item.is_dir() else "📄"
            print(f"    {kind} {item.name}")


if __name__ == "__main__":
    main()
