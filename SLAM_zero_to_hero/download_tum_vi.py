#!/usr/bin/env python3
"""
Download TUM-VI dataset (EuRoC format, 512x512) to ~/data/tum_vi/
with tqdm progress bars.

Reference: https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
License:   Creative Commons 4.0 Attribution
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


BASE_URL = "https://vision.in.tum.de/tumvi/exported/euroc/512_16"

# EuRoC-format .tar archives (smaller than rosbag format)
TUM_VI_URLS = {
    # Room sequences (~2-3 GB each)
    "dataset-room1_512_16.tar": f"{BASE_URL}/dataset-room1_512_16.tar",
    "dataset-room2_512_16.tar": f"{BASE_URL}/dataset-room2_512_16.tar",
    "dataset-room3_512_16.tar": f"{BASE_URL}/dataset-room3_512_16.tar",
    "dataset-room4_512_16.tar": f"{BASE_URL}/dataset-room4_512_16.tar",
    "dataset-room5_512_16.tar": f"{BASE_URL}/dataset-room5_512_16.tar",
    "dataset-room6_512_16.tar": f"{BASE_URL}/dataset-room6_512_16.tar",
    # Corridor sequences (~3-6 GB each)
    "dataset-corridor1_512_16.tar": f"{BASE_URL}/dataset-corridor1_512_16.tar",
    "dataset-corridor2_512_16.tar": f"{BASE_URL}/dataset-corridor2_512_16.tar",
    "dataset-corridor3_512_16.tar": f"{BASE_URL}/dataset-corridor3_512_16.tar",
    "dataset-corridor4_512_16.tar": f"{BASE_URL}/dataset-corridor4_512_16.tar",
    "dataset-corridor5_512_16.tar": f"{BASE_URL}/dataset-corridor5_512_16.tar",
    # Magistrale sequences (~8-15 GB each)
    "dataset-magistrale1_512_16.tar": f"{BASE_URL}/dataset-magistrale1_512_16.tar",
    "dataset-magistrale2_512_16.tar": f"{BASE_URL}/dataset-magistrale2_512_16.tar",
    "dataset-magistrale3_512_16.tar": f"{BASE_URL}/dataset-magistrale3_512_16.tar",
    "dataset-magistrale4_512_16.tar": f"{BASE_URL}/dataset-magistrale4_512_16.tar",
    "dataset-magistrale5_512_16.tar": f"{BASE_URL}/dataset-magistrale5_512_16.tar",
    "dataset-magistrale6_512_16.tar": f"{BASE_URL}/dataset-magistrale6_512_16.tar",
    # Slides sequences (~3-5 GB each)
    "dataset-slides1_512_16.tar": f"{BASE_URL}/dataset-slides1_512_16.tar",
    "dataset-slides2_512_16.tar": f"{BASE_URL}/dataset-slides2_512_16.tar",
    "dataset-slides3_512_16.tar": f"{BASE_URL}/dataset-slides3_512_16.tar",
}

DEST_DIR = Path.home() / "data" / "tum_vi"


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


def extract_tar(tar_path: Path, dest: Path):
    """Extract a .tar file with tqdm progress bar."""
    with tarfile.open(tar_path, "r:") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"  📦 Extracting {tar_path.name}", ncols=100):
            tf.extract(member, dest, filter="data")


def main():
    print("=" * 60)
    print("  TUM-VI Dataset Downloader (EuRoC format, 512x512)")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Sequences:   {len(TUM_VI_URLS)}")
    print(f"\n  Note: Outdoor sequences are excluded (13-29 GB each).")
    print(f"  To download outdoors, add them manually.\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    archive_files: list[Path] = []

    # --- Download ---
    print("📥 Downloading files...\n")
    for filename, url in TUM_VI_URLS.items():
        try:
            af = download_file(url, DEST_DIR, filename)
            archive_files.append(af)
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")
            continue

    # --- Extract ---
    print("\n📦 Extracting archives...\n")
    for af in archive_files:
        if not af.exists():
            continue
        try:
            extract_tar(af, DEST_DIR)
            print(f"  ✅ {af.name} extracted.")
        except tarfile.TarError:
            print(f"  ❌ {af.name} is corrupted, skipping.")
        except Exception as e:
            print(f"  ❌ Failed to extract {af.name}: {e}")

    # --- Cleanup ---
    print("\n🧹 Removing archive files...\n")
    for af in archive_files:
        if af.exists():
            af.unlink()
            print(f"  🗑  Removed {af.name}")

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
