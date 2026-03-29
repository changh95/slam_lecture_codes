#!/usr/bin/env python3
"""
Download Redwood Indoor LiDAR-RGBD dataset to ~/data/redwood/
with tqdm progress bars, then unzip and remove archive files.

Reference: http://redwood-data.org/indoor_lidar_rgbd/download.html

Contains 5 indoor environments:
- LiDAR ground truth scans (.ply)
- RGB-D video sequences (JPEG color + 16-bit PNG depth at 640x480 @ 30Hz)
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


BASE_URL = "http://redwood-data.org/indoor_lidar_rgbd"

REDWOOD_URLS = {
    # RGB-D sequences
    "apartment.zip": f"{BASE_URL}/data/apartment.zip",
    "bedroom.zip": f"{BASE_URL}/data/bedroom.zip",
    "boardroom.zip": f"{BASE_URL}/data/boardroom.zip",
    "lobby.zip": f"{BASE_URL}/data/lobby.zip",
    "loft.zip": f"{BASE_URL}/data/loft.zip",
}

DEST_DIR = Path.home() / "data" / "redwood"


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


def unzip_file(zip_path: Path, dest: Path):
    """Unzip a file with tqdm progress bar."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=f"  📦 Unzipping {zip_path.name}", ncols=100):
            zf.extract(member, dest)


def main():
    print("=" * 60)
    print("  Redwood Indoor LiDAR-RGBD Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Scenes:      {len(REDWOOD_URLS)}\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    zip_files: list[Path] = []

    # --- Download ---
    print("📥 Downloading files...\n")
    for filename, url in REDWOOD_URLS.items():
        try:
            zf = download_file(url, DEST_DIR, filename)
            zip_files.append(zf)
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")
            continue

    # --- Unzip ---
    print("\n📦 Extracting archives...\n")
    for zf in zip_files:
        if not zf.exists():
            continue
        try:
            unzip_file(zf, DEST_DIR)
            print(f"  ✅ {zf.name} extracted.")
        except zipfile.BadZipFile:
            print(f"  ❌ {zf.name} is corrupted, skipping.")
        except Exception as e:
            print(f"  ❌ Failed to extract {zf.name}: {e}")

    # --- Cleanup ---
    print("\n🧹 Removing zip files...\n")
    for zf in zip_files:
        if zf.exists():
            zf.unlink()
            print(f"  🗑  Removed {zf.name}")

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
