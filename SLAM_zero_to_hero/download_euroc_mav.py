#!/usr/bin/env python3
"""
Download EuRoC MAV dataset to ~/data/euroc_mav/
with tqdm progress bars, then unzip and remove zip files.

Reference: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
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


BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

EUROC_MAV_URLS = {
    # Machine Hall sequences
    "MH_01_easy.zip": f"{BASE_URL}/machine_hall/MH_01_easy/MH_01_easy.zip",
    "MH_02_easy.zip": f"{BASE_URL}/machine_hall/MH_02_easy/MH_02_easy.zip",
    "MH_03_medium.zip": f"{BASE_URL}/machine_hall/MH_03_medium/MH_03_medium.zip",
    "MH_04_difficult.zip": f"{BASE_URL}/machine_hall/MH_04_difficult/MH_04_difficult.zip",
    "MH_05_difficult.zip": f"{BASE_URL}/machine_hall/MH_05_difficult/MH_05_difficult.zip",
    # Vicon Room 1 sequences
    "V1_01_easy.zip": f"{BASE_URL}/vicon_room1/V1_01_easy/V1_01_easy.zip",
    "V1_02_medium.zip": f"{BASE_URL}/vicon_room1/V1_02_medium/V1_02_medium.zip",
    "V1_03_difficult.zip": f"{BASE_URL}/vicon_room1/V1_03_difficult/V1_03_difficult.zip",
    # Vicon Room 2 sequences
    "V2_01_easy.zip": f"{BASE_URL}/vicon_room2/V2_01_easy/V2_01_easy.zip",
    "V2_02_medium.zip": f"{BASE_URL}/vicon_room2/V2_02_medium/V2_02_medium.zip",
    "V2_03_difficult.zip": f"{BASE_URL}/vicon_room2/V2_03_difficult/V2_03_difficult.zip",
}

DEST_DIR = Path.home() / "data" / "euroc_mav"


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
            )
        downloaded = block_num * block_size
        if total_size > 0:
            self.pbar.update(min(block_size, total_size - self.pbar.n))
        else:
            self.pbar.update(block_size)

    def close(self):
        if self.pbar:
            self.pbar.close()


def download_file(url: str, dest: Path) -> Path:
    """Download a file with tqdm progress bar. Skips if already exists."""
    filepath = dest / url.split("/")[-1]

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
    print("  EuRoC MAV Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Sequences:   {len(EUROC_MAV_URLS)}\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    zip_files: list[Path] = []

    # --- Download ---
    print("📥 Downloading files...\n")
    for filename, url in EUROC_MAV_URLS.items():
        try:
            zf = download_file(url, DEST_DIR)
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

    # Show directory structure (top level)
    if DEST_DIR.exists():
        print("\n  Contents:")
        for item in sorted(DEST_DIR.iterdir()):
            kind = "📁" if item.is_dir() else "📄"
            print(f"    {kind} {item.name}")


if __name__ == "__main__":
    main()
