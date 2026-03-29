#!/usr/bin/env python3
"""
Download TUM RGB-D dataset to ~/data/tum_rgbd/
with tqdm progress bars, then extract and remove archive files.

Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
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


BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"

TUM_RGBD_URLS = {
    # Freiburg 1 sequences (Kinect v1, handheld)
    "rgbd_dataset_freiburg1_xyz.tgz": f"{BASE_URL}/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
    "rgbd_dataset_freiburg1_rpy.tgz": f"{BASE_URL}/freiburg1/rgbd_dataset_freiburg1_rpy.tgz",
    "rgbd_dataset_freiburg1_desk.tgz": f"{BASE_URL}/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "rgbd_dataset_freiburg1_desk2.tgz": f"{BASE_URL}/freiburg1/rgbd_dataset_freiburg1_desk2.tgz",
    "rgbd_dataset_freiburg1_room.tgz": f"{BASE_URL}/freiburg1/rgbd_dataset_freiburg1_room.tgz",
    # Freiburg 2 sequences (Kinect v1, robot)
    "rgbd_dataset_freiburg2_xyz.tgz": f"{BASE_URL}/freiburg2/rgbd_dataset_freiburg2_xyz.tgz",
    "rgbd_dataset_freiburg2_rpy.tgz": f"{BASE_URL}/freiburg2/rgbd_dataset_freiburg2_rpy.tgz",
    "rgbd_dataset_freiburg2_desk.tgz": f"{BASE_URL}/freiburg2/rgbd_dataset_freiburg2_desk.tgz",
    "rgbd_dataset_freiburg2_desk_with_person.tgz": f"{BASE_URL}/freiburg2/rgbd_dataset_freiburg2_desk_with_person.tgz",
    # Freiburg 3 sequences (Kinect v1, structure vs texture)
    "rgbd_dataset_freiburg3_long_office_household.tgz": f"{BASE_URL}/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz",
    "rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz": f"{BASE_URL}/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz",
    "rgbd_dataset_freiburg3_structure_texture_far.tgz": f"{BASE_URL}/freiburg3/rgbd_dataset_freiburg3_structure_texture_far.tgz",
    "rgbd_dataset_freiburg3_structure_texture_near.tgz": f"{BASE_URL}/freiburg3/rgbd_dataset_freiburg3_structure_texture_near.tgz",
    "rgbd_dataset_freiburg3_walking_xyz.tgz": f"{BASE_URL}/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz",
    "rgbd_dataset_freiburg3_walking_halfsphere.tgz": f"{BASE_URL}/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz",
}

DEST_DIR = Path.home() / "data" / "tum_rgbd"


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


def extract_tgz(tgz_path: Path, dest: Path):
    """Extract a .tgz file with tqdm progress bar."""
    with tarfile.open(tgz_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"  📦 Extracting {tgz_path.name}", ncols=100):
            tf.extract(member, dest, filter="data")


def main():
    print("=" * 60)
    print("  TUM RGB-D Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Sequences:   {len(TUM_RGBD_URLS)}\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    archive_files: list[Path] = []

    # --- Download ---
    print("📥 Downloading files...\n")
    for filename, url in TUM_RGBD_URLS.items():
        try:
            af = download_file(url, DEST_DIR)
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
            extract_tgz(af, DEST_DIR)
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

    # Show directory structure (top level)
    if DEST_DIR.exists():
        print("\n  Contents:")
        for item in sorted(DEST_DIR.iterdir()):
            kind = "📁" if item.is_dir() else "📄"
            print(f"    {kind} {item.name}")


if __name__ == "__main__":
    main()
