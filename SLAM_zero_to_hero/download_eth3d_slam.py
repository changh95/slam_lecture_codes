#!/usr/bin/env python3
"""
Download ETH3D SLAM benchmark dataset to ~/data/eth3d_slam/
with tqdm progress bars, then unzip and remove archive files.

Reference: https://www.eth3d.net/slam_overview
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


BASE_URL = "https://www.eth3d.net/data/slam/datasets"

# Training sequences with mono, stereo, and RGB-D modalities
SEQUENCES = [
    "cable_1",
    "cable_2",
    "cable_3",
    "cable_4",
    "camera_shake_1",
    "camera_shake_2",
    "camera_shake_3",
    "camera_shake_4",
    "desk_1",
    "desk_2",
    "desk_3",
    "desk_changing",
    "einstein_global_light_changes_1",
    "einstein_global_light_changes_2",
    "einstein_flashlight",
    "mannequin_1",
    "mannequin_3",
    "mannequin_face_1",
    "mannequin_face_2",
    "plant_1",
    "plant_2",
    "plant_3",
    "plant_4",
    "plant_scene_1",
    "plant_scene_2",
    "plant_scene_3",
    "planar_1",
    "planar_2",
    "planar_3",
    "sfm_bench",
    "sfm_garden",
    "sfm_house_loop",
    "sfm_lab_room_1",
    "sfm_lab_room_2",
    "sofa_1",
    "sofa_2",
    "sofa_3",
    "sofa_4",
    "table_1",
    "table_2",
    "table_3",
    "table_4",
    "table_5",
    "table_6",
    "table_7",
]

# Download mono and stereo modalities (most useful for SLAM)
MODALITIES = ["mono", "stereo"]

DEST_DIR = Path.home() / "data" / "eth3d_slam"


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
    total = len(SEQUENCES) * len(MODALITIES)
    print("=" * 60)
    print("  ETH3D SLAM Benchmark Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Sequences:   {len(SEQUENCES)}")
    print(f"  Modalities:  {', '.join(MODALITIES)}")
    print(f"  Total files: {total}\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    zip_files: list[Path] = []

    # --- Download ---
    print("📥 Downloading files...\n")
    for seq in SEQUENCES:
        for mod in MODALITIES:
            filename = f"{seq}_{mod}.zip"
            url = f"{BASE_URL}/{filename}"
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
