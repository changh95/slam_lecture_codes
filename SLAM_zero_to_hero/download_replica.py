#!/usr/bin/env python3
"""
Download Replica dataset (NICE-SLAM preprocessed) to ~/data/replica/
with tqdm progress bars, then unzip and remove zip file.

This downloads the NICE-SLAM preprocessed Replica sequences with rendered
RGB + depth frames and camera poses - directly usable for SLAM benchmarking
(nvblox, gaussian_splatting_slam, pin_slam, nvblox, etc.).

For the raw mesh+texture (Facebook original), use Replica-Dataset upstream
instead: https://github.com/facebookresearch/Replica-Dataset
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


REPLICA_URLS = {
    # NICE-SLAM preprocessed Replica (~43 GB): 8 scenes with RGB, depth, poses
    # office0, office1, office2, office3, office4, room0, room1, room2
    "Replica.zip": "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip",
}

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
                desc=f"  > {self.filename}",
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


def download_file(url: str, dest: Path) -> Path:
    """Download a file with tqdm progress bar. Skips if already exists."""
    filepath = dest / url.split("/")[-1]

    if filepath.exists():
        size_gb = filepath.stat().st_size / (1024**3)
        print(f"  Skipping {filepath.name} (already exists, {size_gb:.2f} GB)")
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
        for member in tqdm(members, desc=f"  Unzipping {zip_path.name}", ncols=100):
            zf.extract(member, dest)


def main():
    print("=" * 60)
    print("  Replica Dataset Downloader (NICE-SLAM preprocessed)")
    print("  https://github.com/cvg/nice-slam")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Approximate size: ~43 GB")
    print(f"  Contents: 8 scenes (office0-4, room0-2)")
    print(f"    Each with: rgb/*.jpg, depth/*.png, traj.txt\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    zip_files: list[Path] = []

    # --- Download ---
    print("Downloading files...\n")
    for filename, url in REPLICA_URLS.items():
        try:
            zf = download_file(url, DEST_DIR)
            zip_files.append(zf)
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")
            continue

    # --- Unzip ---
    print("\nExtracting archives...\n")
    for zf in zip_files:
        if not zf.exists():
            continue
        try:
            unzip_file(zf, DEST_DIR)
            print(f"  {zf.name} extracted.")
        except zipfile.BadZipFile:
            print(f"  {zf.name} is corrupted, skipping.")
        except Exception as e:
            print(f"  Failed to extract {zf.name}: {e}")

    # --- Cleanup ---
    print("\nRemoving zip files...\n")
    for zf in zip_files:
        if zf.exists():
            zf.unlink()
            print(f"  Removed {zf.name}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Done! Dataset installed at:")
    print(f"     {DEST_DIR}")
    print("=" * 60)

    if DEST_DIR.exists():
        print("\n  Contents:")
        for item in sorted(DEST_DIR.iterdir()):
            kind = "dir " if item.is_dir() else "file"
            print(f"    [{kind}] {item.name}")


if __name__ == "__main__":
    main()
