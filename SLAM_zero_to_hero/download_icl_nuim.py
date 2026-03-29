#!/usr/bin/env python3
"""
Download ICL-NUIM dataset to ~/data/icl_nuim/
with tqdm progress bars, then extract and remove archive files.

Reference: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
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


BASE_URL = "http://www.doc.ic.ac.uk/~ahanda"

ICL_NUIM_URLS = {
    # Living room sequences (TUM Freiburg PNG format)
    "living_room_traj0_frei_png.tar.gz": f"{BASE_URL}/living_room_traj0_frei_png.tar.gz",
    "living_room_traj1_frei_png.tar.gz": f"{BASE_URL}/living_room_traj1_frei_png.tar.gz",
    "living_room_traj2_frei_png.tar.gz": f"{BASE_URL}/living_room_traj2_frei_png.tar.gz",
    "living_room_traj3_frei_png.tar.gz": f"{BASE_URL}/living_room_traj3_frei_png.tar.gz",
    # Living room sequences with noise
    "living_room_traj0n_frei_png.tar.gz": f"{BASE_URL}/living_room_traj0n_frei_png.tar.gz",
    "living_room_traj1n_frei_png.tar.gz": f"{BASE_URL}/living_room_traj1n_frei_png.tar.gz",
    "living_room_traj2n_frei_png.tar.gz": f"{BASE_URL}/living_room_traj2n_frei_png.tar.gz",
    "living_room_traj3n_frei_png.tar.gz": f"{BASE_URL}/living_room_traj3n_frei_png.tar.gz",
    # Office room sequences (TUM Freiburg PNG format)
    "traj0_frei_png.tar.gz": f"{BASE_URL}/traj0_frei_png.tar.gz",
    "traj1_frei_png.tar.gz": f"{BASE_URL}/traj1_frei_png.tar.gz",
    "traj2_frei_png.tar.gz": f"{BASE_URL}/traj2_frei_png.tar.gz",
    "traj3_frei_png.tar.gz": f"{BASE_URL}/traj3_frei_png.tar.gz",
    # Office room sequences with noise
    "traj0n_frei_png.tar.gz": f"{BASE_URL}/traj0n_frei_png.tar.gz",
    "traj1n_frei_png.tar.gz": f"{BASE_URL}/traj1n_frei_png.tar.gz",
    "traj2n_frei_png.tar.gz": f"{BASE_URL}/traj2n_frei_png.tar.gz",
    "traj3n_frei_png.tar.gz": f"{BASE_URL}/traj3n_frei_png.tar.gz",
}

DEST_DIR = Path.home() / "data" / "icl_nuim"


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


def extract_tgz(tgz_path: Path, dest: Path):
    """Extract a .tar.gz file with tqdm progress bar."""
    with tarfile.open(tgz_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"  📦 Extracting {tgz_path.name}", ncols=100):
            tf.extract(member, dest, filter="data")


def main():
    print("=" * 60)
    print("  ICL-NUIM Dataset Downloader")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Sequences:   {len(ICL_NUIM_URLS)}\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    archive_files: list[Path] = []

    # --- Download ---
    print("📥 Downloading files...\n")
    for filename, url in ICL_NUIM_URLS.items():
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

    if DEST_DIR.exists():
        print("\n  Contents:")
        for item in sorted(DEST_DIR.iterdir()):
            kind = "📁" if item.is_dir() else "📄"
            print(f"    {kind} {item.name}")


if __name__ == "__main__":
    main()
