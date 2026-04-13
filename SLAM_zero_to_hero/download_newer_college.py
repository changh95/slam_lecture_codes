#!/usr/bin/env python3
"""
Download Newer College Dataset information and helper.

NOTE: The Newer College Dataset requires submitting an access form.
Direct download URLs are not publicly available.

Website: https://ori-drs.github.io/newer-college-dataset/
Download form: https://ori-drs.github.io/newer-college-dataset/download/
Contact: newercollegedataset@robots.ox.ac.uk

Sensors:
  Stereo-Cam (2020): Ouster OS-1 64-beam LiDAR, RealSense D435i stereo, IMU
  Multi-Cam (2021): Ouster OS-0 128-beam LiDAR, 4x Alphasense cameras, IMU

This script provides dataset information and can download files once you
have obtained the download links via the access form.
"""

import os
import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


DEST_DIR = Path.home() / "data" / "newer_college"

STEREO_CAM_SEQUENCES = {
    "short_experiment": "Smallest sequence. ~0.5 loop of quad + mid-section + parkland.",
    "long_experiment": "3.5 loops quad + 2 loops mid-section (~6.7 km).",
    "quad_with_dynamics": "4 loops with increasingly aggressive motion.",
    "dynamic_spinning": "Aggressive spinning up to 3.5 rad/sec.",
    "parkland_mound": "Laps of parkland and mound staircase.",
}

MULTI_CAM_SEQUENCES = {
    "Stairs": "Shortest (118s). Staircase traversal.",
    "Quad-Easy": "Smallest multi-cam (198s). Courtyard loop.",
    "Quad-Medium": "190s. Courtyard with moderate dynamics.",
    "Quad-Hard": "187s. Courtyard with fast motion.",
    "Cloister": "278s. Indoor cloister walk.",
    "Maths-Easy": "216s. Mathematics building.",
    "Maths-Medium": "176s. Mathematics building, moderate.",
    "Maths-Hard": "243s. Mathematics building, fast motion.",
    "Park": "1567s (~26 min). Longest sequence, outdoor park.",
}


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
        print(f"  Skipping {filepath.name} (already exists)")
        return filepath

    hook = TqdmDownloadHook(filepath.name)
    try:
        urllib.request.urlretrieve(url, str(filepath), reporthook=hook)
    finally:
        hook.close()

    return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Newer College Dataset downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
NOTE: This dataset requires access form submission first.
  1. Go to https://ori-drs.github.io/newer-college-dataset/download/
  2. Fill the form (name, institution, use case)
  3. You will receive download links via email
  4. Run this script with --url to download:
     python3 download_newer_college.py --url <received_url> [<url2> ...]

Stereo-Cam sequences (2020, Ouster OS-1 64-beam):
"""
        + "\n".join(f"  {k}: {v}" for k, v in STEREO_CAM_SEQUENCES.items())
        + "\n\nMulti-Cam sequences (2021, Ouster OS-0 128-beam):\n"
        + "\n".join(f"  {k}: {v}" for k, v in MULTI_CAM_SEQUENCES.items()),
    )
    parser.add_argument(
        "--url", nargs="+",
        help="Download URL(s) received from the access form",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEST_DIR,
        help=f"Destination directory (default: {DEST_DIR})",
    )
    parser.add_argument(
        "--info", action="store_true", default=False,
        help="Show dataset information and exit",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Newer College Dataset")
    print("  https://ori-drs.github.io/newer-college-dataset/")
    print("=" * 60)

    if args.info or args.url is None:
        print("\n  Stereo-Cam sequences (2020, Ouster OS-1 64-beam):")
        for name, desc in STEREO_CAM_SEQUENCES.items():
            print(f"    - {name}: {desc}")

        print("\n  Multi-Cam sequences (2021, Ouster OS-0 128-beam):")
        for name, desc in MULTI_CAM_SEQUENCES.items():
            print(f"    - {name}: {desc}")

        print("\n  To download, you must first request access:")
        print("    1. Visit https://ori-drs.github.io/newer-college-dataset/download/")
        print("    2. Submit the access form")
        print("    3. Receive download links via email")
        print("    4. Run: python3 download_newer_college.py --url <link1> [<link2> ...]")
        print()
        return

    # Download from provided URLs
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"\n  Destination: {args.dest}\n")

    for url in args.url:
        print(f"Downloading: {url}\n")
        try:
            download_file(url, args.dest)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("  Done! Dataset installed at:")
    print(f"     {args.dest}")
    print("=" * 60)

    if args.dest.exists():
        print("\n  Contents:")
        for item in sorted(args.dest.iterdir()):
            size_mb = item.stat().st_size / (1024 * 1024)
            if size_mb > 1024:
                size_str = f"{size_mb / 1024:.1f} GB"
            else:
                size_str = f"{size_mb:.0f} MB"
            print(f"    {item.name}  ({size_str})")


if __name__ == "__main__":
    main()
