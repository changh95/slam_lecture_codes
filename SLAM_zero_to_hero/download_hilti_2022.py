#!/usr/bin/env python3
"""
Download HILTI SLAM Challenge 2022 dataset to ~/data/hilti_2022/
with tqdm progress bars.

ROS 1 bag files with Hesai PandarXT-32 LiDAR + Alphasense IMU + 5 cameras.
Website: https://hilti-challenge.com/dataset-2022.html
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


S3_BASE = "https://tp-public-facing.s3.eu-north-1.amazonaws.com/Challenges/2022"

# Sequences: name -> approximate size
CHALLENGE_SEQUENCES = {
    "exp01_construction_ground_level": "18 GB",
    "exp02_construction_multilevel": "34 GB",
    "exp03_construction_stairs": "21 GB",
    "exp07_long_corridor": "11 GB",
    "exp09_cupola": "33 GB",
    "exp11_lower_gallery": "12 GB",
    "exp15_attic_to_upper_gallery": "19 GB",
    "exp21_outside_building": "11 GB",
}

ADDITIONAL_SEQUENCES = {
    "exp04_construction_upper_level": "10 GB",
    "exp05_construction_upper_level_2": "10 GB",
    "exp06_construction_upper_level_3": "12 GB",
    "exp10_cupola_2": "27 GB",
    "exp14_basement_2": "6 GB",
    "exp16_attic_to_upper_gallery_2": "15 GB",
    "exp18_corridor_lower_gallery_2": "8 GB",
}

ALL_SEQUENCES = {**CHALLENGE_SEQUENCES, **ADDITIONAL_SEQUENCES}

# Default: smallest sequence for quick testing
DEFAULT_SEQUENCE = "exp14_basement_2"

DEST_DIR = Path.home() / "data" / "hilti_2022"


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
        description="Download HILTI SLAM Challenge 2022 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Challenge sequences:\n"
        + "\n".join(
            f"  {name}  (~{size})"
            for name, size in CHALLENGE_SEQUENCES.items()
        )
        + "\n\nAdditional sequences:\n"
        + "\n".join(
            f"  {name}  (~{size})"
            for name, size in ADDITIONAL_SEQUENCES.items()
        ),
    )
    parser.add_argument(
        "sequences",
        nargs="*",
        default=[DEFAULT_SEQUENCE],
        help=f"Sequence name(s) to download (default: {DEFAULT_SEQUENCE})",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all sequences"
    )
    parser.add_argument(
        "--challenge-only", action="store_true",
        help="Download only the 8 challenge sequences",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEST_DIR,
        help=f"Destination directory (default: {DEST_DIR})",
    )
    args = parser.parse_args()

    if args.all:
        sequences = list(ALL_SEQUENCES.keys())
    elif args.challenge_only:
        sequences = list(CHALLENGE_SEQUENCES.keys())
    else:
        sequences = args.sequences

    # Validate sequence names
    for s in sequences:
        if s not in ALL_SEQUENCES:
            print(f"  Unknown sequence: {s}")
            print(f"  Available: {', '.join(ALL_SEQUENCES.keys())}")
            sys.exit(1)

    total_size = sum(
        int(ALL_SEQUENCES[s].split()[0])
        for s in sequences
    )

    print("=" * 60)
    print("  HILTI SLAM Challenge 2022 Dataset Downloader")
    print("  https://hilti-challenge.com/dataset-2022.html")
    print("=" * 60)
    print(f"\n  Destination: {args.dest}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Estimated total size: ~{total_size} GB\n")

    args.dest.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for seq in sequences:
        url = f"{S3_BASE}/{seq}.bag"
        size = ALL_SEQUENCES[seq]
        print(f"\n--- {seq} (~{size}) ---\n")

        try:
            f = download_file(url, args.dest)
            downloaded.append(f)
        except Exception as e:
            print(f"  Failed to download {seq}: {e}")
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
