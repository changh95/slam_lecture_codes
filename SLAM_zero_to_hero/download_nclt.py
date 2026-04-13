#!/usr/bin/env python3
"""
Download NCLT (North Campus Long-Term) dataset to ~/data/nclt/
with tqdm progress bars, then extract and remove archives.

Sensor data includes Velodyne HDL-32E LiDAR, IMU, GPS, and wheel odometry.
Website: http://robots.engin.umich.edu/nclt/
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


S3_BASE = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"

# Available sessions (date -> approximate velodyne size)
SESSIONS = {
    "2012-01-08": "5 GB",
    "2012-01-15": "7 GB",
    "2012-01-22": "9 GB",
    "2012-02-02": "5 GB",
    "2012-02-04": "9 GB",
    "2012-02-05": "6 GB",
    "2012-02-12": "4 GB",
    "2012-02-18": "4 GB",
    "2012-02-19": "5 GB",
    "2012-03-17": "4 GB",
    "2012-03-25": "5 GB",
    "2012-03-31": "5 GB",
    "2012-04-29": "5 GB",
    "2012-05-11": "5 GB",
    "2012-05-26": "7 GB",
    "2012-06-15": "6 GB",
    "2012-08-04": "7 GB",
    "2012-08-20": "5 GB",
    "2012-09-28": "7 GB",
    "2012-10-28": "5 GB",
    "2012-11-04": "6 GB",
    "2012-11-16": "5 GB",
    "2012-11-17": "4 GB",
    "2012-12-01": "4 GB",
    "2013-01-10": "3 GB",
    "2013-02-23": "5 GB",
    "2013-04-05": "5 GB",
}

# Default: smallest session for quick testing
DEFAULT_SESSION = "2013-01-10"

DEST_DIR = Path.home() / "data" / "nclt"


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


def extract_tar(tar_path: Path, dest: Path):
    """Extract a tar.gz file with progress."""
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"  Extracting {tar_path.name}", ncols=100):
            tf.extract(member, dest)


def get_session_urls(session: str) -> dict:
    """Get download URLs for a given session date."""
    return {
        "sensor_data": f"{S3_BASE}/sensor_data/{session}_sen.tar.gz",
        "velodyne": f"{S3_BASE}/velodyne_data/{session}_vel.tar.gz",
        "ground_truth": f"{S3_BASE}/ground_truth/groundtruth_{session}.csv",
        "covariance": f"{S3_BASE}/covariance/cov_{session}.csv",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NCLT dataset sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available sessions:\n" + "\n".join(
            f"  {date}  (~{size} velodyne)" for date, size in SESSIONS.items()
        ),
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        default=[DEFAULT_SESSION],
        help=f"Session date(s) to download (default: {DEFAULT_SESSION})",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all sessions"
    )
    parser.add_argument(
        "--no-velodyne", action="store_true",
        help="Skip Velodyne LiDAR data (large files)",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEST_DIR,
        help=f"Destination directory (default: {DEST_DIR})",
    )
    args = parser.parse_args()

    sessions = list(SESSIONS.keys()) if args.all else args.sessions

    # Validate session names
    for s in sessions:
        if s not in SESSIONS:
            print(f"  Unknown session: {s}")
            print(f"  Available: {', '.join(SESSIONS.keys())}")
            sys.exit(1)

    print("=" * 60)
    print("  NCLT Dataset Downloader")
    print("  http://robots.engin.umich.edu/nclt/")
    print("=" * 60)
    print(f"\n  Destination: {args.dest}")
    print(f"  Sessions: {', '.join(sessions)}\n")

    for session in sessions:
        session_dir = args.dest / session
        session_dir.mkdir(parents=True, exist_ok=True)
        urls = get_session_urls(session)

        print(f"\n--- Session: {session} (~{SESSIONS[session]} velodyne) ---\n")

        archives = []

        # Download sensor data
        print("Downloading sensor data (IMU, GPS, odometry)...")
        try:
            f = download_file(urls["sensor_data"], session_dir)
            archives.append(f)
        except Exception as e:
            print(f"  Failed: {e}")

        # Download velodyne
        if not args.no_velodyne:
            print("Downloading Velodyne HDL-32E LiDAR data...")
            try:
                f = download_file(urls["velodyne"], session_dir)
                archives.append(f)
            except Exception as e:
                print(f"  Failed: {e}")

        # Download ground truth
        print("Downloading ground truth...")
        try:
            download_file(urls["ground_truth"], session_dir)
        except Exception as e:
            print(f"  Failed: {e}")

        # Download covariance
        print("Downloading covariance...")
        try:
            download_file(urls["covariance"], session_dir)
        except Exception as e:
            print(f"  Failed: {e}")

        # Extract tar.gz archives
        print("\nExtracting archives...")
        for archive in archives:
            if archive.exists() and archive.suffix == ".gz":
                try:
                    extract_tar(archive, session_dir)
                    print(f"  Extracted {archive.name}")
                    archive.unlink()
                    print(f"  Removed {archive.name}")
                except Exception as e:
                    print(f"  Failed to extract {archive.name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  Done! Dataset installed at:")
    print(f"     {args.dest}")
    print("=" * 60)

    if args.dest.exists():
        print("\n  Contents:")
        for item in sorted(args.dest.iterdir()):
            kind = "dir " if item.is_dir() else "file"
            print(f"    [{kind}] {item.name}")


if __name__ == "__main__":
    main()
