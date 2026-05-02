#!/usr/bin/env python3
"""
Download the cow_and_lady_dataset rosbag from ETH ASL (used with voxblox).
Destination: ~/data/cow_and_lady/

Tries multiple mirror URLs in order, with retries and a minimum size check.
"""

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


# Fallback URLs in priority order.
DATA_BAG_URLS = [
    # ETH Research Collection permanent handle
    # Landing page: https://www.research-collection.ethz.ch/handle/20.500.11850/721636
    "https://www.research-collection.ethz.ch/bitstreams/bfb68f88-fcb2-4e09-aa53-434d9162cef5/download",
    # ASL datasets server (legacy direct link)
    "http://robotics.ethz.ch/~asl-datasets/iros_2017_voxblox/data.bag",
    # ASL projects mirror
    "https://projects.asl.ethz.ch/datasets/voxblox/data.bag",
]

DATA_BAG_FILE = "data.bag"
DEST_DIR = Path.home() / "data" / "cow_and_lady"

CONNECT_TIMEOUT = 30        # seconds per connection attempt
MAX_RETRIES = 3             # retries per URL
MIN_SIZE_BYTES = 1 * 1024 ** 3  # 1 GB minimum (actual bag is ~4.6 GB)


class TqdmDownloadHook:
    def __init__(self, filename: str):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
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
                leave=True,
            )
        if total_size > 0:
            self.pbar.update(min(block_size, total_size - self.pbar.n))
        else:
            self.pbar.update(block_size)

    def close(self):
        if self.pbar:
            self.pbar.close()


def try_download(url: str, dest: Path, attempt: int) -> bool:
    """
    Attempt to download *url* to *dest*.
    Returns True on success (file exists and is large enough), False otherwise.
    Cleans up a partial/undersized file before returning False.
    """
    hook = TqdmDownloadHook(DATA_BAG_FILE)
    try:
        print(f"    Attempt {attempt}: {url}")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                block_size = 8192
                downloaded = 0
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    hook(downloaded // block_size, block_size, total)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"\n    Error: {exc}")
        return False
    finally:
        hook.close()

    # Verify minimum size
    if dest.exists():
        size = dest.stat().st_size
        size_gb = size / (1024 ** 3)
        if size < MIN_SIZE_BYTES:
            print(f"\n    File too small ({size_gb:.2f} GB < 1 GB minimum). Removing.")
            dest.unlink(missing_ok=True)
            return False
        print(f"\n    Downloaded {size_gb:.2f} GB.")
        return True

    return False


def download_with_fallback(dest: Path) -> bool:
    """
    Try each URL up to MAX_RETRIES times. Returns True if the file was
    successfully downloaded and verified.
    """
    for url_index, url in enumerate(DATA_BAG_URLS, start=1):
        print(f"\n  [URL {url_index}/{len(DATA_BAG_URLS)}] {url}")
        for attempt in range(1, MAX_RETRIES + 1):
            if try_download(url, dest, attempt):
                return True
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
        print(f"  All {MAX_RETRIES} attempts failed for this URL. Trying next...")

    return False


def main():
    print("=" * 60)
    print("  Cow and Lady Dataset Downloader (ETH ASL)")
    print("  https://www.research-collection.ethz.ch/handle/20.500.11850/721636")
    print("=" * 60)
    print(f"\n  Destination: {DEST_DIR}")
    print(f"  Approximate size: 4.6 GB (compressed rosbag)")
    print(f"  Sensors: depth camera (Kinect / Asus Xtion) + Vicon pose")
    print(f"  No IMU (voxblox uses external pose via Vicon)\n")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    out = DEST_DIR / DATA_BAG_FILE

    if out.exists():
        size_gb = out.stat().st_size / (1024 ** 3)
        if out.stat().st_size >= MIN_SIZE_BYTES:
            print(f"  Skipping {out.name} (already exists, {size_gb:.2f} GB)")
        else:
            print(f"  Found incomplete file ({size_gb:.2f} GB). Re-downloading...")
            out.unlink()
            if not download_with_fallback(out):
                print("\n  ERROR: Download failed from all URLs.")
                sys.exit(1)
    else:
        print(f"  Downloading {DATA_BAG_FILE}...")
        if not download_with_fallback(out):
            print("\n  ERROR: Download failed from all URLs.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  Done. Files at:")
    print(f"     {DEST_DIR}")
    print("=" * 60)

    # Topics (for reference):
    print("\n  ROS topics in the bag:")
    print("    /camera/depth_registered/points   sensor_msgs/PointCloud2")
    print("    /kinect/vrpn_client/estimated_transform  geometry_msgs/TransformStamped")
    print("    /tf, /tf_static")
    print("\n  Launch with voxblox:")
    print("    roslaunch voxblox_ros cow_and_lady_dataset.launch")


if __name__ == "__main__":
    main()
