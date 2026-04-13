#!/usr/bin/env python3
"""
Download the cow_and_lady_dataset rosbag from ETH ASL (used with voxblox).
Destination: ~/data/cow_and_lady/
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


# ETH Research Collection permanent URL
# Landing page: https://www.research-collection.ethz.ch/handle/20.500.11850/721636
DATA_BAG_URL = (
    "https://www.research-collection.ethz.ch/bitstreams/"
    "bfb68f88-fcb2-4e09-aa53-434d9162cef5/download"
)
DATA_BAG_FILE = "data.bag"

DEST_DIR = Path.home() / "data" / "cow_and_lady"


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
        size_gb = out.stat().st_size / (1024**3)
        print(f"  Skipping {out.name} (already exists, {size_gb:.2f} GB)")
    else:
        print(f"Downloading {DATA_BAG_FILE}...")
        hook = TqdmDownloadHook(DATA_BAG_FILE)
        try:
            urllib.request.urlretrieve(DATA_BAG_URL, str(out), reporthook=hook)
        finally:
            hook.close()

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
