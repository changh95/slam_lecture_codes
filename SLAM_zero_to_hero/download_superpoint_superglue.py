#!/usr/bin/env python3
"""
Download SuperPoint + SuperGlue ONNX model weights for the part2_ch01_04 and
part2_ch01_10 demos.

Source: https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT (master/weights)
These are the FP32 ONNX exports already simplified for TensorRT (`_sim_int32`
suffix) -- the demos reference these exact filenames.

Drops the .onnx files into:
  part2_ch01_04/weights/
  part2_ch01_10/weights/  (only superpoint, since the demo only uses extraction)
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --user -q")
    from tqdm import tqdm


BASE_URL = "https://raw.githubusercontent.com/yuefanhao/SuperPoint-SuperGlue-TensorRT/master/weights"

REPO_ROOT = Path(__file__).resolve().parent

# (filename_in_upstream, list of (dest_dir, dest_filename))
WEIGHTS = [
    (
        "superpoint_v1_sim_int32.onnx",
        [
            (REPO_ROOT / "part2_ch01_04" / "weights", "superpoint_v1.onnx"),
            (REPO_ROOT / "part2_ch01_10" / "weights", "superpoint.onnx"),
        ],
    ),
    (
        "superglue_indoor_sim_int32.onnx",
        [
            (REPO_ROOT / "part2_ch01_04" / "weights", "superglue_indoor.onnx"),
        ],
    ),
    (
        "superglue_outdoor_sim_int32.onnx",
        [
            (REPO_ROOT / "part2_ch01_04" / "weights", "superglue_outdoor.onnx"),
        ],
    ),
]


class TqdmDownloadHook:
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
                leave=True,
            )
        if total_size > 0:
            self.pbar.update(min(block_size, total_size - self.pbar.n))
        else:
            self.pbar.update(block_size)

    def close(self):
        if self.pbar:
            self.pbar.close()


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  ⏭  {dest} already exists ({dest.stat().st_size / (1024 * 1024):.1f} MB), skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    hook = TqdmDownloadHook(dest.name)
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    finally:
        hook.close()


def main():
    print("=" * 60)
    print("  SuperPoint + SuperGlue ONNX Weights Downloader")
    print("  Source: github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT")
    print("=" * 60)
    print()

    # Cache one copy of each upstream file under /tmp, then copy/rename into
    # each demo's weights/ directory. Saves bandwidth when the same file
    # (e.g. superpoint) is shared by multiple demos.
    cache = Path("/tmp/superpoint_superglue_weights")
    cache.mkdir(parents=True, exist_ok=True)

    for upstream_name, dests in WEIGHTS:
        url = f"{BASE_URL}/{upstream_name}"
        cached = cache / upstream_name
        print(f"--- {upstream_name} ---")
        download(url, cached)
        # Copy/rename into each consumer demo
        for dest_dir, dest_name in dests:
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / dest_name
            if target.exists() and target.stat().st_size == cached.stat().st_size:
                print(f"  ⏭  {target} already up-to-date")
                continue
            target.write_bytes(cached.read_bytes())
            print(f"  ✓  {target} ({target.stat().st_size / (1024 * 1024):.1f} MB)")
        print()

    print("=" * 60)
    print("  Done. Weights installed at:")
    for _, dests in WEIGHTS:
        for dest_dir, dest_name in dests:
            print(f"     {dest_dir / dest_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
