#!/usr/bin/env python3
"""
Download the UZH-FPV Drone Racing Dataset to ~/data/uzh_fpv/
with tqdm progress bars, then unzip the calibration archives.

Reference: https://fpv.ifi.uzh.ch/
Paper:     https://rpg.ifi.uzh.ch/docs/ICRA19_Delmerico.pdf
           "Are We Ready for Autonomous Drone Racing? The UZH-FPV Drone
            Racing Dataset", Delmerico et al., ICRA 2019

The drone carries two camera systems. `snapdragon` is a 640x480 stereo fisheye
pair plus a 500 Hz IMU; `davis` is a 346x260 event camera that also emits
frames. SVO Pro consumes ordinary frames, so snapdragon (higher resolution and
stereo) is the default here.

Only sequences whose name ends in `_with_gt` carry ground truth. Default is
indoor_forward_3, the sequence the svo_pro_open demo is verified against.

Examples:
    python3 download_uzh_fpv.py                      # indoor_forward_3 + calib
    python3 download_uzh_fpv.py --list
    python3 download_uzh_fpv.py --sequences indoor_forward_3 indoor_forward_5
    python3 download_uzh_fpv.py --sensor davis --sequences indoor_45_2
    python3 download_uzh_fpv.py --calib-only
"""

import argparse
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


BASE_URL = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3"
CALIB_URL = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib"

# Sequence name -> has public ground truth. Taken from the download table at
# https://fpv.ifi.uzh.ch/datasets/ . Sequences without ground truth still run,
# they just cannot be scored.
SEQUENCES = {
    "indoor_forward_3": True,
    "indoor_forward_5": True,
    "indoor_forward_6": True,
    "indoor_forward_7": True,
    "indoor_forward_8": False,
    "indoor_forward_9": True,
    "indoor_forward_10": True,
    "indoor_forward_11": False,
    "indoor_forward_12": False,
    "indoor_45_1": False,
    "indoor_45_2": True,
    "indoor_45_3": False,
    "indoor_45_4": True,
    "indoor_45_9": True,
    "indoor_45_11": False,
    "indoor_45_12": True,
    "indoor_45_13": True,
    "indoor_45_14": True,
    "indoor_45_16": False,
    "outdoor_forward_1": True,
    "outdoor_forward_2": False,
    "outdoor_forward_3": True,
    "outdoor_forward_5": True,
    "outdoor_forward_6": False,
    "outdoor_forward_9": False,
    "outdoor_forward_10": False,
    "outdoor_45_1": True,
    "outdoor_45_2": False,
}

# One calibration per (environment, sensor) pair, shared by every sequence in it.
CALIB_GROUPS = ["indoor_forward", "indoor_45", "outdoor_forward", "outdoor_45"]

DEST_DIR = Path.home() / "data" / "uzh_fpv"

DEFAULT_SEQUENCES = ["indoor_forward_3"]


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
        downloaded = block_num * block_size
        if total_size > 0:
            self.pbar.update(min(downloaded, total_size) - self.pbar.n)
        else:
            self.pbar.update(block_size)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()


def download_file(url: str, dest: Path, filename: str) -> Path:
    """Download url to dest/filename, skipping files that are already complete."""
    out = dest / filename
    dest.mkdir(parents=True, exist_ok=True)

    # The server reports Content-Length, so a size match means a finished file.
    try:
        with urllib.request.urlopen(url) as resp:
            remote_size = int(resp.headers.get("Content-Length", 0))
    except Exception as exc:
        print(f"  !! cannot reach {url}: {exc}")
        return None

    if out.exists() and remote_size and out.stat().st_size == remote_size:
        size = (f"{remote_size / 2**30:.2f} GiB" if remote_size >= 2**30
                else f"{remote_size / 2**20:.1f} MiB")
        print(f"  = {filename} already complete ({size}), skipping")
        return out

    hook = TqdmDownloadHook(filename)
    try:
        urllib.request.urlretrieve(url, out, reporthook=hook)
    except Exception as exc:
        hook.close()
        print(f"  !! failed {filename}: {exc}")
        return None
    hook.close()
    return out


def unzip_file(zip_path: Path, dest: Path):
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def calib_group_of(sequence: str) -> str:
    for g in CALIB_GROUPS:
        if sequence.startswith(g):
            return g
    raise ValueError(f"cannot infer calibration group for {sequence}")


def main():
    ap = argparse.ArgumentParser(
        description="Download the UZH-FPV Drone Racing Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES,
                    help="sequence names (default: indoor_forward_3)")
    ap.add_argument("--sensor", choices=["snapdragon", "davis"], default="snapdragon",
                    help="sensor set to download (default: snapdragon)")
    ap.add_argument("--all-with-gt", action="store_true",
                    help="download every sequence that has ground truth (~40 GB)")
    ap.add_argument("--calib-only", action="store_true",
                    help="download only the calibration archives")
    ap.add_argument("--list", action="store_true",
                    help="list the available sequences and exit")
    ap.add_argument("--dest", type=Path, default=DEST_DIR)
    args = ap.parse_args()

    if args.list:
        print(f"{'sequence':<24} ground truth")
        for name, gt in SEQUENCES.items():
            print(f"{name:<24} {'yes' if gt else 'no'}")
        return 0

    if args.all_with_gt:
        args.sequences = [s for s, gt in SEQUENCES.items() if gt]

    unknown = [s for s in args.sequences if s not in SEQUENCES]
    if unknown:
        print(f"Unknown sequence(s): {', '.join(unknown)}")
        print("Run with --list to see the valid names.")
        return 1

    dest = args.dest
    calib_dir = dest / "calib"

    # Calibration first: it is small, and a sequence is useless without it.
    groups = sorted({calib_group_of(s) for s in args.sequences})
    print(f"=== calibration ({args.sensor}) ===")
    for g in groups:
        name = f"{g}_calib_{args.sensor}.zip"
        path = download_file(f"{CALIB_URL}/{name}", calib_dir, name)
        if path:
            unzip_file(path, calib_dir)

    if args.calib_only:
        print(f"\nCalibration in {calib_dir}")
        return 0

    print(f"\n=== sequences ({args.sensor}) ===")
    failed = []
    for seq in args.sequences:
        suffix = "_with_gt" if SEQUENCES[seq] else ""
        name = f"{seq}_{args.sensor}{suffix}.bag"
        if not SEQUENCES[seq]:
            print(f"  (note: {seq} has no public ground truth)")
        if download_file(f"{BASE_URL}/{name}", dest, name) is None:
            failed.append(name)

    print(f"\nDone. Data in {dest}")
    if failed:
        print("Failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
