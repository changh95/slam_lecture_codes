#!/usr/bin/env python3
"""
Download the DSP-SLAM example sequences and network weights to ~/data/dsp_slam/
with tqdm progress bars, then unzip the sequence archives.

Reference: https://github.com/JingwenWang95/DSP-SLAM
Paper:     https://arxiv.org/abs/2108.09481
           "DSP-SLAM: Object Oriented SLAM with Deep Shape Priors",
            Wang, Ruenz, Agapito, 3DV 2021

The authors host everything in a public UCL SharePoint folder. Opening the
anonymous share link hands out a FedAuth cookie, and with that cookie the
SharePoint REST API serves each file. No login is needed.

KITTI 07 is the stereo+LiDAR sequence the dsp_slam demo runs. Its zip carries
the stereo images, the Velodyne scans, and the pre-computed MaskRCNN (2D) and
PointPillars (3D) labels, so DSP-SLAM can run with `detect_online: false` and
without mmdetection. The DeepSDF `cars_64` decoder is the shape prior.

Layout after download (what the demo mounts at /data):
    ~/data/dsp_slam/kitti/07/{image_0,image_1,velodyne,labels,calib.txt,times.txt}
    ~/data/dsp_slam/weights/deepsdf/cars_64/{specs.json,ModelParameters,LatentCodes}
    ~/data/dsp_slam/weights/{maskrcnn,pointpillars}/model.pth

Examples:
    python3 download_dsp_slam.py                       # kitti_07 + cars_64 (~3.9 GB)
    python3 download_dsp_slam.py --list
    python3 download_dsp_slam.py --items kitti_07 deepsdf_cars_64 maskrcnn pointpillars
    python3 download_dsp_slam.py --items freiburg_car001 deepsdf_cars_64
    python3 download_dsp_slam.py --keep-zip
"""

import argparse
import http.cookiejar
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system(f"{sys.executable} -m pip install tqdm --break-system-packages -q")
    from tqdm import tqdm


SHARE_URL = ("https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabjw4_ucl_ac_uk/"
             "Eh3nHv6D-LZHkuny4iNOexQBGdDVxloM_nwbEZdxeRfStw?e=sYO1Ot")
SITE = "https://liveuclac-my.sharepoint.com/personal/ucabjw4_ucl_ac_uk"
ROOT = "/personal/ucabjw4_ucl_ac_uk/Documents/dsp-slam"

DEST_DIR = Path.home() / "data" / "dsp_slam"

# item -> (remote folder under ROOT, local folder under dest, files, is_zip, note)
# For DeepSDF only specs.json and the `latest` checkpoints are fetched: that is
# all deep_sdf/workspace.py loads. The optimizer states are skipped.
DEEPSDF_FILES = ["specs.json", "ModelParameters/latest.pth", "LatentCodes/latest.pth"]
ITEMS = {
    "kitti_07":        ("data/kitti", "kitti", ["07.zip"], True,
                        "KITTI odometry 07: stereo + Velodyne + 2D/3D labels (3.6 GiB)"),
    "deepsdf_cars_64": ("weights/deepsdf/cars_64", "weights/deepsdf/cars_64", DEEPSDF_FILES, False,
                        "DeepSDF car decoder, 64-d code (used for KITTI / Freiburg)"),
    "deepsdf_cars_32": ("weights/deepsdf/cars_32", "weights/deepsdf/cars_32", DEEPSDF_FILES, False,
                        "DeepSDF car decoder, 32-d code"),
    "deepsdf_chairs_64": ("weights/deepsdf/chairs_64", "weights/deepsdf/chairs_64", DEEPSDF_FILES, False,
                          "DeepSDF chair decoder (Redwood chairs)"),
    "maskrcnn":        ("weights/maskrcnn", "weights/maskrcnn", ["model.pth"], False,
                        "MaskRCNN weights, only for detect_online=true (242 MiB)"),
    "pointpillars":    ("weights/pointpillars", "weights/pointpillars", ["model.pth"], False,
                        "PointPillars weights, only for detect_online=true (18 MiB)"),
    "second":          ("weights/second", "weights/second", ["model.pth"], False,
                        "SECOND weights (alternative 3D detector)"),
    "freiburg_car001": ("data/freiburg_cars", "freiburg", ["Car001.zip"], True, "Freiburg Cars 001, mono (1.3 GiB)"),
    "freiburg_car002": ("data/freiburg_cars", "freiburg", ["Car002.zip"], True, "Freiburg Cars 002, mono (1.9 GiB)"),
    "freiburg_car010": ("data/freiburg_cars", "freiburg", ["Car010.zip"], True, "Freiburg Cars 010, mono (1.4 GiB)"),
    "redwood_01053":   ("data/redwood_chairs", "redwood", ["01053.zip"], True, "Redwood chair 01053, mono (1.6 GiB)"),
    "redwood_02484":   ("data/redwood_chairs", "redwood", ["02484.zip"], True, "Redwood chair 02484, mono (1.9 GiB)"),
    "redwood_09374":   ("data/redwood_chairs", "redwood", ["09374.zip"], True, "Redwood chair 09374, mono (0.9 GiB)"),
    "redwood_09647":   ("data/redwood_chairs", "redwood", ["09647.zip"], True, "Redwood chair 09647, mono (1.1 GiB)"),
}
DEFAULT_ITEMS = ["kitti_07", "deepsdf_cars_64"]


def make_opener():
    """Open the anonymous share link once; the FedAuth cookie it sets unlocks the REST API."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    opener.open(SHARE_URL).read()
    if not any(c.name == "FedAuth" for c in jar):
        raise RuntimeError("SharePoint did not hand out a FedAuth cookie; the share link may be dead")
    return opener


def file_url(remote_path: str) -> str:
    quoted = urllib.parse.quote(remote_path.replace("'", "''"))
    return f"{SITE}/_api/web/GetFileByServerRelativeUrl('{quoted}')/$value"


def remote_size(opener, remote_path: str) -> int:
    quoted = urllib.parse.quote(remote_path.replace("'", "''"))
    req = urllib.request.Request(
        f"{SITE}/_api/web/GetFileByServerRelativeUrl('{quoted}')?$select=Length",
        headers={"Accept": "application/json;odata=nometadata"})
    with opener.open(req) as r:
        return int(json.load(r)["Length"])


def download_file(opener, remote_path: str, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = remote_size(opener, remote_path)
    except Exception as exc:
        print(f"  !! cannot stat {remote_path}: {exc}")
        return False
    if out.exists() and out.stat().st_size == size:
        print(f"  = {out.name} already complete ({size / 2**20:.1f} MiB), skipping")
        return True

    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with opener.open(file_url(remote_path)) as r, open(tmp, "wb") as f, \
                tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024,
                     desc=f"  ↓ {out.name}", ncols=100, mininterval=0.5) as pbar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    except Exception as exc:
        print(f"  !! failed {out.name}: {exc}")
        return False
    if tmp.stat().st_size != size:
        print(f"  !! {out.name}: got {tmp.stat().st_size} bytes, expected {size}")
        return False
    tmp.rename(out)
    return True


def unzip_file(zip_path: Path, dest: Path):
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def main():
    ap = argparse.ArgumentParser(
        description="Download DSP-SLAM example sequences and weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--items", nargs="+", default=DEFAULT_ITEMS,
                    help="items to fetch (default: kitti_07 deepsdf_cars_64)")
    ap.add_argument("--list", action="store_true", help="list the available items and exit")
    ap.add_argument("--keep-zip", action="store_true", help="keep sequence zips after extracting")
    ap.add_argument("--dest", type=Path, default=DEST_DIR)
    args = ap.parse_args()

    if args.list:
        print(f"{'item':<20} note")
        for name, (*_, note) in ITEMS.items():
            print(f"{name:<20} {note}{'  [default]' if name in DEFAULT_ITEMS else ''}")
        return 0

    unknown = [i for i in args.items if i not in ITEMS]
    if unknown:
        print(f"Unknown item(s): {', '.join(unknown)}. Run with --list.")
        return 1

    opener = make_opener()
    failed = []
    for name in args.items:
        remote_dir, local_dir, files, is_zip, note = ITEMS[name]
        print(f"=== {name}: {note} ===")
        for fn in files:
            out = args.dest / local_dir / fn
            if is_zip and (out.parent / Path(fn).stem).is_dir() and not out.exists():
                print(f"  = {Path(fn).stem}/ already extracted, skipping")
                continue
            if not download_file(opener, f"{ROOT}/{remote_dir}/{fn}", out):
                failed.append(f"{name}/{fn}")
                continue
            if is_zip:
                unzip_file(out, out.parent)
                if not args.keep_zip:
                    out.unlink()

    print(f"\nData in {args.dest}")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
