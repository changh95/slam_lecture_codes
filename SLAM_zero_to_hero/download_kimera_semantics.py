#!/usr/bin/env python3
"""
Download the Kimera-Semantics demo data to ~/data/kimera_semantics/.

Kimera-Semantics builds a semantically-labelled 3D mesh from a depth image, a
2D semantic segmentation image and a pose (TF). Its README points at one demo
bag, `kimera_semantics_demo.bag` (a TESSE / uHumans Unity scene):

  https://github.com/MIT-SPARK/Kimera-Semantics#in-simulation-with-semantics

That Google Drive file is GONE: both ids ever published for it (the README's
1SG8cfJ6... and the older 1jpuE6tM... from ToniRV/Kimera-Semantics-1) answer
HTTP 404 as of 2026-09, and no mirror is known. This script still tries it
first, so it picks the file up again if MIT-SPARK ever restores it.

The fallback -- and the bag the lecture actually uses -- is uHumans2, the
successor dataset from the same lab and the same TESSE simulator, with exactly
the topics Kimera-Semantics consumes (and a `kimera_semantics_uHumans2.launch`
upstream):

  /tesse/depth_cam/mono/image_raw   depth (32FC1, metres)
  /tesse/seg_cam/rgb/image_raw      2D semantic segmentation (colour-coded)
  /tesse/left_cam/rgb/image_raw     RGB, + camera_info for all three
  /tesse/imu/*, /tesse/odom         IMU and ground-truth odometry
  /tf, /tf_static                   ground-truth poses (world -> left_cam)

  https://web.mit.edu/sparklab/datasets/uHumans2/

The default fallback is `uHumans2_office_s1_00h` (16.8 GB, a 55 x 53 m office
floor, no humans), the scene the lecture uses. `apartment_00h` (2.7 GB) is the quick
option; subway / neighborhood bags are 13-21 GB each.

Transfers use `curl` (resume + retry) if present, else `wget`, else urllib.

Usage:
    python3 download_kimera_semantics.py              # demo bag, else office_00h
    python3 download_kimera_semantics.py --list
    python3 download_kimera_semantics.py office_00h subway_00h
"""

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# name -> (file name on disk, google drive id, size as reported by Drive)
BAGS = {
    # Kimera-Semantics README demo bag. HTTP 404 since at least 2026-09.
    "demo":              ("kimera_semantics_demo.bag",          "1SG8cfJ6JEfY2PGXcxDPAMYzCcGBEh4Qq", "?"),
    # uHumans2 (web.mit.edu/sparklab/datasets/uHumans2). *_XXh = XX humans.
    "apartment_00h":     ("uHumans2_apartment_s1_00h.bag",     "1kU_drpyG7glQ8pJyeztpiy214WbBEtbM", "2.7G"),
    "apartment_01h":     ("uHumans2_apartment_s1_01h.bag",     "1jp7HrRsfGbmC-z757wwXDEpgNPHw0SbK", "2.9G"),
    "apartment_02h":     ("uHumans2_apartment_s1_02h.bag",     "1ai2p6QVNFaPJfEFOOec6URp5Anu0MBoo", "3.0G"),
    "office_00h":        ("uHumans2_office_s1_00h.bag",        "1CA_1Awu-bewJKpDrILzWok_H_6cOkGDb", "16.8G"),
    "office_06h":        ("uHumans2_office_s1_06h.bag",        "1zECekG47mlGafaJ84vCbwcx3Dz03NuvN", "17.3G"),
    "office_12h":        ("uHumans2_office_s1_12h.bag",        "1Of7s_QTE9nL1Hd69SFW1R5uDiJDiQrZr", "19.4G"),
    "subway_00h":        ("uHumans2_subway_s1_00h.bag",        "1ChL1SW1tfZrCjn5XEf4GJG8nm_Cb5AEm", "19.6G"),
    "subway_24h":        ("uHumans2_subway_s1_24h.bag",        "1ifatqW3hzL9yo8m7Jt3BCqIr-6kzIols", "21.3G"),
    "subway_36h":        ("uHumans2_subway_s1_36h.bag",        "1xFG565R-9LKXC60Rfx-7fruy3BP4TrHl", "20.4G"),
    "neighborhood_00h":  ("uHumans2_neighborhood_s1_00h.bag",  "1p_Uv4RLbl1GtjRxu2tldopFRKmgg_Vsy", "13.2G"),
    "neighborhood_24h":  ("uHumans2_neighborhood_s1_24h.bag",  "1LXloULyuohBzFLumBoScBMlFrT5nRPcE", "14.2G"),
    "neighborhood_36h":  ("uHumans2_neighborhood_s1_36h.bag",  "1AwgGpqe2g12T2Lm4Nilz4EaNm2_OtfHL", "14.3G"),
}

DEFAULT = "demo"
FALLBACK = "office_00h"  # the lecture default since 2026-09-28 (larger space than the apartment)
DEST_DIR = Path.home() / "data" / "kimera_semantics"
ROSBAG_MAGIC = b"#ROSBAG V2.0"


def remote_size(url: str) -> int:
    """Total size via a 1-byte Range probe. Raises HTTPError (404) if gone."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        crange = r.headers.get("Content-Range", "")
        return int(crange.split("/")[-1]) if "/" in crange else 0


def fetch_curl(url: str, dest: Path) -> bool:
    # -f: without it a Drive error page is saved as the ".bag" and curl exits 0.
    cmd = ["curl", "-fL", "--retry", "5", "--retry-delay", "2", "--retry-connrefused",
           "--connect-timeout", "30", "-C", "-", "--progress-bar", "-o", str(dest), url]
    return subprocess.run(cmd).returncode == 0


def fetch_wget(url: str, dest: Path) -> bool:
    cmd = ["wget", "-c", "--tries=5", "--timeout=60", "--waitretry=2",
           "--progress=dot:giga", "-O", str(dest), url]
    return subprocess.run(cmd).returncode == 0


def fetch_urllib(url: str, dest: Path, have: int, total: int) -> bool:
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    bar = tqdm(total=total or None, initial=have, unit="B", unit_scale=True,
               unit_divisor=1024, desc=f"  > {dest.name}", ncols=100) if tqdm else None
    try:
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(dest, "ab" if have else "wb") as f:
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
                if bar:
                    bar.update(len(chunk))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Transfer error: {exc}")
        return False
    finally:
        if bar:
            bar.close()


def pick_backend() -> str:
    for tool in ("curl", "wget"):
        if shutil.which(tool):
            return tool
    return "urllib"


def download(name: str, dest_dir: Path, backend: str) -> bool:
    fname, file_id, size = BAGS[name]
    dest = dest_dir / fname
    url = DOWNLOAD_URL.format(id=file_id)
    print(f"\n--- {name}: {fname} ({size}) ---")

    try:
        total = remote_size(url)
    except urllib.error.HTTPError as exc:
        print(f"  Not available: HTTP {exc.code} for drive id {file_id}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not query {fname}: {exc}")
        return False

    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"  Skipping {fname} (complete, {have / 1024**3:.2f} GB)")
        return True
    if have > total > 0:
        print(f"  {fname} is larger than expected ({have} > {total}); starting over")
        dest.unlink()
        have = 0
    if have:
        print(f"  Resuming at {have / 1024**3:.2f} of {total / 1024**3:.2f} GB")

    if backend == "curl":
        ok = fetch_curl(url, dest)
    elif backend == "wget":
        ok = fetch_wget(url, dest)
    else:
        ok = fetch_urllib(url, dest, have, total)
    if not ok:
        if dest.exists() and dest.stat().st_size:
            print(f"  Incomplete; partial file kept at {dest} -- re-run to resume.")
        return False

    got = dest.stat().st_size if dest.exists() else 0
    if total and got != total:
        print(f"  Size mismatch for {fname}: {got} != {total}. Re-run to resume.")
        return False
    with open(dest, "rb") as f:
        if f.read(len(ROSBAG_MAGIC)) != ROSBAG_MAGIC:
            print(f"  {fname} is not a ROS 1 bag (an HTML error page?). Delete it and retry.")
            return False
    print(f"  OK: {dest} ({got / 1024**3:.2f} GB)")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="*", default=[DEFAULT],
                    help=f"bag name(s) from --list (default: {DEFAULT}, "
                         f"falling back to {FALLBACK})")
    ap.add_argument("--list", action="store_true", help="list bags and exit")
    ap.add_argument("--no-fallback", action="store_true",
                    help=f"do not fetch {FALLBACK} when the demo bag is unavailable")
    ap.add_argument("--dest", type=Path, default=DEST_DIR,
                    help=f"destination directory (default: {DEST_DIR})")
    ap.add_argument("--backend", choices=("auto", "curl", "wget", "urllib"),
                    default="auto")
    args = ap.parse_args()

    if args.list:
        print("Kimera-Semantics demo bag + uHumans2 (MIT SPARK, TESSE simulator)\n")
        for name, (fname, _, size) in BAGS.items():
            tag = ""
            if name == DEFAULT:
                tag = "  <- default (README demo bag; HTTP 404 since 2026-09)"
            elif name == FALLBACK:
                tag = "  <- fallback, used by SLAM_zero_to_hero/kimera"
            print(f"  {name:18s} {size:>6s}  {fname}{tag}")
        return

    unknown = [b for b in args.bags if b not in BAGS]
    if unknown:
        sys.exit(f"unknown bag(s): {', '.join(unknown)}\nuse --list to see them")

    backend = pick_backend() if args.backend == "auto" else args.backend
    print("=" * 68)
    print("  Kimera-Semantics Dataset Downloader")
    print("=" * 68)
    print(f"  Destination: {args.dest}")
    print(f"  Bags:        {', '.join(args.bags)}")
    print(f"  Transfer:    {backend}")
    args.dest.mkdir(parents=True, exist_ok=True)

    wanted = list(args.bags)
    ok, failed = [], []
    while wanted:
        name = wanted.pop(0)
        if download(name, args.dest, backend):
            ok.append(name)
            continue
        failed.append(name)
        if name == "demo" and not args.no_fallback and FALLBACK not in args.bags + ok:
            print(f"  The README demo bag is unavailable; fetching {FALLBACK} "
                  "(same TESSE simulator and topics) instead.")
            wanted.insert(0, FALLBACK)

    print("\n" + "=" * 68)
    print(f"  Downloaded: {', '.join(ok) or 'nothing'}")
    if failed:
        print(f"  Failed:     {', '.join(failed)}")
    print("=" * 68)
    print("\n  Run with Kimera-Semantics: see SLAM_zero_to_hero/kimera/README.md")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
