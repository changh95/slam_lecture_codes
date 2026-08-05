#!/usr/bin/env python3
"""
Download the OpenLane rosbag conversion to ~/data/openlane/ for MonoLaneMapping.

The MonoLaneMapping authors publish the OpenLane *validation* split already
converted to ROS 1 bags, which is the only form the mapping demo needs:
  https://github.com/HKUST-Aerial-Robotics/MonoLaneMapping
  https://github.com/qiaozhijian/openlane_bag   (defines the LaneList messages)
See SLAM_zero_to_hero/monolane_mapping/README.md.

One 433 MB zip holds all 202 segments (630 MB unpacked). Each segment is a
20-second Waymo drive carrying three topics and no images at all:

    /gt_pose_wc      geometry_msgs/PoseStamped    vehicle pose
    /lanes_gt        openlane_bag/LaneList        ground-truth 3D lanes
    /lanes_predict   openlane_bag/LaneList        PersFormer's per-frame predictions

The monocular detector is not run here -- the predictions are baked into the
bags -- so the original OpenLane image and annotation download (which is gated
behind a registration form, and is ~500 GB) is *not* required. Only two things
want it: `stream_mapping.py --image_dir/--annotation_dir` for the front-camera
panel, and `run_mapping.py --all_segments` scoring, which reads the camera
extrinsics out of the per-frame json.

Alongside the bags the zip ships `lane3d_1000/test/*.txt`, OpenLane's own
scenario splits (curve, night, intersection...). They list image frames, but the
segment name is in the path, so `--scenario curve` turns them into a list of
bags to try. Categories overlap: most segments are in `intersection`.

About the URL: the upstream readme hands out the SharePoint share link
`/:u:/g/personal/.../<id>?download=1`, and that form now answers 403 to any
client without the share page's cookie. This script uses the equivalent
`_layouts/15/download.aspx?share=<id>` endpoint on the same file, which streams
the bytes with no cookie, reports Content-Length and honours Range requests, so
curl -C - / wget -c resume correctly. A Baidu mirror is in the upstream readme:
  https://pan.baidu.com/s/1Hrd8ashoiB4_f0B-iz6OHQ?pwd=2023

Transfers go through `curl` if present, else `wget`, else stdlib urllib, and the
zip is unpacked with stdlib zipfile, so there are no third-party dependencies.
Same resume/verify structure as download_gwanghwamun_coex.py.

Examples:
    python3 download_openlane.py                     # download + unpack, 1.1 GB peak
    python3 download_openlane.py --list
    python3 download_openlane.py --no-extract        # just fetch the zip
    python3 download_openlane.py --rm-archive        # unpack, then drop the zip
    python3 download_openlane.py --scenario curve    # which bags are curves
"""

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


# The share id of OpenLane.zip on the author's HKUST OneDrive. `download.aspx`
# rather than the readme's `?download=1`: see the note in the module docstring.
SHARE_HOST = "https://hkustconnect-my.sharepoint.com"
SHARE_USER = "zqiaoac_connect_ust_hk"
SHARE_ID = "EQxCBwl1Wc5Foq1wNOJ7ZKQBrNik0GK_qa7qEed_zrbGmQ"
ARCHIVE_URL = f"{SHARE_HOST}/personal/{SHARE_USER}/_layouts/15/download.aspx?share={SHARE_ID}"

BAIDU_MIRROR = "https://pan.baidu.com/s/1Hrd8ashoiB4_f0B-iz6OHQ?pwd=2023"

ARCHIVE = "OpenLane.zip"

# Exact Content-Length the endpoint reports, so a truncated transfer is detected
# rather than silently accepted, and the total of the zip's uncompressed sizes.
ARCHIVE_SIZE = 453897948
EXTRACTED_SIZE = 660107893

# What the zip unpacks to, relative to --dest. Everything is under one top-level
# OpenLane/ directory, which is what the demo mounts as /data/OpenLane.
ROOT_DIR = "OpenLane"
BAG_DIR = f"{ROOT_DIR}/lane3d_1000/rosbag"
SPLIT_DIR = f"{ROOT_DIR}/lane3d_1000/test"

N_BAGS = 202
N_FILES = 212

# OpenLane's scenario splits, as shipped in SPLIT_DIR. `bags` is how many of the
# listed segments actually have a rosbag -- curve names one segment that the
# conversion left out. A segment can be in several categories at once.
SCENARIOS = {
    "curve": dict(file="1000_curve.txt", bags=61,
                  note="Sweeping bends. The clearest showcase for the spline map."),
    "extreme_weather": dict(file="1000_extreme_weather.txt", bags=21,
                            note="Rain and glare, so the weakest PersFormer input."),
    "intersection": dict(file="1000_intersection.txt", bags=171,
                         note="Most of the split. Lanes stop and restart across the junction."),
    "merge_split_case": dict(file="1000_merge_split_case.txt", bags=30,
                             note="Lane count changes -- exercises the association logic."),
    "night": dict(file="1000_night.txt", bags=22, note="Low light."),
    "updown": dict(file="1000_updown.txt", bags=33,
                   note="Slopes, where the flat-ground assumption hurts most."),
}

DEST_DIR = Path.home() / "data" / "openlane"

# The segment baked into the image at examples/data/, which run_mapping.py and
# stream_mapping.py use when given no --bag. It is in the intersection split.
IMAGE_BAG = "segment-14486517341017504003_3406_349_3426_349_with_camera_labels"

# Local zip header magic; catches an HTML error page saved under a .zip name.
ZIP_MAGIC = b"PK\x03\x04"


def human(n: int) -> str:
    return f"{n / 1024**3:.2f} GB" if n >= 1024**3 else f"{n / 1024**2:.1f} MB"


def is_zip(path: Path) -> bool:
    """True if `path` starts with the zip local-header magic.

    Checked on a *pre-existing* complete file as well as on a fresh transfer: a
    file of exactly the right length is not necessarily the right file.
    """
    try:
        with open(path, "rb") as f:
            return f.read(len(ZIP_MAGIC)) == ZIP_MAGIC
    except OSError:
        return False


def fetch_curl(url: str, dest: Path) -> bool:
    """curl, resuming with -C -.

    `-f` is not optional: without it a SharePoint error response lands on disk as
    a short HTML or text page and curl still exits 0, i.e. you get a ".zip" that
    is really an error message. With -f, curl exits 22 and writes nothing.
    """
    cmd = ["curl", "-fL", "--retry", "5", "--retry-delay", "2",
           "--retry-connrefused", "--connect-timeout", "30",
           "-C", "-", "--progress-bar", "-o", str(dest), url]
    return subprocess.run(cmd).returncode == 0


def fetch_wget(url: str, dest: Path) -> bool:
    """wget, resuming with -c (this endpoint answers 206 for a ranged -O -c)."""
    cmd = ["wget", "-c", "--tries=5", "--timeout=60", "--waitretry=2",
           "--progress=dot:giga", "-O", str(dest), url]
    return subprocess.run(cmd).returncode == 0


def fetch_urllib(url: str, dest: Path, have: int, total: int) -> bool:
    """Fallback with no external tools: stdlib only, manual Range resume."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    done = have
    tty = sys.stdout.isatty()
    step = 1 if tty else 10
    shown = -1
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
            while chunk := r.read(1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if not total:
                    continue
                pct = int(100 * done / total)
                if pct // step > shown // step:
                    shown = pct
                    end, nl = ("\r", "") if tty else ("\n", "")
                    print(f"{nl}  > {dest.name}: {done / 1024**2:.0f}/{total / 1024**2:.0f} MB "
                          f"({pct:3d}%)", end=end, flush=True)
        if tty:
            print()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Transfer error: {exc}")
        return False


def pick_backend() -> str:
    for tool in ("curl", "wget"):
        if shutil.which(tool):
            return tool
    return "urllib"


def download(dest: Path, total: int, backend: str) -> bool:
    """Fetch OpenLane.zip to `dest`, resuming a partial file if one exists."""
    have = dest.stat().st_size if dest.exists() else 0

    if have == total:
        if not is_zip(dest):
            print(f"  {dest.name} is the right size but is not a zip; re-fetching")
            dest.unlink()
            have = 0
        else:
            print(f"  Skipping {dest.name} (complete, {human(have)})")
            return True
    if have > total:
        # Never resume onto an over-long file: that silently produces a corrupt
        # archive that still passes the magic-byte check.
        print(f"  {dest.name} is larger than expected ({have} > {total}); starting over")
        dest.unlink()
        have = 0
    if have:
        print(f"  Resuming {dest.name} at {human(have)} of {human(total)}")

    if backend == "curl":
        ok = fetch_curl(ARCHIVE_URL, dest)
    elif backend == "wget":
        ok = fetch_wget(ARCHIVE_URL, dest)
    else:
        ok = fetch_urllib(ARCHIVE_URL, dest, have, total)

    if not ok:
        if dest.exists() and dest.stat().st_size:
            print(f"  Incomplete; partial file kept at {dest} -- re-run to resume.")
        print(f"  If SharePoint is unreachable, the Baidu mirror is {BAIDU_MIRROR}")
        return False

    got = dest.stat().st_size if dest.exists() else 0
    if got != total:
        print(f"  Size mismatch for {dest.name}: {got} != {total}. Re-run to resume.")
        return False
    if not is_zip(dest):
        print(f"  {dest.name} is not a zip (a SharePoint error page?). Delete it and retry.")
        return False
    return True


def extract(archive: Path, dest: Path) -> bool:
    """Unpack the zip into `dest`, skipping if the bags are already there."""
    bags = dest / BAG_DIR
    if bags.is_dir() and len(list(bags.glob("*.bag"))) == N_BAGS:
        print(f"  Already extracted: {bags} ({N_BAGS} bags)")
        return True

    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            # The zip is trusted, but an unpacker that can write outside --dest
            # on a swapped file is not worth shipping.
            bad = [n for n in names if n.startswith(("/", "..")) or ".." in Path(n).parts]
            if bad:
                print(f"  Refusing to extract: {len(bad)} member(s) escape {dest}, "
                      f"first is {bad[0]}")
                return False
            print(f"  Extracting {archive.name} -> {dest} ({len(names)} members)")
            zf.extractall(dest)
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"  Extraction failed: {exc}")
        print(f"  Unpack by hand: unzip -q {archive} -d {dest}")
        return False
    return True


def verify(dest: Path) -> bool:
    """Count what actually landed on disk."""
    bags = sorted((dest / BAG_DIR).glob("*.bag")) if (dest / BAG_DIR).is_dir() else []
    splits = sorted((dest / SPLIT_DIR).glob("*.txt")) if (dest / SPLIT_DIR).is_dir() else []
    total = sum(b.stat().st_size for b in bags)
    print(f"  {len(bags)}/{N_BAGS} bags ({human(total)}), "
          f"{len(splits)}/{len(SCENARIOS)} scenario lists")
    if len(bags) != N_BAGS:
        print(f"  Expected {N_BAGS} bags in {dest / BAG_DIR}. Re-run to repair.")
        return False
    return True


def scenario_segments(dest: Path, name: str) -> list:
    """Bag paths for one OpenLane scenario split, from the shipped frame list.

    Each line is `validation/<segment>/<frame>.jpg`, so the segment name -- and
    therefore the bag name -- is the second path component. Commentary goes to
    stderr so that `--scenario curve > bags.txt` is a clean list of paths.
    """
    listing = dest / SPLIT_DIR / SCENARIOS[name]["file"]
    if not listing.exists():
        print(f"  {listing} is missing; download and extract first.", file=sys.stderr)
        return []
    segments = {line.split("/")[1] for line in listing.read_text().splitlines() if "/" in line}
    bag_dir = dest / BAG_DIR
    found = sorted(p for p in (bag_dir / f"{s}.bag" for s in segments) if p.exists())
    missing = len(segments) - len(found)
    if missing:
        print(f"  ({missing} of the {len(segments)} listed segments "
              f"{'has' if missing == 1 else 'have'} no rosbag)", file=sys.stderr)
    return found


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the OpenLane rosbag conversion (202 validation "
                    "segments) used by the MonoLaneMapping demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Contents: {ARCHIVE} ({human(ARCHIVE_SIZE)}) -> {human(EXTRACTED_SIZE)}, {N_FILES} files
  {BAG_DIR}/    {N_BAGS} bags, 20 s each
  {SPLIT_DIR}/      OpenLane's scenario splits

Scenario splits (overlapping):
""" + "\n".join(f"  {n:18s} {m['bags']:3d} bags  {m['note']}"
                for n, m in SCENARIOS.items())
        + f"""

Run it (see SLAM_zero_to_hero/monolane_mapping/README.md):
  podman run --rm -it -p 9090:9090 -p 9877:9877 \\
    -v {DEST_DIR / ROOT_DIR}:/data/OpenLane:ro -v "$(pwd)/results":/out \\
    slam_zero_to_hero:monolane_mapping \\
    python3 stream_mapping.py --output_dir /out/stream --rate 10 --odo_noise

The image already carries {IMAGE_BAG[:40]}...
at examples/data/, so that command works before any of this is downloaded. Pass
--bag /data/OpenLane/{BAG_DIR.split('/', 1)[1]}/<segment>.bag to pick another.
""",
    )
    parser.add_argument("--list", action="store_true",
                        help="Describe the archive and the scenario splits, then exit")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS),
                        help="List the extracted bags in one scenario split, then exit")
    parser.add_argument("--no-extract", action="store_true",
                        help="Download the zip but leave it packed")
    parser.add_argument("--rm-archive", action="store_true",
                        help=f"Delete {ARCHIVE} once extracted (saves {human(ARCHIVE_SIZE)})")
    parser.add_argument("--verify", action="store_true",
                        help="Only check what is already on disk, then exit")
    parser.add_argument("--dest", type=Path, default=DEST_DIR,
                        help=f"Destination directory (default: {DEST_DIR})")
    parser.add_argument("--backend", choices=("auto", "curl", "wget", "urllib"), default="auto",
                        help="Transfer tool (default: auto = curl, else wget, else stdlib urllib)")
    args = parser.parse_args()

    if args.list:
        print("OpenLane validation split, converted to ROS 1 bags for MonoLaneMapping")
        print(f"  {ARCHIVE}  {human(ARCHIVE_SIZE)} zipped, {human(EXTRACTED_SIZE)} unpacked, "
              f"{N_BAGS} segments")
        print(f"  Topics per bag: /gt_pose_wc, /lanes_gt, /lanes_predict -- no images.")
        print(f"  Source: {ARCHIVE_URL}")
        print(f"  Mirror: {BAIDU_MIRROR}\n")
        print("Scenario splits (a segment can be in several):")
        for name, meta in SCENARIOS.items():
            print(f"  {name:18s} {meta['bags']:3d} bags   {meta['note']}")
        print(f"\nUse --scenario <name> after extracting to get the bag paths.")
        return

    if args.scenario:
        bags = scenario_segments(args.dest, args.scenario)
        if not bags:
            sys.exit(1)
        for path in bags:
            print(path)
        print(f"\n  {len(bags)} bags in the {args.scenario} split", file=sys.stderr)
        return

    if args.verify:
        print(f"Checking {args.dest}")
        sys.exit(0 if verify(args.dest) else 1)

    backend = pick_backend() if args.backend == "auto" else args.backend
    if args.backend != "auto" and backend != "urllib" and not shutil.which(backend):
        print(f"  {backend} was requested but is not installed.")
        sys.exit(1)

    archive = args.dest / ARCHIVE
    have_archive = archive.exists() and archive.stat().st_size == ARCHIVE_SIZE
    extracting = not args.no_extract

    print("=" * 72)
    print("  OpenLane Rosbag Downloader (MonoLaneMapping)")
    print(f"  {N_BAGS} 20-second Waymo segments: PersFormer 3D lanes + GT + pose")
    print("=" * 72)
    print(f"\n  Destination: {args.dest}")
    print(f"  Bags unpack to: {args.dest / BAG_DIR}")
    print(f"  To download: {'nothing, the zip is already here' if have_archive else human(ARCHIVE_SIZE)}")
    if extracting:
        print(f"  Extracted:   {human(EXTRACTED_SIZE)}"
              + ("" if args.rm_archive else f"  (peak {human(ARCHIVE_SIZE + EXTRACTED_SIZE)} "
                                            "with the zip kept)"))

    free = shutil.disk_usage(args.dest if args.dest.exists() else Path.home()).free
    need = (0 if have_archive else ARCHIVE_SIZE) + (EXTRACTED_SIZE if extracting else 0)
    print(f"  Free space:  {human(free)}")
    if free < need:
        print(f"\n  WARNING: {human(need)} needed but only {human(free)} free.")
    print()

    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"--- {ARCHIVE} ({human(ARCHIVE_SIZE)}) ---")
    ok = download(archive, ARCHIVE_SIZE, backend)

    if ok and extracting:
        ok = extract(archive, args.dest)
        if ok:
            ok = verify(args.dest)
        if ok and args.rm_archive:
            archive.unlink()
            print(f"  Removed {archive.name}")

    print()
    print("=" * 72)
    if not ok:
        print(f"  Incomplete -- re-run to resume.")
        print("=" * 72)
        sys.exit(1)

    print(f"  Ready: {args.dest}")
    print("=" * 72)
    if args.no_extract:
        print(f"\n  Unpack with:\n    unzip -q {archive} -d {args.dest}")
    else:
        print(f"\n  Mount {args.dest / ROOT_DIR} at /data/OpenLane and run "
              "stream_mapping.py;")
        print("  see SLAM_zero_to_hero/monolane_mapping/README.md, or --help for the "
              "full command.")


if __name__ == "__main__":
    main()
