#!/usr/bin/env python3
"""
Download the U-AMC handheld-scanner dataset to ~/data/gwanghwamun_coex/.

ROS 2 bags recorded with the UAMHD handheld rig (Livox Avia + Livox Mid-360 +
Luxonis Oak-D Pro on a Jetson Orin, PTP-synced), for FAST-LIVO2-ROS2:
  https://github.com/U-AMC/UAMHD-Mapping
  https://github.com/U-AMC/FAST-LIVO2-ROS2
See SLAM_zero_to_hero/uamc/README.md.

Two locations in Seoul, named in the bags by abbreviation:
  * `ghm`  -- Gwanghwamun. One 14-minute sequence, the longest by far.
  * `coex` -- the COEX centre in Gangnam. Three shorter sequences.

Each archive is a zstd-compressed tar of one rosbag2 directory
(`metadata.yaml` + a single sqlite3 `.db3`). Both LiDARs record at once, and
they arrive as *different message types*:

    /livox/lidar_3JEDM180010C211   livox_interfaces/msg/CustomMsg   ~10 Hz   (Avia, by serial)
    /livox/imu_3JEDM180010C211     sensor_msgs/msg/Imu             ~125 Hz
    /livox/lidar_192_168_1_150     sensor_msgs/msg/PointCloud2      ~10 Hz   (Mid-360, by IP)
    /livox/imu_192_168_1_150       sensor_msgs/msg/Imu             ~200 Hz
    /oak/rgb/image_raw             sensor_msgs/msg/Image           ~21 Hz
    /oak/rgb/camera_info           sensor_msgs/msg/CameraInfo      ~21 Hz

Three things to know before picking a sequence:

  * `lvi_ghm_set` is the flagship: 841.8 s, 326,594 messages, the full sensor
    set. No public download link is known for it -- ask U-AMC. The script still
    handles it if the archive is already in the destination directory, so
    `--extract lvi_ghm_set` works offline.

  * `multi_lidar_coex_set` carries **zero** camera messages. Both /oak topics
    were advertised and recorded empty, so this bag can only drive the
    LiDAR-inertial pipelines (`mapping_*.launch.py`), never the LVI ones.

  * `lvi_coex_set_2` and `lvi_set_2_restamped` are the **same recording**: same
    334.7 s duration, same 129,083 messages, same start time (2026-05-13
    14:50 UTC), and `.db3` files of byte-identical length. The `_restamped` tar
    was built three days later and is the one to use; the other is kept here
    only so the pair can be compared. Do not download both expecting two
    sequences -- that is 21.6 GB of duplicate data.

The Avia topic's type is `livox_interfaces/msg/CustomMsg`, but FAST-LIVO2-ROS2
builds against `livox_ros_driver2`, whose CustomMsg has a different type hash.
Playing the bag as-is makes the node see no LiDAR at all. Patch the extracted
`metadata.yaml` -- this is upstream README §4.2, with a different source type
than the one documented there:

    -      type: livox_interfaces/msg/CustomMsg
    +      type: livox_ros_driver2/msg/CustomMsg

`--patch-type` does that edit for you, keeping a .orig backup.

Archives live in the destination directory and bags unpack into its
`extracted/` subdirectory. The three downloadable archives total 23.0 GB and
unpack to 46.6 GB; `lvi_ghm_set` is another 25.1 GB packed and 54.3 GB
unpacked.

Transfers go through `curl` if present, else `wget`, else stdlib urllib, so
there are no third-party Python dependencies (no gdown needed) and resume works.
Same Drive-download endpoint and resume/verify logic as download_cerberus2.py.
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# Google Drive's large-file endpoint. The plain `uc?id=` URL serves an HTML
# confirmation page for anything over ~100 MB; this one streams bytes and honours
# Range requests, so curl -C - / wget -c resume correctly.
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# name -> dict(file, id, size, extracted, dir, venue, duration, messages, camera, note)
# `size` is the exact Content-Length Drive reports, so a truncated transfer is
# detected rather than silently accepted. `extracted` is the .db3 size from the
# tar header, for the disk-space estimate. `id=None` means no known link.
SEQUENCES = {
    "lvi_ghm_set": dict(
        file="lvi_ghm_set.zst",
        id=None,
        size=26981137132,
        extracted=58315124736,
        dir="lvi_ghm_set",
        venue="Gwanghwamun",
        duration=841.8, messages=326594, camera=17976,
        note="The flagship sequence: 14 min, full sensor set, recorded "
             "2026-03-27. No public link known -- ask U-AMC. --extract works "
             "if the archive is already in --dest.",
    ),
    "lvi_set_2_restamped": dict(
        file="lvi_set_2_restamped.zst",
        id="1J9iQjG3BI7FyhRUsqE950Cc3GPKLadQm",
        size=11142572881,
        extracted=23207309312,
        dir="lvi_set_2_restamped",
        venue="COEX",
        duration=334.7, messages=129083, camera=7157,
        note="LiDAR-visual-inertial, both LiDARs + Oak-D RGB at 21.4 Hz. "
             "Timestamps rewritten; the COEX sequence to use.",
    ),
    "lvi_coex_set_2": dict(
        file="lvi_coex_set_2.zst",
        id="1lGg2DXJN4rO7NhqWrSlrZuAfn0pj_-py",
        size=11142655427,
        extracted=23207309312,
        dir="lvi_coex_set_2",
        venue="COEX",
        duration=334.7, messages=129083, camera=7157,
        note="The SAME recording as lvi_set_2_restamped, before restamping. "
             "Only for comparing the two; not a second sequence.",
    ),
    "multi_lidar_coex_set": dict(
        file="multi_lidar_coex_set.zst",
        id="1Zl6l_iufDU2jF2aZ_NbyHMzKu9tn6N__",
        size=2389320152,
        extracted=3694292992,
        dir="multi_lidar_coex_set",
        venue="COEX",
        duration=364.8, messages=125167, camera=0,
        note="Two LiDARs + two IMUs, NO camera data (both /oak topics are "
             "empty). LiDAR-inertial pipelines only.",
    ),
}

DEFAULT_SEQUENCES = ["lvi_set_2_restamped"]

DEST_DIR = Path.home() / "data" / "gwanghwamun_coex"

# Bags unpack here, beside the archives rather than among them.
EXTRACT_SUBDIR = "extracted"

# zstd frame magic; catches an HTML error page saved under a .zst name.
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

WRONG_TYPE = "livox_interfaces/msg/CustomMsg"
RIGHT_TYPE = "livox_ros_driver2/msg/CustomMsg"


def human(n: int) -> str:
    return f"{n / 1024**3:.2f} GB" if n >= 1024**3 else f"{n / 1024**2:.1f} MB"


def is_zstd(path: Path) -> bool:
    """True if `path` starts with the zstd frame magic.

    Checked on a *pre-existing* complete file as well as on a fresh transfer:
    a file of exactly the right length is not necessarily the right file.
    """
    try:
        with open(path, "rb") as f:
            return f.read(len(ZSTD_MAGIC)) == ZSTD_MAGIC
    except OSError:
        return False


def fetch_curl(url: str, dest: Path) -> bool:
    """curl, resuming with -C -.

    `-f` is not optional: without it a Drive error response lands on disk as a
    ~1.6 kB HTML page and curl still exits 0, i.e. you get a ".zst" that is
    really an error message. With -f, curl exits 22 and writes nothing.
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
                    print(f"{nl}  > {dest.name}: {done / 1024**3:.2f}/{total / 1024**3:.2f} GB "
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


def download(file_id, dest: Path, total: int, backend: str) -> bool:
    """Fetch a Drive file to `dest`, resuming a partial file if one exists.

    `file_id` of None means no link is known: succeed if the archive is already
    there, fail with an explanation if it is not.
    """
    have = dest.stat().st_size if dest.exists() else 0

    if file_id is None:
        if have == total and is_zstd(dest):
            print(f"  Using the local copy of {dest.name} ({human(have)})")
            return True
        if have:
            print(f"  {dest.name} is present but is {human(have)}, not the expected "
                  f"{human(total)}; no download link is known to repair it.")
        else:
            print(f"  No download link is known for {dest.name}. Ask U-AMC for it, "
                  f"then put the archive in {dest.parent} and re-run with --extract.")
        return False

    url = DOWNLOAD_URL.format(id=file_id)

    if have == total:
        if not is_zstd(dest):
            print(f"  {dest.name} is the right size but is not a zstd archive; re-fetching")
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
    if got != total:
        print(f"  Size mismatch for {dest.name}: {got} != {total}. Re-run to resume.")
        return False
    if not is_zstd(dest):
        print(f"  {dest.name} is not a zstd archive (an HTML error page?). "
              "Delete it and retry.")
        return False
    return True


def extract(archive: Path, bags: Path, bag_dir: str) -> bool:
    """tar --zstd the archive into `bags`, skipping if the bag is already there."""
    out = bags / bag_dir
    if (out / "metadata.yaml").exists():
        print(f"  Already extracted: {out}")
        return True
    if not shutil.which("zstd"):
        print("  zstd is not installed; cannot extract. `sudo apt install zstd`, "
              f"or unpack by hand:\n    tar --zstd -xf {archive} -C {bags}")
        return False
    bags.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive.name} -> {out}")
    cmd = ["tar", "--zstd", "-xf", str(archive), "-C", str(bags)]
    if subprocess.run(cmd).returncode != 0:
        print(f"  Extraction failed. Unpack by hand: tar --zstd -xf {archive} -C {bags}")
        return False
    return (out / "metadata.yaml").exists()


def patch_type(bags: Path, bag_dir: str) -> None:
    """Rewrite the Avia topic's message type to the livox_ros_driver2 one.

    FAST-LIVO2-ROS2 builds against livox_ros_driver2, so a bag advertising
    livox_interfaces/msg/CustomMsg deserialises to nothing on that topic.
    """
    meta = bags / bag_dir / "metadata.yaml"
    if not meta.exists():
        print(f"  No metadata.yaml at {meta}; extract first.")
        return
    text = meta.read_text()
    if RIGHT_TYPE in text:
        print(f"  {meta.name} already uses {RIGHT_TYPE}")
        return
    if WRONG_TYPE not in text:
        print(f"  {meta.name} mentions neither type; leaving it alone.")
        return
    backup = meta.with_suffix(".yaml.orig")
    if not backup.exists():
        backup.write_text(text)
    meta.write_text(text.replace(WRONG_TYPE, RIGHT_TYPE))
    print(f"  Patched {meta} ({WRONG_TYPE} -> {RIGHT_TYPE}); original at {backup.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the U-AMC handheld-scanner dataset "
                    "(2x Livox LiDAR + 2x IMU + Oak-D RGB, Gwanghwamun and COEX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sequences:\n"
        + "\n".join(
            f"  {n:22s} {human(m['size']):>9s} -> {human(m['extracted']):>9s}  "
            f"{m['duration']:6.1f} s  {m['venue']:<12s} camera: {m['camera'] or 'NONE'}"
            + ("  [no link]" if m["id"] is None else "")
            for n, m in SEQUENCES.items())
        + f"\n\nDefault: {', '.join(DEFAULT_SEQUENCES)}"
        + "\n\nRun with FAST-LIVO2-ROS2 (see SLAM_zero_to_hero/uamc/README.md):\n"
        "  ros2 launch fast_livo mapping_aviz_lvi.launch.py use_rviz:=True\n"
        f"  ros2 bag play -p {DEST_DIR / EXTRACT_SUBDIR / 'lvi_ghm_set'}\n",
    )
    parser.add_argument("sequences", nargs="*", default=DEFAULT_SEQUENCES,
                        help=f"Sequence name(s) (default: {', '.join(DEFAULT_SEQUENCES)})")
    parser.add_argument("--all", action="store_true",
                        help="Every sequence with a link (23.0 GB; includes the duplicate pair)")
    parser.add_argument("--list", action="store_true", help="List sequences and exit")
    parser.add_argument("--extract", action="store_true",
                        help=f"tar --zstd -x each archive into <dest>/{EXTRACT_SUBDIR}/")
    parser.add_argument("--patch-type", action="store_true",
                        help=f"With --extract, rewrite {WRONG_TYPE} to {RIGHT_TYPE} in metadata.yaml")
    parser.add_argument("--rm-archive", action="store_true",
                        help="Delete each .zst once extracted (halves the peak disk need)")
    parser.add_argument("--dest", type=Path, default=DEST_DIR,
                        help=f"Destination directory (default: {DEST_DIR})")
    parser.add_argument("--backend", choices=("auto", "curl", "wget", "urllib"), default="auto",
                        help="Transfer tool (default: auto = curl, else wget, else stdlib urllib)")
    args = parser.parse_args()

    if args.list:
        print("U-AMC handheld-scanner dataset (UAMHD rig -> FAST-LIVO2-ROS2)")
        print("Venues: ghm = Gwanghwamun, coex = the COEX centre, both in Seoul\n")
        for name, meta in SEQUENCES.items():
            mark = " <- default" if name in DEFAULT_SEQUENCES else ""
            link = "" if meta["id"] else "  [no download link]"
            print(f"  {name:22s} {human(meta['size']):>9s} archive, "
                  f"{human(meta['extracted']):>9s} extracted{mark}{link}")
            print(f"      {meta['venue']}, {meta['duration']:.1f} s, "
                  f"{meta['messages']:,} messages, camera frames: {meta['camera'] or 'NONE'}")
            print(f"      {meta['note']}")
        return

    if args.all:
        sequences = [n for n, m in SEQUENCES.items() if m["id"]]
    else:
        sequences = args.sequences

    for name in sequences:
        if name not in SEQUENCES:
            print(f"  Unknown sequence: {name}")
            print(f"  Available: {', '.join(SEQUENCES)}")
            sys.exit(1)

    if args.patch_type and not args.extract:
        print("  --patch-type needs --extract (it edits the extracted metadata.yaml).")
        sys.exit(1)

    backend = pick_backend() if args.backend == "auto" else args.backend
    if args.backend != "auto" and backend != "urllib" and not shutil.which(backend):
        print(f"  {backend} was requested but is not installed.")
        sys.exit(1)

    bags = args.dest / EXTRACT_SUBDIR
    to_fetch = [n for n in sequences if not (args.dest / SEQUENCES[n]["file"]).exists()]
    archives = sum(SEQUENCES[n]["size"] for n in to_fetch)
    unpacked = sum(SEQUENCES[n]["extracted"] for n in sequences) if args.extract else 0

    print("=" * 72)
    print("  U-AMC Handheld-Scanner Dataset Downloader")
    print("  UAMHD rig: Livox Avia + Livox Mid-360 + Oak-D Pro RGB, PTP-synced")
    print("=" * 72)
    print(f"\n  Destination: {args.dest}")
    print(f"  Bags unpack to: {bags}")
    print(f"  Sequences:   {len(sequences)} ({', '.join(sequences)})")
    print(f"  To download: {human(archives)}" + (" (the rest are already here)" if archives and len(to_fetch) < len(sequences) else ""))
    if args.extract:
        print(f"  Extracted:   {human(unpacked)}")

    free = shutil.disk_usage(args.dest if args.dest.exists() else Path.home()).free
    need = archives + unpacked
    print(f"  Free space:  {human(free)}")
    if free < need:
        print(f"\n  WARNING: {human(need)} needed but only {human(free)} free.")
    if {"lvi_coex_set_2", "lvi_set_2_restamped"} <= set(sequences):
        print("\n  NOTE: lvi_coex_set_2 and lvi_set_2_restamped are the same recording")
        print("  (same duration, message counts and start time). Unless you want to")
        print("  compare them, lvi_set_2_restamped alone is enough.")
    print()

    args.dest.mkdir(parents=True, exist_ok=True)

    ok, failed = [], []
    for name in sequences:
        meta = SEQUENCES[name]
        archive = args.dest / meta["file"]
        cam = f"{meta['camera']:,} camera frames" if meta["camera"] else "NO camera data"
        print(f"--- {name} ({meta['venue']}, {meta['duration']:.1f} s, {cam}) ---")
        print(f"  {meta['file']} ({human(meta['size'])})")
        good = download(meta["id"], archive, meta["size"], backend)
        if good and args.extract:
            good = extract(archive, bags, meta["dir"])
            if good and args.patch_type:
                patch_type(bags, meta["dir"])
            if good and args.rm_archive:
                archive.unlink()
                print(f"  Removed {archive.name}")
        (ok if good else failed).append(name)
        print()

    print("=" * 72)
    print(f"  Ready: {len(ok)}/{len(sequences)} sequences in {args.dest}")
    if failed:
        print(f"  Incomplete: {', '.join(failed)} -- re-run to resume.")
    print("=" * 72)

    if ok and not args.extract:
        print("\n  Unpack with:")
        for name in ok:
            print(f"    tar --zstd -xf {args.dest / SEQUENCES[name]['file']} -C {bags}")

    if ok and args.extract and not args.patch_type:
        print(f"\n  NOTE: the Avia topic is typed {WRONG_TYPE}, but FAST-LIVO2-ROS2 builds")
        print(f"  against livox_ros_driver2. Rewrite it to {RIGHT_TYPE}")
        print("  in each metadata.yaml (or re-run with --patch-type), or the node sees no LiDAR.")

    if any(SEQUENCES[n]["camera"] == 0 for n in ok):
        print("\n  NOTE: multi_lidar_coex_set has no camera messages. Run it with the")
        print("  LiDAR-inertial launches (mapping_aviz.launch.py / mapping_mid360.launch.py),")
        print("  not the _lvi ones.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
