#!/usr/bin/env python3
"""
Download the FAST-LIVO2-Dataset to ~/data/fast_livo2/.

Transfers go through `curl` if present, else `wget`, else stdlib urllib -- so there
are no third-party Python dependencies, and on a normal machine you get curl's
retry/backoff and resume for free. Force one with --backend. Note that `gdown`
does NOT work for this folder at all: it fails with "Cannot retrieve the public
link of the file ... or have had many accesses" even though the files are public.

ROS 1 bag files from a handheld Livox Avia (LiDAR + built-in IMU) plus an RGB
camera -- the hardware FAST-LIVO2's `avia.yaml` / `mapping_avia.launch` are
tuned for. Topics: /livox/lidar, /livox/imu, /left_camera/image.

Hosted on Google Drive by the FAST-LIVO2 authors. FAST-LIVO2's own README points
at the Global-LVBA repository for this dataset, because the older HKU SharePoint
links rotate:
  https://github.com/hku-mars/FAST-LIVO2
  https://github.com/xuankuzcr/Global-LVBA   (Section IV: Dataset download)

IMPORTANT -- calibration is per-sequence. The authors recalibrated the rig every
few months, so `calibration.yaml` holds FOUR different Rcl/Pcl + intrinsics
sets. FAST-LIVO2's shipped avia.yaml + camera_pinhole.yaml match the
Retail_Street / CBD_Building_01 / Bright_Screen_Wall group; every other sequence
needs the matching block pasted in. See CALIB_GROUPS below and
SLAM_zero_to_hero/fast_livo2/README.md.

Do NOT confuse this with the *LVBA-Dataset* in the Global-LVBA repo: that one is
FAST-LIVO2's *output* (per-frame PNG + PCD + pose files) and feeds Global-LVBA,
not FAST-LIVO2.
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


DRIVE_FOLDER = "https://drive.google.com/drive/folders/1bf5LQ8iSxw-fD8BObZmouw7lRxNacfrA"

# Google Drive's large-file endpoint. The plain `uc?id=` URL serves an HTML
# confirmation page for anything over ~100 MB; this one streams bytes and honours
# Range requests (verified: HTTP 206 + Content-Range), so plain curl -C - or
# wget -c resume correctly and no Drive-specific client is needed.
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# name -> (google drive file id, size as reported by Drive)
SEQUENCES = {
    # The two sequences FAST-LIVO2's README showcases, plus the one its
    # avia.yaml already names in `evo.seq_name`.
    "Retail_Street":            ("1DKOfH8pfObRenoWf4-IfkTMHcliXnigD", "1.8G"),
    "Red_Sculpture":            ("1t9plYoZJteeVINTRmogIcySiPOCsMTeZ", "4.2G"),
    "CBD_Building_01":          ("1UniOSfIgotDULta_uPpUE769azP0TWl_", "4.9G"),
    # Small, high-dynamic-range scene -- handy as a quick smoke test.
    "Bright_Screen_Wall":       ("1P0rwzyIyj3c58tywdDZEdLt8P5DK5sEI", "360M"),
    # The rest.
    "CBD_Building_02":          ("1Jd09V7J7YkEU7ZTuuUPJ2faivYro1nAV", "9.6G"),
    "CBD_Building_03":          ("1y4t7zXk50nu-5ZxQ8nTri78cUgORdudt", "11G"),
    "HIT_Graffiti_Wall_01":     ("1OudCWy998kISnJelu8xNFbUAKM2w3cgH", "20G"),
    "HIT_Graffiti_Wall_02":     ("18SyuS4SR7DCmKPx8MYbVLtxmdoo3R3To", "22G"),
    "HIT_Graffiti_Wall_03":     ("1CA27aTscYmrPyt7B1LV-14qorw700sZp", "10G"),
    "HIT_Graffiti_Wall_04":     ("1q2EZjrQV1T4NtYMAqgy3KMDN4BL3Xdt7", "7.6G"),
    "HKU_Centennial_Garden_01": ("15X21HhXfRxEvAq-KPUwFQvnJEo-7-58B", "3.8G"),
    "HKU_Centennial_Garden_02": ("1innScHcQvQsTBsXHyPPgvIg1S4aSV_X5", "3.7G"),
    "HKU_Cultural_Center_01":   ("180-SKN4xsIVc1TJWtyH1C4pz1OBmJtul", "8.1G"),
    "HKU_Cultural_Center_02":   ("1HPdHWMmv_YyMRB2_PttxLA0JhFd_6NmH", "9.9G"),
    "HKU_Landmark":             ("194gEIlkvrUsExoQlJXbq4vM6v1pvuJSP", "3.7G"),
    "HKU_Lecture_Center_01":    ("1W3ex0-180l0GhXXf3rwEvUN74NzplSS1", "3.2G"),
    "HKU_Lecture_Center_02":    ("1aTc3db6T1zoUF3GNZ0CaaZvrv5ZufByx", "3.0G"),
    "HKU_Main_Building":        ("1eyOdK9yteK3LUkxaM3UAxg85vukq0B51", "3.5G"),
    "SYSU_01":                  ("1R1y-T-0toyl38K4ZVQ6KW9qmxNWzjaZZ", "5.5G"),
    "SYSU_02":                  ("1eEVRhlOxvDUo8wmJiib17IuLXnU6Rhmj", "5.6G"),
}

# Sensor calibration shipped alongside the bags. Always fetched -- it is tiny.
CALIBRATION = ("calibration.yaml", "1wbC88P8xgM0YnexRzNaKsPYxpKpsY86u")

# Which sequences share a calibration block in calibration.yaml. Group 1 is the
# one FAST-LIVO2's stock avia.yaml + camera_pinhole.yaml already encode.
CALIB_GROUPS = {
    "group 1 (stock avia.yaml/camera_pinhole.yaml)":
        ["Retail_Street", "CBD_Building_01", "Bright_Screen_Wall"],
    "group 2":
        ["HKU_Landmark", "HKU_Centennial_Garden_01", "HKU_Centennial_Garden_02",
         "HKU_Main_Building", "HKU_Cultural_Center_01", "HKU_Cultural_Center_02",
         "HKU_Lecture_Center_01", "HKU_Lecture_Center_02",
         "CBD_Building_02", "CBD_Building_03"],
    "group 3":
        ["HIT_Graffiti_Wall_01", "HIT_Graffiti_Wall_02", "HIT_Graffiti_Wall_03",
         "HIT_Graffiti_Wall_04", "SYSU_01", "SYSU_02"],
    "group 4":
        ["Red_Sculpture"],
}

# FAST-LIVO2's two headline demos plus the sequence its config already names.
POPULAR = ["Retail_Street", "Red_Sculpture", "CBD_Building_01"]

# Smallest full sequence, the one upstream demos first, and covered by the
# stock calibration -- so it runs with no config edits at all.
DEFAULT_SEQUENCE = "Retail_Street"

DEST_DIR = Path.home() / "data" / "fast_livo2"

ROSBAG_MAGIC = b"#ROSBAG V2.0"


def size_to_gb(size: str) -> float:
    """'1.8G' -> 1.8, '360M' -> 0.36."""
    value, unit = float(size[:-1]), size[-1]
    return value if unit == "G" else value / 1024.0


def remote_size(url: str) -> int:
    """Total size of the target, via a 1-byte Range probe. 0 if unknown."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        crange = r.headers.get("Content-Range", "")
        return int(crange.split("/")[-1]) if "/" in crange else 0


def fetch_curl(url: str, dest: Path) -> bool:
    """curl, resuming with -C -.

    `-f` is not optional: without it, a Drive error response is written to disk
    as a 1652-byte HTML page and curl still exits 0 -- i.e. you get a ".bag"
    that is really an error message. With -f, curl exits 22 and writes nothing.
    """
    cmd = ["curl", "-fL", "--retry", "5", "--retry-delay", "2",
           "--retry-connrefused", "--connect-timeout", "30",
           "-C", "-", "--progress-bar", "-o", str(dest), url]
    return subprocess.run(cmd).returncode == 0


def fetch_wget(url: str, dest: Path) -> bool:
    """wget, resuming with -c.

    Verified against this endpoint: `-c` with `-O` does honour the offset (the
    server answers 206 Partial Content and wget continues rather than
    restarting). On an HTTP error wget exits non-zero but leaves a 0-byte file
    behind, which the size check below catches.
    """
    cmd = ["wget", "-c", "--tries=5", "--timeout=60", "--waitretry=2",
           "--progress=dot:giga", "-O", str(dest), url]
    return subprocess.run(cmd).returncode == 0


def fetch_urllib(url: str, dest: Path, have: int, total: int) -> bool:
    """Fallback with no external tools: stdlib only, manual Range resume."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    done = have
    # A carriage return only collapses on a terminal; when stdout is a log file
    # every update becomes another line, so report far less often there.
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
    """curl > wget > urllib. curl and wget bring real retry/backoff and resume."""
    for tool in ("curl", "wget"):
        if shutil.which(tool):
            return tool
    return "urllib"


def download(file_id: str, dest: Path, expect_rosbag: bool = True,
             backend: str = "auto") -> bool:
    """Fetch a Drive file to `dest`, resuming a partial file if one exists."""
    url = DOWNLOAD_URL.format(id=file_id)

    try:
        total = remote_size(url)
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not query {dest.name}: {exc}")
        return False

    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"  Skipping {dest.name} (complete, {have / 1024**3:.2f} GB)")
        return True
    if have > total > 0:
        # Never let a resume append onto an over-long file: that silently
        # produces a corrupt archive that still passes a magic-byte check.
        print(f"  {dest.name} is larger than expected ({have} > {total}); starting over")
        dest.unlink()
        have = 0
    if have:
        print(f"  Resuming {dest.name} at {have / 1024**3:.2f} GB")

    if backend == "auto":
        backend = pick_backend()

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

    # Belt and braces, whichever backend ran: exact length, then file type.
    got = dest.stat().st_size if dest.exists() else 0
    if total and got != total:
        print(f"  Size mismatch for {dest.name}: {got} != {total}. Re-run to resume.")
        return False
    if expect_rosbag:
        with open(dest, "rb") as f:
            if f.read(len(ROSBAG_MAGIC)) != ROSBAG_MAGIC:
                print(f"  {dest.name} does not start with '{ROSBAG_MAGIC.decode()}' "
                      "-- not a ROS 1 bag (an HTML error page?). Delete it and retry.")
                return False
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the FAST-LIVO2-Dataset (Livox Avia + IMU + RGB camera)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sequences:\n"
        + "\n".join(f"  {n:26s} {s}" for n, (_, s) in SEQUENCES.items())
        + f"\n\nDefault: {DEFAULT_SEQUENCE}"
        + f"\nPopular subset (--popular): {', '.join(POPULAR)}"
        + "\n\nRun with FAST-LIVO2:\n"
        "  roslaunch fast_livo mapping_avia.launch    # avia.yaml + camera_pinhole.yaml\n"
        "  rosbag play Retail_Street.bag\n"
        "Only the group-1 sequences work with the stock calibration; see --calib.\n"
        "SLAM_zero_to_hero/fast_livo2/README.md has a headless container recipe.",
    )
    parser.add_argument(
        "sequences", nargs="*", default=[DEFAULT_SEQUENCE],
        help=f"Sequence name(s) to download (default: {DEFAULT_SEQUENCE})",
    )
    parser.add_argument(
        "--popular", action="store_true",
        help=f"Download the popular subset: {', '.join(POPULAR)}",
    )
    parser.add_argument("--all", action="store_true", help="Download every sequence (~150 GB)")
    parser.add_argument("--list", action="store_true", help="List sequences and exit")
    parser.add_argument("--calib", action="store_true",
                        help="Show which sequences share a calibration block, and exit")
    parser.add_argument(
        "--dest", type=Path, default=DEST_DIR,
        help=f"Destination directory (default: {DEST_DIR})",
    )
    parser.add_argument(
        "--backend", choices=("auto", "curl", "wget", "urllib"), default="auto",
        help="Transfer tool (default: auto = curl, else wget, else stdlib urllib)",
    )
    args = parser.parse_args()

    if args.calib:
        print("calibration.yaml holds one block per group; sequences in different\n"
              "groups need different Rcl/Pcl and camera intrinsics.\n")
        for group, names in CALIB_GROUPS.items():
            print(f"  {group}:")
            for n in names:
                print(f"      {n}")
        return

    if args.list:
        print(f"FAST-LIVO2-Dataset ({DRIVE_FOLDER})\n")
        for name, (_, size) in SEQUENCES.items():
            tags = []
            if name in POPULAR:
                tags.append("popular")
            if name in CALIB_GROUPS["group 1 (stock avia.yaml/camera_pinhole.yaml)"]:
                tags.append("stock calib")
            suffix = f"  <- {', '.join(tags)}" if tags else ""
            print(f"  {name:26s} {size:>6s}{suffix}")
        print(f"\n  {CALIBRATION[0]:26s}  small   <- always downloaded")
        return

    if args.all:
        sequences = list(SEQUENCES)
    elif args.popular:
        sequences = list(POPULAR)
    else:
        sequences = args.sequences

    for name in sequences:
        if name not in SEQUENCES:
            print(f"  Unknown sequence: {name}")
            print(f"  Available: {', '.join(SEQUENCES)}")
            sys.exit(1)

    total = sum(size_to_gb(SEQUENCES[n][1]) for n in sequences)

    backend = pick_backend() if args.backend == "auto" else args.backend
    if args.backend != "auto" and backend != "urllib" and not shutil.which(backend):
        print(f"  {backend} was requested but is not installed.")
        sys.exit(1)

    print("=" * 68)
    print("  FAST-LIVO2-Dataset Downloader")
    print("  Livox Avia + built-in IMU + RGB camera (ROS 1 bags)")
    print("=" * 68)
    print(f"\n  Destination: {args.dest}")
    print(f"  Sequences:   {len(sequences)} ({', '.join(sequences)})")
    print(f"  Total size:  ~{total:.1f} GB")
    print(f"  Transfer:    {backend}\n")

    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"--- {CALIBRATION[0]} ---")
    download(CALIBRATION[1], args.dest / CALIBRATION[0], expect_rosbag=False,
             backend=backend)

    ok, failed = [], []
    for name in sequences:
        file_id, size = SEQUENCES[name]
        print(f"\n--- {name}.bag ({size}) ---")
        (ok if download(file_id, args.dest / f"{name}.bag", backend=backend)
         else failed).append(name)

    print("\n" + "=" * 68)
    print(f"  Downloaded {len(ok)}/{len(sequences)} sequences to {args.dest}")
    if failed:
        print(f"  Incomplete: {', '.join(failed)} -- re-run to resume.")
    print("=" * 68)

    if args.dest.exists():
        print("\n  Contents:")
        for item in sorted(args.dest.iterdir()):
            mb = item.stat().st_size / 1024**2
            print(f"    {item.name:32s} "
                  f"{f'{mb / 1024:.2f} GB' if mb > 1024 else f'{mb:.1f} MB'}")

    for name in ok:
        if name not in CALIB_GROUPS["group 1 (stock avia.yaml/camera_pinhole.yaml)"]:
            print(f"\n  NOTE: {name} is NOT covered by FAST-LIVO2's stock calibration.")
            print("  Copy its Rcl/Pcl + intrinsics out of calibration.yaml before running.")
            print("  Run this script with --calib to see the groups.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
