#!/usr/bin/env python3
"""
Download the Cerberus 2.0 legged-robot dataset to ~/data/cerberus2/.

ROS 1 bags recorded on a Unitree Go1 by the CMU Robotic Exploration Lab, for
Cerberus 2.0 (visual-inertial-leg odometry):
  https://github.com/ShuoYangRobotics/Cerberus2.0
  https://drive.google.com/drive/folders/1Jz3hRNc_yewCvL_H8dJJ5vGTxAGHD31e

Every bag carries the same eight topics -- one body IMU, four foot IMUs, joint
encoders and a rectified stereo infrared pair:

    /unitree_hardware/imu                    sensor_msgs/Imu          ~400 Hz
    /unitree_hardware/joint_foot             sensor_msgs/JointState   ~400 Hz
    /WT901_47_Data .. /WT901_50_Data         sensor_msgs/Imu          ~200 Hz  (RR,FR,FL,RL feet)
    /camera_forward/infra1/image_rect_raw    sensor_msgs/Image         ~15 Hz  (left)
    /camera_forward/infra2/image_rect_raw    sensor_msgs/Image         ~15 Hz  (right)

Ground truth differs by location, and this is the single most important thing to
know before picking a sequence:
  * indoor  -- Optitrack, published in the bag as /natnet_ros/Shuo_Go1/pose.
               cerberus2_main writes it straight out as gt-<dataset>.csv.
  * outdoor -- no pose topic. What ships instead is a MATLAB Mobile `.mat` of
               iPhone GPS/IMU, stored as MATLAB `timetable` *objects* (MCOS),
               which scipy.io.loadmat cannot decode. Turning it into a
               trajectory needs MATLAB plus upstream's
               script/matlab/mobile_gps_process/ on the docker_free_desktop
               branch. Download it anyway (a few MB) so the option exists.

Transfers go through `curl` if present, else `wget`, else stdlib urllib, so there
are no third-party Python dependencies (no gdown needed) and resume works.

Same Drive-download endpoint and resume/verify logic as download_fast_livo2.py.
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


DRIVE_FOLDER = "https://drive.google.com/drive/folders/1Jz3hRNc_yewCvL_H8dJJ5vGTxAGHD31e"

# Google Drive's large-file endpoint. The plain `uc?id=` URL serves an HTML
# confirmation page for anything over ~100 MB; this one streams bytes and honours
# Range requests, so curl -C - / wget -c resume correctly.
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# name -> dict(dir, scene, gt, note, files=[(filename, drive id, exact bytes)])
# Sizes are the exact Content-Length Drive reports, so a truncated transfer is
# detected rather than silently accepted.
SEQUENCES = {
    "mill19_trail": dict(
        dir="mill19_trail", scene="outdoor", gt="iphone-gps",
        note="wooded trail at Mill 19, Pittsburgh; 419 s trotting. The sequence "
             "upstream's README showcases as a video.",
        files=[
            ("230628-mil19-trot-07-039-wild1445.bag", "16l9bc1as-8vsA9Ctg1-Qu9GsIh4WLrBJ", 4153084583),
            ("230628-mil19-trot-07-039-wild.mat", "1EHDuuuC3langG-qkuifgLs47XG5HQfDJ", 3869549),
        ],
    ),
    "cmu_garage": dict(
        dir="cmu_garage", scene="outdoor", gt="iphone-gps (flagged bad by the recorder)",
        note="CMU east campus parking garage, 644 s. Multi-level, so a good test "
             "of vertical drift. Upstream's README figure 1 is this sequence.",
        files=[
            ("230828-cmu-trot-06-040-east-campus-garage-bad-gps.bag", "1M2NeDhnDuCVGbmg8aGZW9pGxfmwSCiUR", 6389213088),
            ("230828-cmu-trot-06-040-east-campus-garage-bad-gps.mat", "1cXNfM15ttiC04mktwaBr50na0AzdRp6L", 5967516),
        ],
    ),
    "frick_park": dict(
        dir="frick_park", scene="outdoor", gt="iphone-gps",
        note="Frick Park arch, Pittsburgh. Smallest outdoor sequence.",
        files=[
            ("230626-frick-park-trot-06-04-arch.bag", "1gTRtgCDblUFTsmkcb4p_D3leSa-EbptI", 3625238272),
            ("230626-frick-park-trot-06-04-arch.mat", "1inh7rDx0JemOMbEihGDzHZlS_FthmgPv", 351063),
        ],
    ),
    "schenley_park": dict(
        dir="schenley_park", scene="outdoor", gt="iphone-gps",
        note="Schenley Park tennis courts; a closed loop, so start-to-end "
             "distance is a usable drift proxy without GPS.",
        files=[
            ("230630-schen-trot-08-038-loop-tennis.bag", "1SmmlS5uB4Ng7qyScVlCV-AuMRqwwgyeg", 2982520404),
            ("230630-schen-trot-08-038-loop-tennis.mat", "1bp5c1-bIlb2POPRZKBH96x8vFNtgyS9c", 2753307),
        ],
    ),
    "st_mary_cemetery": dict(
        dir="st_mary_cemetery", scene="outdoor", gt="iphone-gps",
        note="St Mary Catholic Cemetery loop. Largest sequence in the release.",
        files=[
            ("230828-st-mary-catholic-cemetery-trot-06-040-loop.bag", "1LQ51sLAcuNyNcp23vmf5HubQwxGXnOwx", 7003264032),
            ("230828-st-mary-catholic-cemetery-trot-06-040-loop.mat", "1ArQmC1pNJ8iyF4gcSXn1GeOp6z6JSIl8", 6459875),
        ],
    ),
    "wightman_park_flying_trot": dict(
        dir="wightman_park", scene="outdoor", gt="iphone-gps",
        note="The bag upstream's config/go1_config/hardware_go1_vilo_config.yaml "
             "and launch/vilo_launch.launch name by default. Flying trot, right loop.",
        files=[
            ("20230304_wightman_park_flying_trot_right_loop.bag", "1PLWlBaUQ2WtErG5JdHwv4G2CXRLD2sG1", 1976264808),
            ("20230304wightmanpark_flying_trot.mat", "1LTVRQ-Kbx3z7G_PFlyyRnPptzCoWXAfI", 76756435),
        ],
    ),
    "wightman_park_trot_bridge": dict(
        dir="wightman_park", scene="outdoor", gt="iphone-gps",
        note="Wightman Park, trot across the bridge loop.",
        files=[
            ("20230304_wightman_park_trot_bridge_loop.bag", "19jutPtEWhMZ5_G-HTxHqH1tCqQC8F0b0", 4174522648),
            ("20230304wightmanpark_trot_bridge_loop.mat", "15HLlYt-7Eq7p5ApW7C3Tmc0LWIXVh9Bt", 160638470),
        ],
    ),
    # Indoor sequences carry Optitrack in the bag, so they are the only way to get
    # a real ATE out of this dataset without MATLAB. Pick from the 20230615 /
    # 20230620 / 20230625 series: the 20230517 series records the foot IMUs at
    # 27 Hz instead of 200 Hz and the estimator stops emitting after ~11 s on it.
    "indoor_square_31s": dict(
        dir="indoor", scene="indoor", gt="optitrack (/natnet_ros/Shuo_Go1/pose)",
        note="RISQH lab, stand-trot round a 3 m square, 31 s. Smallest sequence in "
             "the release and the one cerberus_2/README.md quotes an ATE from.",
        files=[
            ("230620-risqh-standtrot-05-06-33square1.bag", "1SKDBbw1ZMEuGHXe2h6oPK9SsnpAJCvX8", 305442466),
        ],
    ),
    "indoor_square_93s": dict(
        dir="indoor", scene="indoor", gt="optitrack (/natnet_ros/Shuo_Go1/pose)",
        note="RISQH lab, stand-trot 0.6 m/s round a square, 93 s. Three times longer "
             "than indoor_square_31s, and every estimator variant diverges on it.",
        files=[
            ("20230615-risqh-standtrot-06-06-square.bag", "16fuIvS2XIcUp44ENxhA7x6KGDXh69OZx", 933235147),
        ],
    ),
    "indoor_two_loops_27hz": dict(
        dir="indoor", scene="indoor", gt="optitrack (/natnet_ros/Shuo_Go1/pose)",
        note="RISQH lab, 0.6 m/s, 4-4 gait, two loops. NOT RECOMMENDED: foot IMUs at "
             "27 Hz, estimator stops emitting after ~11 s. Kept for reference.",
        files=[
            ("20230517_risqh_06speed_44gait_two_loops.bag", "1rigruHNjHgz9ftSK2RdW7tpxEgBG5A1Y", 790846032),
        ],
    ),
}

# The two sequences cerberus_2/README.md verifies, and upstream's two headline
# results (Mill19 Trail video, CMU Garage figure).
DEFAULT_SEQUENCES = ["mill19_trail", "cmu_garage"]

DEST_DIR = Path.home() / "data" / "cerberus2"

ROSBAG_MAGIC = b"#ROSBAG V2.0"


def human(n: int) -> str:
    return f"{n / 1024**3:.2f} GB" if n >= 1024**3 else f"{n / 1024**2:.1f} MB"


def seq_bytes(name: str) -> int:
    return sum(sz for _, _, sz in SEQUENCES[name]["files"])


def fetch_curl(url: str, dest: Path) -> bool:
    """curl, resuming with -C -.

    `-f` is not optional: without it a Drive error response lands on disk as a
    ~1.6 kB HTML page and curl still exits 0, i.e. you get a ".bag" that is really
    an error message. With -f, curl exits 22 and writes nothing.
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


def download(file_id: str, dest: Path, total: int, backend: str) -> bool:
    """Fetch a Drive file to `dest`, resuming a partial file if one exists."""
    url = DOWNLOAD_URL.format(id=file_id)

    have = dest.stat().st_size if dest.exists() else 0
    if have == total:
        print(f"  Skipping {dest.name} (complete, {human(have)})")
        return True
    if have > total:
        # Never resume onto an over-long file: that silently produces a corrupt
        # bag that still passes the magic-byte check.
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
    if dest.suffix == ".bag":
        with open(dest, "rb") as f:
            if f.read(len(ROSBAG_MAGIC)) != ROSBAG_MAGIC:
                print(f"  {dest.name} does not start with '{ROSBAG_MAGIC.decode()}' "
                      "-- not a ROS 1 bag (an HTML error page?). Delete it and retry.")
                return False
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the Cerberus 2.0 Go1 dataset (stereo + body IMU + 4 foot IMUs + joints)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sequences:\n"
        + "\n".join(f"  {n:26s} {human(seq_bytes(n)):>9s}  {SEQUENCES[n]['scene']:<8s} gt: {SEQUENCES[n]['gt']}"
                    for n in SEQUENCES)
        + f"\n\nDefault: {', '.join(DEFAULT_SEQUENCES)}"
        + "\n\nRun with Cerberus 2.0 (see SLAM_zero_to_hero/cerberus_2/README.md):\n"
        "  podman run ... slam_zero_to_hero:cerberus_2 bash /opt/cerberus2_demo/run_demo.sh\n",
    )
    parser.add_argument("sequences", nargs="*", default=DEFAULT_SEQUENCES,
                        help=f"Sequence name(s) (default: {', '.join(DEFAULT_SEQUENCES)})")
    parser.add_argument("--all", action="store_true", help="Download every sequence (~33 GB)")
    parser.add_argument("--indoor", action="store_true",
                        help="Download only the two mocap-ground-truth indoor sequences")
    parser.add_argument("--list", action="store_true", help="List sequences and exit")
    parser.add_argument("--dest", type=Path, default=DEST_DIR,
                        help=f"Destination directory (default: {DEST_DIR})")
    parser.add_argument("--backend", choices=("auto", "curl", "wget", "urllib"), default="auto",
                        help="Transfer tool (default: auto = curl, else wget, else stdlib urllib)")
    args = parser.parse_args()

    if args.list:
        print(f"Cerberus 2.0 dataset ({DRIVE_FOLDER})\n")
        for name, meta in SEQUENCES.items():
            mark = " <- default" if name in DEFAULT_SEQUENCES else ""
            print(f"  {name:26s} {human(seq_bytes(name)):>9s}  {meta['scene']:<8s}{mark}")
            print(f"      gt: {meta['gt']}")
            print(f"      {meta['note']}")
        return

    if args.all:
        sequences = list(SEQUENCES)
    elif args.indoor:
        sequences = [n for n, m in SEQUENCES.items() if m["scene"] == "indoor"]
    else:
        sequences = args.sequences

    for name in sequences:
        if name not in SEQUENCES:
            print(f"  Unknown sequence: {name}")
            print(f"  Available: {', '.join(SEQUENCES)}")
            sys.exit(1)

    backend = pick_backend() if args.backend == "auto" else args.backend
    if args.backend != "auto" and backend != "urllib" and not shutil.which(backend):
        print(f"  {backend} was requested but is not installed.")
        sys.exit(1)

    total = sum(seq_bytes(n) for n in sequences)

    print("=" * 72)
    print("  Cerberus 2.0 Dataset Downloader")
    print("  Unitree Go1: body IMU + 4 foot IMUs + joint encoders + stereo IR")
    print("=" * 72)
    print(f"\n  Destination: {args.dest}")
    print(f"  Sequences:   {len(sequences)} ({', '.join(sequences)})")
    print(f"  Total size:  {human(total)}")
    print(f"  Transfer:    {backend}\n")

    ok, failed = [], []
    for name in sequences:
        meta = SEQUENCES[name]
        out = args.dest / meta["dir"]
        out.mkdir(parents=True, exist_ok=True)
        print(f"--- {name} ({meta['scene']}, {human(seq_bytes(name))}) -> {out} ---")
        good = True
        for fname, fid, fsize in meta["files"]:
            print(f"  {fname} ({human(fsize)})")
            good &= download(fid, out / fname, fsize, backend)
        (ok if good else failed).append(name)
        print()

    print("=" * 72)
    print(f"  Downloaded {len(ok)}/{len(sequences)} sequences to {args.dest}")
    if failed:
        print(f"  Incomplete: {', '.join(failed)} -- re-run to resume.")
    print("=" * 72)

    if any(SEQUENCES[n]["scene"] == "outdoor" for n in ok):
        print("\n  NOTE: the outdoor .mat files are MATLAB Mobile `timetable` objects")
        print("  (MCOS). scipy.io.loadmat cannot read them; producing a GPS ground-truth")
        print("  trajectory needs MATLAB and upstream's script/matlab/mobile_gps_process/")
        print("  on the docker_free_desktop branch. The bags themselves carry no pose topic.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
