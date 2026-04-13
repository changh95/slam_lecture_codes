#!/usr/bin/env python3
"""
ORB-SLAM2 -> Rerun replay visualizer.

ORB-SLAM2 saves CameraTrajectory.txt on Shutdown() in the format:
    timestamp tx ty tz qx qy qz qw

This script runs ORB-SLAM2 on a KITTI sequence, then replays the saved
trajectory + KITTI images in Rerun.

Usage:
    python3 replay_rerun.py <kitti_sequence_path>
    # e.g. python3 replay_rerun.py /data/sequences/00
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import rerun as rr
from PIL import Image


def run_orbslam2(seq_path: str, workdir: str) -> str:
    """Run ORB-SLAM2 mono_kitti on the sequence, return path to trajectory file."""
    os.chdir(workdir)
    voc = "/Portable_ORB_SLAM2/Vocabulary/ORBvoc.txt"
    cfg = "/Portable_ORB_SLAM2/Examples/Monocular/KITTI00-02.yaml"
    exe = "/Portable_ORB_SLAM2/Examples/Monocular/mono_kitti"
    print(f"Running: {exe} {voc} {cfg} {seq_path}")
    env = os.environ.copy()
    env.setdefault("DISPLAY", "")
    # Xvfb wrap so Pangolin GUI doesn't fail
    subprocess.run(
        ["xvfb-run", "-a", exe, voc, cfg, seq_path],
        check=False,
        env=env,
    )
    traj = os.path.join(workdir, "KeyFrameTrajectory.txt")
    if not os.path.exists(traj):
        raise RuntimeError(f"Trajectory file not found at {traj}")
    return traj


def load_trajectory(path: str) -> np.ndarray:
    """Load timestamp tx ty tz qx qy qz qw into an (N, 8) array."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data


def main():
    parser = argparse.ArgumentParser(description="ORB-SLAM2 -> Rerun replay")
    parser.add_argument("seq", help="Path to KITTI sequence (e.g. /data/sequences/00)")
    parser.add_argument("--traj", help="Pre-computed trajectory file; skips running SLAM")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames")
    parser.add_argument("--fps", type=float, default=10.0, help="Replay frames per second")
    args = parser.parse_args()

    # Initialize Rerun with web viewer
    rr.init("orb_slam2")
    rr.serve_grpc(grpc_port=9876)
    rr.serve_web_viewer(
        web_port=9090,
        open_browser=False,
        connect_to="rerun+http://localhost:9876/proxy",
    )
    print("\n=== Rerun web viewer at http://localhost:9090/?url=rerun+http://localhost:9876/proxy ===\n")

    # World axes
    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    workdir = "/tmp/orbslam2_run"
    os.makedirs(workdir, exist_ok=True)

    # Run ORB-SLAM2 (unless a trajectory file is provided)
    if args.traj:
        traj_file = args.traj
    else:
        print("Step 1/2: Running ORB-SLAM2 on sequence (this takes a few minutes)...")
        traj_file = run_orbslam2(args.seq, workdir)
        print(f"Trajectory saved: {traj_file}")

    # Load trajectory
    traj = load_trajectory(traj_file)
    print(f"Loaded {len(traj)} keyframes")

    # Replay: for each keyframe, log pose + matching image
    print("Step 2/2: Replaying trajectory in Rerun...")
    trajectory_xyz = []

    # Match keyframe timestamps to image frame indices via times.txt
    times_path = os.path.join(args.seq, "times.txt")
    with open(times_path) as f:
        img_times = np.array([float(line) for line in f.readlines()])

    n = len(traj)
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    for i in range(n):
        ts, tx, ty, tz, qx, qy, qz, qw = traj[i]
        rr.set_time("frame", timestamp=ts)

        trajectory_xyz.append([tx, ty, tz])
        rr.log(
            "slam/pose",
            rr.Transform3D(
                translation=[tx, ty, tz],
                quaternion=[qx, qy, qz, qw],
            ),
        )
        rr.log("slam/pose/body", rr.Boxes3D(centers=[[0, 0, 0.8]], sizes=[[1.6, 1.65, 2.71]]))
        rr.log("slam/trajectory", rr.LineStrips3D([np.array(trajectory_xyz)], colors=[[0, 255, 0]]))

        # Find closest matching image
        idx = int(np.argmin(np.abs(img_times - ts)))
        img_path = os.path.join(args.seq, "image_0", f"{idx:06d}.png")
        if os.path.exists(img_path):
            img = np.asarray(Image.open(img_path))
            rr.log("slam/pose/cam/image", rr.Image(img).compress(jpeg_quality=80))

        if i % 50 == 0:
            print(f"  Replayed {i}/{n}")
        time.sleep(1.0 / args.fps)

    print(f"Done. Replayed {n} poses.")
    print("Web viewer still running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
