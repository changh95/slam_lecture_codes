#!/usr/bin/env python3
"""
Live ORB-SLAM2 -> Rerun bridge.

Runs the patched `mono_kitti_rerun` binary (which prints JSON lines) as a
subprocess and logs all messages to Rerun in real time.

Message types:
  pose    - current camera pose
  kpts    - current frame keypoints (matched vs unmatched)
  tmpts   - 3D map points matched in current frame (yellow)
  lmpts   - local/reference map points used in local mapping (green)
  ampts   - all map points (faded blue), sent every 20 frames

Usage:
    python3 live_rerun.py <kitti_sequence_path>
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import rerun as rr
from PIL import Image

VOC = "/Portable_ORB_SLAM2/Vocabulary/ORBvoc.txt"
CFG = "/Portable_ORB_SLAM2/Examples/Monocular/KITTI00-02.yaml"
EXE = "/Portable_ORB_SLAM2/Examples/Monocular/mono_kitti_rerun"

# Colors (RGB)
COLOR_KPT_MATCHED = [0, 255, 0]     # green - keypoints with map point
COLOR_KPT_RAW     = [255, 0, 0]     # red   - unmatched keypoints
COLOR_TMPTS       = [255, 200, 0]   # yellow - tracked map points (current frame)
COLOR_LMPTS       = [0, 255, 0]     # green - local/reference map points
COLOR_AMPTS       = [120, 160, 255] # pale blue - all map points
COLOR_TRAJ        = [0, 255, 0]     # green trajectory


def load_times(seq_path: str) -> np.ndarray:
    with open(os.path.join(seq_path, "times.txt")) as f:
        return np.array([float(line) for line in f.readlines()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seq", help="Path to KITTI sequence (e.g. /data/sequences/00)")
    args = parser.parse_args()

    rr.init("orb_slam2")
    rr.serve_grpc(grpc_port=9876)
    rr.serve_web_viewer(
        web_port=9090,
        open_browser=False,
        connect_to="rerun+http://localhost:9876/proxy",
    )
    print("\n=== http://localhost:9090/?url=rerun+http://localhost:9876/proxy ===\n", flush=True)

    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    img_times = load_times(args.seq)
    trajectory = []

    cmd = [EXE, VOC, CFG, args.seq]
    print(f"Launching: {' '.join(cmd)}", flush=True)

    workdir = "/tmp/orbslam2_run"
    os.makedirs(workdir, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        cwd=workdir,
    )

    current_image = None
    current_ts = 0.0
    last_pose_ts = -1.0
    frame_count = 0

    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line.startswith("{"):
                print(line, flush=True)
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("t")
            ts = msg.get("ts", 0.0)
            rr.set_time("frame", timestamp=ts)

            if mtype == "pose":
                current_ts = ts
                tx, ty, tz = msg["tx"], msg["ty"], msg["tz"]
                qx, qy, qz, qw = msg["qx"], msg["qy"], msg["qz"], msg["qw"]

                trajectory.append([tx, ty, tz])
                rr.log(
                    "slam/pose",
                    rr.Transform3D(
                        translation=[tx, ty, tz],
                        quaternion=[qx, qy, qz, qw],
                    ),
                )
                rr.log("slam/pose/body", rr.Boxes3D(
                    centers=[[0, 0, 0]], sizes=[[1.6, 1.65, 2.71]]
                ))
                rr.log("slam/trajectory", rr.LineStrips3D(
                    [np.array(trajectory, dtype=np.float32)],
                    colors=[COLOR_TRAJ],
                ))

                # Log matching image (under the camera transform, 2D view)
                idx = int(np.argmin(np.abs(img_times - ts)))
                img_path = os.path.join(args.seq, "image_0", f"{idx:06d}.png")
                if os.path.exists(img_path):
                    current_image = np.asarray(Image.open(img_path))
                    rr.log("slam/pose/cam/image", rr.Image(current_image).compress(jpeg_quality=70))

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"  Frame {frame_count}: traj_len={len(trajectory)}", flush=True)

            elif mtype == "kpts":
                uv_matched = np.array(msg.get("uv_matched", []), dtype=np.float32)
                uv_raw = np.array(msg.get("uv_raw", []), dtype=np.float32)
                if len(uv_matched):
                    rr.log(
                        "slam/pose/cam/image/kpts_matched",
                        rr.Points2D(uv_matched, radii=3.0, colors=[COLOR_KPT_MATCHED]),
                    )
                else:
                    rr.log("slam/pose/cam/image/kpts_matched", rr.Points2D([]))
                if len(uv_raw):
                    rr.log(
                        "slam/pose/cam/image/kpts_raw",
                        rr.Points2D(uv_raw, radii=2.0, colors=[COLOR_KPT_RAW]),
                    )
                else:
                    rr.log("slam/pose/cam/image/kpts_raw", rr.Points2D([]))

            elif mtype == "tmpts":
                xyz = np.array(msg.get("xyz", []), dtype=np.float32)
                if len(xyz):
                    rr.log(
                        "slam/map/tracked",
                        rr.Points3D(xyz, colors=[COLOR_TMPTS], radii=0.15),
                    )
                else:
                    rr.log("slam/map/tracked", rr.Points3D([]))

            elif mtype == "lmpts":
                xyz = np.array(msg.get("xyz", []), dtype=np.float32)
                if len(xyz):
                    rr.log(
                        "slam/map/local",
                        rr.Points3D(xyz, colors=[COLOR_LMPTS], radii=0.1),
                    )
                else:
                    rr.log("slam/map/local", rr.Points3D([]))

            elif mtype == "ampts":
                xyz = np.array(msg.get("xyz", []), dtype=np.float32)
                if len(xyz):
                    rr.log(
                        "slam/map/all",
                        rr.Points3D(xyz, colors=[COLOR_AMPTS], radii=0.05),
                    )

    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()
    print(f"ORB-SLAM2 finished. {frame_count} frames streamed. Viewer still serving.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
