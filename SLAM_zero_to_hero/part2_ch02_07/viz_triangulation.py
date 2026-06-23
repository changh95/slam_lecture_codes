#!/usr/bin/env python3
"""
Rerun viewer for the triangulation demos.

Reads one or more JSON files produced by triangulation_demo / triangulation_opengv
and logs:
  - the left/right KITTI images,
  - the ORB inlier keypoints overlaid on each image,
  - one 3D point cloud per triangulation method,
  - 3D camera frustums for cam0 (left, at origin) and cam1 (right, +X baseline).

Usage:
    python3 viz_triangulation.py triangulation_demo.json [triangulation_opengv.json ...]
    python3 viz_triangulation.py --save out.rrd triangulation_demo.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image

# Distinct color per triangulation method.
METHOD_COLORS = {
    "opencv":         (0,   200, 0),    # green
    "dlt":            (0,   120, 255),  # blue
    "midpoint":       (220, 60,  60),   # red
    "stereo":         (240, 200, 0),    # yellow
    "opengv_linear":  (180, 0,   220),  # purple
    "opengv_midpoint": (255, 100, 200),  # pink
}


def load_image(path: str) -> np.ndarray:
    try:
        return np.asarray(Image.open(path).convert("L"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot read image: {path}")


def log_camera(name: str, position: np.ndarray, fx: float, fy: float,
               cx: float, cy: float, width: int, height: int, image: np.ndarray):
    rr.log(name, rr.Transform3D(translation=position))
    rr.log(name,
           rr.Pinhole(focal_length=[fx, fy], principal_point=[cx, cy],
                      resolution=[width, height],
                      camera_xyz=rr.ViewCoordinates.RDF))
    rr.log(f"{name}/image", rr.Image(image))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+", help="JSON files from the C++ demos")
    ap.add_argument("--save", default=None, help="Save to .rrd instead of spawning a viewer")
    ap.add_argument("--connect", nargs="?", const="rerun+http://127.0.0.1:9876/proxy",
                    default=None, metavar="URL",
                    help="stream to an already-running rerun viewer at this gRPC "
                         "URL (default rerun+http://127.0.0.1:9876/proxy)")
    args = ap.parse_args()

    rr.init("triangulation", spawn=(args.save is None and args.connect is None))
    if args.connect:
        rr.connect_grpc(args.connect)
    elif args.save:
        rr.save(args.save)

    # World axes (1 m each).
    rr.log("world",
           rr.Arrows3D(vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                       labels=["x", "y", "z"]),
           static=True)

    cameras_logged = False
    for jf in args.json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        if not cameras_logged:
            fx, fy = data["fx"], data["fy"]
            cx, cy = data["cx"], data["cy"]
            W, H = data["width"], data["height"]
            baseline = data["baseline"]
            left  = load_image(data["left_image"])
            right = load_image(data["right_image"])

            log_camera("world/cam_left",  np.array([0.0, 0.0, 0.0]),
                       fx, fy, cx, cy, W, H, left)
            log_camera("world/cam_right", np.array([baseline, 0.0, 0.0]),
                       fx, fy, cx, cy, W, H, right)

            kp_left  = np.asarray(data["keypoints_left"],  dtype=np.float32)
            kp_right = np.asarray(data["keypoints_right"], dtype=np.float32)
            rr.log("world/cam_left/image/keypoints",
                   rr.Points2D(kp_left,  colors=[255, 255, 0], radii=2.0))
            rr.log("world/cam_right/image/keypoints",
                   rr.Points2D(kp_right, colors=[255, 255, 0], radii=2.0))
            cameras_logged = True

        source_tag = Path(jf).stem
        for method, color in METHOD_COLORS.items():
            if method not in data:
                continue
            entries = data[method]
            if not entries:
                continue
            xyz = np.asarray([e["xyz"] for e in entries], dtype=np.float32)
            rr.log(f"world/points/{source_tag}/{method}",
                   rr.Points3D(xyz, colors=color, radii=0.04,
                               labels=[method] * len(xyz)))
            print(f"[{source_tag}] {method:>16s}: {len(xyz)} points")

    if args.connect:
        try:
            rr.flush()  # block until the recording reaches the viewer
        except Exception:
            import time
            time.sleep(2.0)
        print(f"Streamed to viewer at {args.connect}")
    elif args.save:
        print(f"Saved to {args.save}")
    else:
        print("Rerun viewer launched. Close the window to exit.")


if __name__ == "__main__":
    sys.exit(main())
