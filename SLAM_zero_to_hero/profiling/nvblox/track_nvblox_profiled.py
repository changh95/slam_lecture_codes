#!/usr/bin/env python3
"""
track_nvblox_profiled.py - nvblox 3D reconstruction profiler with easy_profiler-compatible
JSON output.

nvblox is open-source (Apache 2.0) and provides a C++ API. This script wraps the
nvblox_python bindings (or falls back to subprocess-calling the C++ fuse_3dmatch binary)
and profiles each major pipeline stage:

  Mapper/Init       - scene setup and TSDF/color/mesh layer construction
  Mapper/Integrate  - per-frame TSDF integration (GPU kernel)
  Mapper/Color      - per-frame color integration
  Mapper/Mesh       - incremental mesh update
  Mapper/Query      - extracting the final mesh/ESDF

Supported datasets:
  - 3D Match (depth + color PNG pairs + pose txt) - default
  - Replica (depth + color PNG pairs + pose txt)
  Both follow the same directory layout used by nvblox's bundled examples.

Usage:
    python3 track_nvblox_profiled.py <dataset_path> [output.json]

    <dataset_path>  root of the scene (e.g. /data/3dmatch/sun3d-home_at-home_at_scan1_2013_jan_1)
                    Must contain: depth/ color/ pose/ (or depth.txt + rgb.txt + groundtruth.txt
                    in TUM format)

Example:
    python3 track_nvblox_profiled.py /data/3dmatch/scene /output/nvblox_profiler.json
"""

import argparse
import os
import sys
import glob
import time

import numpy as np

from py_profiler import enable, dump, block


# ---------------------------------------------------------------------------
# Dataset loader helpers
# ---------------------------------------------------------------------------

def _load_pose(path: str) -> np.ndarray:
    """Load a 4x4 pose matrix from a text file (row-major, space separated)."""
    return np.loadtxt(path).reshape(4, 4)


def _discover_frames_3dmatch(root: str):
    """
    Discover frames from a 3DMatch / Replica style dataset.
    Expected layout:
        root/depth/frame-XXXXXX.depth.png   (or .png)
        root/color/frame-XXXXXX.color.png   (or .jpg)
        root/pose/frame-XXXXXX.pose.txt
    Returns sorted list of (depth_path, color_path, pose_path).
    """
    depth_files = sorted(glob.glob(os.path.join(root, "depth", "*.png")))
    if not depth_files:
        depth_files = sorted(glob.glob(os.path.join(root, "depth", "*.pgm")))
    frames = []
    for d in depth_files:
        base = os.path.basename(d).replace(".depth.png", "").replace(".png", "").replace(".pgm", "")
        # Try common color extensions
        color = None
        for ext in [".color.png", ".color.jpg", ".png", ".jpg"]:
            candidate = os.path.join(root, "color", base + ext)
            if os.path.exists(candidate):
                color = candidate
                break
        pose_path = os.path.join(root, "pose", base + ".pose.txt")
        if not os.path.exists(pose_path):
            pose_path = os.path.join(root, "pose", base + ".txt")
        if color is not None and os.path.exists(pose_path):
            frames.append((d, color, pose_path))
    return frames


def _discover_frames_tum(root: str):
    """
    Discover frames from a TUM-format dataset (depth.txt, rgb.txt, groundtruth.txt).
    Returns list of (depth_path, color_path, pose_4x4).
    Poses are interpolated to nearest depth timestamp.
    """
    def _read_assoc(txt):
        entries = []
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                entries.append((float(parts[0]), parts[1]))
        return entries

    depth_txt = os.path.join(root, "depth.txt")
    rgb_txt = os.path.join(root, "rgb.txt")
    gt_txt = os.path.join(root, "groundtruth.txt")
    if not (os.path.exists(depth_txt) and os.path.exists(gt_txt)):
        return []

    depth_entries = _read_assoc(depth_txt)
    rgb_entries = _read_assoc(rgb_txt) if os.path.exists(rgb_txt) else []
    gt_raw = []
    with open(gt_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            gt_raw.append((ts, tx, ty, tz, qx, qy, qz, qw))

    def _quat_to_mat(tx, ty, tz, qx, qy, qz, qw):
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
        T[0, 0] = 1 - 2*(qy*qy + qz*qz)
        T[0, 1] = 2*(qx*qy - qz*qw)
        T[0, 2] = 2*(qx*qz + qy*qw)
        T[1, 0] = 2*(qx*qy + qz*qw)
        T[1, 1] = 1 - 2*(qx*qx + qz*qz)
        T[1, 2] = 2*(qy*qz - qx*qw)
        T[2, 0] = 2*(qx*qz - qy*qw)
        T[2, 1] = 2*(qy*qz + qx*qw)
        T[2, 2] = 1 - 2*(qx*qx + qy*qy)
        return T

    gt_ts = np.array([r[0] for r in gt_raw])

    def _nearest_pose(ts):
        idx = np.argmin(np.abs(gt_ts - ts))
        r = gt_raw[idx]
        return _quat_to_mat(r[1], r[2], r[3], r[4], r[5], r[6], r[7])

    rgb_ts = np.array([e[0] for e in rgb_entries]) if rgb_entries else None
    frames = []
    for d_ts, d_rel in depth_entries:
        d_path = os.path.join(root, d_rel)
        # Nearest color
        color_path = None
        if rgb_ts is not None and len(rgb_ts) > 0:
            idx = np.argmin(np.abs(rgb_ts - d_ts))
            color_path = os.path.join(root, rgb_entries[idx][1])
        pose = _nearest_pose(d_ts)
        frames.append((d_path, color_path, pose))
    return frames


# ---------------------------------------------------------------------------
# nvblox integration shim
# ---------------------------------------------------------------------------

def _try_import_nvblox():
    """Try importing nvblox Python bindings; return module or None."""
    try:
        import nvblox  # noqa: F401
        return nvblox
    except ImportError:
        return None


def _run_with_python_bindings(nvblox, frames, voxel_size, output_path):
    """Profile nvblox via Python bindings."""
    enable()

    with block("nvblox/Init"):
        mapper = nvblox.Mapper(voxel_size_m=voxel_size, projective_layer_type="tsdf")

    print(f"Processing {len(frames)} frames with nvblox Python bindings...")
    for i, frame in enumerate(frames):
        depth_path, color_path, pose = frame
        if isinstance(pose, str):
            pose = _load_pose(pose)

        with block("nvblox/FrameProcess"):
            with block("nvblox/LoadDepth"):
                depth_img = nvblox.DepthImage.fromFile(depth_path)

            with block("nvblox/Integrate"):
                mapper.integrateDepth(depth_img, pose)

            if color_path is not None and os.path.exists(str(color_path)):
                with block("nvblox/IntegrateColor"):
                    color_img = nvblox.ColorImage.fromFile(color_path)
                    mapper.integrateColor(color_img, pose)

            # Trigger mesh update every 10 frames to match typical usage
            if i % 10 == 0:
                with block("nvblox/UpdateMesh"):
                    mapper.updateMesh()

        if i % 100 == 0:
            print(f"  Frame {i}/{len(frames)}")

    with block("nvblox/ExtractMesh"):
        mesh = mapper.extractMesh()

    print(f"Mesh extracted: {mesh}")
    dump(output_path)
    return mesh


def _run_subprocess_fallback(binary_path, dataset_path, voxel_size, output_path):
    """
    Fallback: run the nvblox fuse_3dmatch or fuse_replica binary via subprocess
    and time its wall-clock duration as a single profiler block.
    """
    import subprocess
    enable()

    cmd = [binary_path, dataset_path, "--voxel_size", str(voxel_size)]
    print(f"Running nvblox binary: {' '.join(cmd)}")

    with block("nvblox/FullRun"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout[-2000:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        print(f"[WARN] nvblox exited with code {result.returncode}", file=sys.stderr)
        print(result.stderr[-1000:], file=sys.stderr)

    dump(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile nvblox 3D reconstruction pipeline"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="/data",
        help="Path to dataset root (3DMatch layout or TUM format)",
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        default="/output/nvblox_profiler.json",
        help="Output profiler JSON path",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="TSDF voxel size in metres (default: 0.05)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of frames (0 = all)",
    )
    parser.add_argument(
        "--binary",
        default="/usr/local/bin/fuse_3dmatch",
        help="Path to nvblox C++ binary (used if Python bindings unavailable)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_json
    voxel_size = args.voxel_size

    print(f"Dataset:    {dataset_path}")
    print(f"Output:     {output_path}")
    print(f"Voxel size: {voxel_size} m")

    nvblox = _try_import_nvblox()

    if nvblox is not None:
        # Discover frames
        frames = _discover_frames_3dmatch(dataset_path)
        if not frames:
            frames = _discover_frames_tum(dataset_path)
        if not frames:
            print("[ERROR] No frames discovered. Check dataset layout.", file=sys.stderr)
            sys.exit(1)

        if args.max_frames > 0:
            frames = frames[: args.max_frames]

        print(f"Discovered {len(frames)} frames.")
        _run_with_python_bindings(nvblox, frames, voxel_size, output_path)

    elif os.path.exists(args.binary):
        print(f"nvblox Python bindings not available. Using binary: {args.binary}")
        _run_subprocess_fallback(args.binary, dataset_path, voxel_size, output_path)

    else:
        print(
            "[ERROR] Neither nvblox Python bindings nor C++ binary found.\n"
            "  Install bindings:  pip3 install nvblox\n"
            "  Or build from source and ensure fuse_3dmatch is on PATH.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
