#!/usr/bin/env python3
"""
run_profiled.py - MASt3R-SLAM profiled runner with per-component instrumentation.

Patches main.py and tracker.py with py_profiler blocks, then runs SLAM.
The child process (main.py) handles profiler enable/dump directly.

Profiles: FullRun, ImageLoad, FrameCreation, MonoInference, Tracking
(FeatureMatching, PoseEstimation, PointmapUpdate), KeyframeInsert.
"""

import argparse
import os
import subprocess
import sys
import time


MAST3R_ROOT = "/MASt3R-SLAM"
DEFAULT_CONFIG = os.path.join(MAST3R_ROOT, "config", "base.yaml")
DEFAULT_OUTPUT = "/output/mast3r_slam_desktop.json"


def main():
    parser = argparse.ArgumentParser(description="MASt3R-SLAM profiled runner")
    parser.add_argument("dataset_path", help="Path to TUM sequence / image folder")
    parser.add_argument(
        "output",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
    )
    parser.add_argument("--calib", default=None)
    parser.add_argument("--save-as", default="profiler_run")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"[ERROR] Dataset not found: {args.dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset : {args.dataset_path}")
    print(f"Config  : {args.config}")
    print(f"Output  : {args.output}")

    # Apply profiler instrumentation patches
    print("\nApplying profiler patches...")
    result = subprocess.run(
        ["python", "/profiling/profiler_patch.py"],
        cwd=MAST3R_ROOT,
    )
    if result.returncode != 0:
        print("[ERROR] Failed to apply profiler patches", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "python", "-u", "main.py",
        "--dataset", args.dataset_path,
        "--config", args.config,
        "--no-viz",
        "--save-as", args.save_as,
    ]
    if args.calib:
        cmd.extend(["--calib", args.calib])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PROFILER_OUTPUT"] = args.output

    print(f"\nRunning: {' '.join(cmd)}\n")

    start_wall = time.time()
    result = subprocess.run(cmd, cwd=MAST3R_ROOT, env=env)
    wall_sec = time.time() - start_wall

    print(f"\nExit code : {result.returncode}")
    print(f"Wall time : {wall_sec:.1f} s")

    # Merge frontend + backend profiler data
    backend_output = args.output.replace('.json', '_backend.json')
    if os.path.exists(args.output) and os.path.exists(backend_output):
        import json
        with open(args.output) as f:
            frontend = json.load(f)
        with open(backend_output) as f:
            backend = json.load(f)
        # Add backend threads to frontend data
        for bt in backend.get('threads', []):
            bt['threadName'] = 'BackendThread'
            frontend['threads'].append(bt)
        with open(args.output, 'w') as f:
            json.dump(frontend, f)
        os.remove(backend_output)
        print(f"Merged profiler data: {args.output}")
    elif os.path.exists(args.output):
        print(f"Profiler data: {args.output} (frontend only)")
    else:
        print("[WARNING] No profiler output generated")


if __name__ == "__main__":
    main()
