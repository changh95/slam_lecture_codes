#!/usr/bin/env python3
"""
track_rerun.py - MASt3R-SLAM with real-time Rerun visualization.

Patches main.py with Rerun hooks, then runs SLAM with live streaming
to the Rerun web viewer.

Usage:
    python3 track_rerun.py <dataset_path> --web [--config cfg.yaml]
"""

import argparse
import os
import subprocess
import sys


MAST3R_ROOT = "/MASt3R-SLAM"


def main():
    parser = argparse.ArgumentParser(description="MASt3R-SLAM Rerun visualization")
    parser.add_argument("dataset_path", help="TUM sequence directory")
    parser.add_argument("--web", action="store_true", help="Serve web viewer at :9090")
    parser.add_argument("--rrd", help="Save to .rrd file")
    parser.add_argument("--config", default=os.path.join(MAST3R_ROOT, "config", "base.yaml"))
    parser.add_argument("--save-as", default="rerun_viz")
    args = parser.parse_args()

    if not args.web and not args.rrd:
        print("ERROR: specify --web or --rrd")
        sys.exit(1)

    # Apply rerun patch to main.py
    print("Applying Rerun patch to main.py...")
    result = subprocess.run(["python3", "/rerun_viz/rerun_patch.py"], cwd=MAST3R_ROOT)
    if result.returncode != 0:
        print("Failed to apply Rerun patch", file=sys.stderr)
        sys.exit(1)

    # Set env vars for rerun_patch to pick up
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.web:
        env["RERUN_WEB"] = "1"
    if args.rrd:
        env["RERUN_RRD"] = args.rrd

    cmd = [
        "python3", "main.py",
        "--dataset", args.dataset_path,
        "--config", args.config,
        "--no-viz",
        "--save-as", args.save_as,
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Open http://localhost:9090 to see live visualization\n")

    proc = subprocess.run(cmd, cwd=MAST3R_ROOT, env=env)

    if proc.returncode != 0:
        print(f"SLAM exited with code {proc.returncode}", file=sys.stderr)

    if args.web:
        print("\nSLAM complete. Web viewer still running at http://localhost:9090")
        print("Press Ctrl+C to exit.")
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
