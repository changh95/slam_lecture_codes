#!/usr/bin/env python3
"""
run_profiled.py - nvblox per-component profiler.

Runs fuse_replica with --timing_output_path and converts nvblox's native
timing::Timer report into py_profiler JSON (compatible with
analyze_profiler.py and the other SLAM system outputs).

nvblox's timing output format (tab-separated):
    namespace/tag  NumSamples  TotalTime  (Mean +- StdDev)  [Min,Max]

All units are seconds in the source; we convert to nanoseconds and emit
one block per tag under a top-level SLAM/FullRun block.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


FUSE_REPLICA = "/usr/local/bin/nvblox/fuse_replica"
DEFAULT_OUTPUT = "/output/nvblox.json"


def parse_nvblox_timing(timing_txt):
    """Parse nvblox's timing dump into a list of (tag, count, total_s, mean_s)."""
    entries = []
    lines = timing_txt.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("-") or line.lower().startswith("nvblox") \
           or line.startswith("namespace"):
            continue
        # Tags can contain slashes and underscores; fields are separated by
        # runs of whitespace/tabs.
        m = re.match(
            r"^(\S+)\s+(\d+)\s+([\d.]+)\s+\(([\d.]+)\s*\+\-\s*([\d.]+)\)\s+\[([\d.]+),([\d.]+)\]",
            line,
        )
        if not m:
            continue
        tag = m.group(1)
        count = int(m.group(2))
        total_s = float(m.group(3))
        mean_s = float(m.group(4))
        min_s = float(m.group(6))
        max_s = float(m.group(7))
        entries.append({
            "tag": tag,
            "count": count,
            "total_s": total_s,
            "mean_s": mean_s,
            "min_s": min_s,
            "max_s": max_s,
        })
    return entries


def entries_to_profiler_json(entries, full_run_s):
    """Convert parsed entries to py_profiler JSON format."""
    now_ns = int(time.time() * 1e9)
    children = []
    cursor = now_ns
    for e in entries:
        dur_ns = int(e["total_s"] * 1e9)
        child = {
            "id": len(children),
            "name": f"SLAM/{e['tag']}",
            "start": cursor,
            "stop": cursor + dur_ns,
            "descriptor": 0,
            "children": [],
        }
        children.append(child)
        cursor += dur_ns

    full_run = {
        "id": len(children),
        "name": "SLAM/FullRun",
        "start": now_ns,
        "stop": now_ns + int(full_run_s * 1e9),
        "descriptor": 0,
        "children": children,
    }
    return {
        "version": "py_profiler-1.0",
        "timeUnits": "ns",
        "blockDescriptors": [],
        "threads": [{
            "threadId": 0,
            "threadName": "MainThread",
            "children": [full_run],
        }],
    }


def main():
    parser = argparse.ArgumentParser(description="nvblox per-component profiler")
    parser.add_argument("dataset_path", help="Replica sequence dir (e.g. /data/office0)")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="-1 = all frames")
    parser.add_argument("--mesh-output", default="/output/nvblox_mesh.ply")
    args = parser.parse_args()

    timing_file = "/tmp/nvblox_timing.txt"

    cmd = [
        FUSE_REPLICA, args.dataset_path,
        "--voxel_size", str(args.voxel_size),
        "--num_frames", str(args.num_frames),
        "--mesh_output_path", args.mesh_output,
        "--timing_output_path", timing_file,
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - start

    print(f"Exit code: {result.returncode}")
    print(f"Wall time: {wall:.1f} s")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:], file=sys.stderr)
        sys.exit(result.returncode)

    if not os.path.exists(timing_file):
        print(f"[ERROR] Timing file not produced at {timing_file}", file=sys.stderr)
        sys.exit(1)

    with open(timing_file) as f:
        timing_txt = f.read()

    entries = parse_nvblox_timing(timing_txt)
    print(f"Parsed {len(entries)} timing entries")
    for e in entries[:5]:
        print(f"  {e['tag']}: {e['count']}x total={e['total_s']:.3f}s "
              f"mean={e['mean_s']*1000:.2f}ms")

    out_json = entries_to_profiler_json(entries, wall)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_json, f)
    print(f"\nProfiler data saved to {args.output}")


if __name__ == "__main__":
    main()
