#!/usr/bin/env python3
"""
parse_nvblox_timing.py - Convert nvblox's built-in Timing report into the
shared analyze_profiler CSV schema.

nvblox uses its own nvblox::timing::Timer class (not easy_profiler). When
fuse_replica / fuse_3dmatch are run with --timing_output_path, they dump a
plaintext file like:

    NVBlox Timings (in seconds)
    namespace/tag - NumSamples - TotalTime - (Mean +- StdDev) - [Min,Max]
    -----------
    tsdf/integrate                500  00.259960  (00.000520 +- 00.000532)  [00.000089,00.145761]
    ...

This script parses that file and emits a CSV with the same columns as
analyze_profiler.py's --export csv output so the two can be compared
side-by-side. nvblox does not report median/p95/p99 (its Timer only tracks
running mean/stddev/min/max), so those columns are left empty.

All time values are converted from seconds to milliseconds to match the
easy_profiler side.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


LINE_RE = re.compile(
    r"^(?P<name>\S+)\s+"
    r"(?P<n>\d+)\s+"
    r"(?P<total>[\d.]+)\s+"
    r"\(\s*(?P<mean>[\d.]+)\s*\+-\s*(?P<std>[\d.]+)\s*\)\s+"
    r"\[\s*(?P<min>[\d.]+)\s*,\s*(?P<max>[\d.]+)\s*\]"
)


def parse(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        rows.append({
            "block": "SLAM/" + name,
            "count": int(m.group("n")),
            "total_ms": float(m.group("total")) * 1000.0,
            "mean_ms": float(m.group("mean")) * 1000.0,
            "std_ms": float(m.group("std")) * 1000.0,
            "min_ms": float(m.group("min")) * 1000.0,
            "max_ms": float(m.group("max")) * 1000.0,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("timing_file", type=Path)
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--platform", default="desktop")
    ap.add_argument("--algorithm", default="nvblox")
    args = ap.parse_args()

    rows = parse(args.timing_file)
    if not rows:
        print(f"[error] no timing lines parsed from {args.timing_file}", file=sys.stderr)
        return 1

    headers = [
        "filename", "platform", "algorithm", "block", "count",
        "mean_ms", "median_ms", "min_ms", "max_ms", "std_ms",
        "p95_ms", "p99_ms", "total_ms",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["block"]):
            w.writerow({
                "filename": str(args.timing_file),
                "platform": args.platform,
                "algorithm": args.algorithm,
                "block": r["block"],
                "count": r["count"],
                "mean_ms": r["mean_ms"],
                "median_ms": "",
                "min_ms": r["min_ms"],
                "max_ms": r["max_ms"],
                "std_ms": r["std_ms"],
                "p95_ms": "",
                "p99_ms": "",
                "total_ms": r["total_ms"],
            })
    print(f"Exported CSV -> {args.output} ({len(rows)} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
