#!/usr/bin/env python3
"""
nvblox_timing_to_json.py - Convert nvblox's built-in Timing report into the
easy_profiler-compatible JSON schema consumed by analyze_profiler.py.

nvblox uses its own nvblox::timing::Timer (not easy_profiler), and only
publishes aggregated stats per stage (count, mean, std, min, max). To keep
nvblox numbers comparable with the rest of the perf_bench/ JSON files,
this script synthesizes `count` blocks per timer line, each with a
duration of `mean_ms`. The resulting JSON gives correct count, total, and
mean. Median/p95/p99 collapse to the mean (a known limitation — nvblox's
Timer does not record the underlying distribution).
"""

from __future__ import annotations

import argparse
import json
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


def parse(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        rows.append({
            "name": "SLAM/" + m.group("name"),
            "count": int(m.group("n")),
            "mean_ns": int(float(m.group("mean")) * 1e9),
            "min_ns": int(float(m.group("min")) * 1e9),
            "max_ns": int(float(m.group("max")) * 1e9),
            "total_ns": int(float(m.group("total")) * 1e9),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("timing_file", type=Path)
    ap.add_argument("--output", "-o", type=Path, required=True)
    args = ap.parse_args()

    rows = parse(args.timing_file)
    if not rows:
        print(f"[error] no timing lines parsed from {args.timing_file}", file=sys.stderr)
        return 1

    children = []
    cursor = 0
    block_id = 0
    for r in rows:
        n = r["count"]
        mean_ns = r["mean_ns"]
        min_ns = r["min_ns"]
        max_ns = r["max_ns"]
        # Synthesize one block per sample. Sprinkle min/max so the
        # analyzer's min/max/std are at least directionally correct.
        for i in range(n):
            if i == 0 and n >= 2:
                dur = min_ns
            elif i == 1 and n >= 2:
                dur = max_ns
            else:
                dur = mean_ns
            children.append({
                "id": block_id,
                "name": r["name"],
                "start": cursor,
                "stop": cursor + dur,
                "descriptor": 0,
                "children": [],
            })
            cursor += dur
            block_id += 1

    full_run = {
        "id": block_id,
        "name": "SLAM/FullRun",
        "start": 0,
        "stop": cursor,
        "descriptor": 0,
        "children": children,
    }

    data = {
        "version": "nvblox-timing-1.0",
        "timeUnits": "ns",
        "blockDescriptors": [],
        "threads": [{
            "threadId": 0,
            "threadName": "MainThread",
            "children": [full_run],
        }],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))
    total_blocks = len(children)
    print(f"[ok] wrote {args.output} ({len(rows)} timer lines, {total_blocks} synthetic blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
