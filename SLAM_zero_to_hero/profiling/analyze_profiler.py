#!/usr/bin/env python3
"""
analyze_profiler.py - Analyze easy_profiler JSON output for SLAM algorithms.

Reads JSON files produced by easy_profiler_converter (from .prof binary files)
and provides statistics, cross-platform comparisons, and visualizations.

Usage:
    python3 analyze_profiler.py <file1.json> [file2.json ...]
    python3 analyze_profiler.py --compare desktop.json jetson.json rpi.json
    python3 analyze_profiler.py --export csv --output results.csv *.json
    python3 analyze_profiler.py --plot comparison_chart.png file1.json file2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

NS_TO_MS = 1e-6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BlockStats:
    """Statistics for a single named profiler block."""
    count: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float

    @classmethod
    def from_durations_ns(cls, durations_ns: List[int]) -> "BlockStats":
        durations_ms = [d * NS_TO_MS for d in durations_ns]
        n = len(durations_ms)
        sorted_ms = sorted(durations_ms)

        def percentile(data: List[float], pct: float) -> float:
            if not data:
                return 0.0
            idx = (len(data) - 1) * pct / 100.0
            lo = int(idx)
            hi = min(lo + 1, len(data) - 1)
            frac = idx - lo
            return data[lo] + frac * (data[hi] - data[lo])

        std = statistics.stdev(durations_ms) if n > 1 else 0.0

        return cls(
            count=n,
            mean_ms=statistics.mean(durations_ms),
            median_ms=statistics.median(durations_ms),
            min_ms=min(durations_ms),
            max_ms=max(durations_ms),
            std_ms=std,
            p95_ms=percentile(sorted_ms, 95),
            p99_ms=percentile(sorted_ms, 99),
            total_ms=sum(durations_ms),
        )


@dataclass
class ProfileResult:
    """Parsed result from a single profiler JSON file."""
    filename: str
    platform: str
    algorithm: str
    blocks: Dict[str, BlockStats] = field(default_factory=dict)

    @classmethod
    def from_filename(cls, path: str) -> "ProfileResult":
        stem = Path(path).stem  # e.g. fast_lio2_desktop
        # Try to extract algorithm and platform by splitting on last underscore
        # Heuristic: known platforms
        known_platforms = [
            "desktop", "jetson_thor", "dgx_spark", "jetson_orin_nano",
            "jetson_orin", "rpi", "raspberrypi",
        ]
        stem_lower = stem.lower()
        platform = "unknown"
        algorithm = stem

        for plat in sorted(known_platforms, key=len, reverse=True):
            if stem_lower.endswith("_" + plat):
                platform = plat
                algorithm = stem[: -(len(plat) + 1)]
                break
        else:
            # Fallback: last token is platform
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                algorithm, platform = parts

        return cls(filename=path, platform=platform, algorithm=algorithm)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def collect_blocks(node: dict, accumulator: Dict[str, List[int]]) -> None:
    """Recursively walk the profiler tree and collect SLAM/* block durations."""
    name = node.get("name", "")
    if name.startswith("SLAM/"):
        duration_ns = node.get("stop", 0) - node.get("start", 0)
        if duration_ns > 0:
            accumulator.setdefault(name, []).append(duration_ns)

    for child in node.get("children", []):
        collect_blocks(child, accumulator)


def parse_json(path: str) -> ProfileResult:
    """Parse an easy_profiler JSON file and return a ProfileResult."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    accumulator: Dict[str, List[int]] = {}

    threads = data.get("threads", [])
    for thread in threads:
        for child in thread.get("children", []):
            collect_blocks(child, accumulator)

    result = ProfileResult.from_filename(path)
    for block_name, durations in accumulator.items():
        result.blocks[block_name] = BlockStats.from_durations_ns(durations)

    return result


# ---------------------------------------------------------------------------
# Auto-conversion from .prof
# ---------------------------------------------------------------------------

def try_convert_prof(prof_path: str) -> Optional[str]:
    """
    Attempt to convert a .prof binary file to JSON using easy_profiler_converter.
    Returns the JSON path on success, None if converter is unavailable.
    """
    json_path = Path(prof_path).with_suffix(".json")
    try:
        subprocess.run(
            ["easy_profiler_converter", str(prof_path), str(json_path)],
            check=True,
            capture_output=True,
        )
        print(f"[auto-convert] {prof_path} -> {json_path}")
        return str(json_path)
    except FileNotFoundError:
        print(
            f"[error] easy_profiler_converter not found. "
            f"Please convert '{prof_path}' to JSON manually:\n"
            f"  easy_profiler_converter {prof_path} {json_path}\n"
            f"Or run: ./convert_prof.sh {prof_path}"
        )
        return None
    except subprocess.CalledProcessError as exc:
        print(f"[error] Conversion failed for '{prof_path}': {exc.stderr.decode()}")
        return None


def resolve_input(path: str) -> Optional[str]:
    """Resolve a path, auto-converting .prof files if needed."""
    if path.endswith(".prof"):
        return try_convert_prof(path)
    if not os.path.isfile(path):
        print(f"[error] File not found: {path}")
        return None
    return path


# ---------------------------------------------------------------------------
# Terminal output (rich with tabulate fallback)
# ---------------------------------------------------------------------------

def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _tabulate_available() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def print_single_stats(result: ProfileResult) -> None:
    """Print a statistics table for a single profile result."""
    if not result.blocks:
        print(f"[warn] No SLAM/* blocks found in {result.filename}")
        return

    headers = [
        "Block Name", "Count", "Mean(ms)", "Median(ms)",
        "Min(ms)", "Max(ms)", "Std(ms)", "P95(ms)", "P99(ms)", "Total(ms)",
    ]

    rows = []
    for name, s in sorted(result.blocks.items()):
        rows.append([
            name,
            s.count,
            f"{s.mean_ms:.3f}",
            f"{s.median_ms:.3f}",
            f"{s.min_ms:.3f}",
            f"{s.max_ms:.3f}",
            f"{s.std_ms:.3f}",
            f"{s.p95_ms:.3f}",
            f"{s.p99_ms:.3f}",
            f"{s.total_ms:.3f}",
        ])

    title = f"Profile: {result.algorithm} | Platform: {result.platform} | File: {result.filename}"

    if _rich_available():
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, show_lines=True)
        for h in headers:
            table.add_column(h, justify="right" if h != "Block Name" else "left")
        for row in rows:
            table.add_row(*[str(c) for c in row])
        console.print(table)
    elif _tabulate_available():
        from tabulate import tabulate
        print(f"\n{title}")
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(f"\n{title}")
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(c) for c in row))


def print_summary(result: ProfileResult) -> None:
    """Print key insights for a single profile result."""
    if not result.blocks:
        return

    print_single_stats(result)

    # Key insights
    by_mean = sorted(result.blocks.items(), key=lambda kv: kv[1].mean_ms)
    slowest_name, slowest = by_mean[-1]
    fastest_name, fastest = by_mean[0]

    # Estimate FPS from the top-level block (shortest prefix = SLAM/<algo>)
    top_blocks = [
        (n, s) for n, s in result.blocks.items()
        if n.count("/") == 1
    ]
    total_ms = None
    fps = None
    if top_blocks:
        # Use block with most accumulated time as frame proxy
        top_name, top_stats = max(top_blocks, key=lambda kv: kv[1].total_ms)
        total_ms = top_stats.mean_ms
        fps = 1000.0 / total_ms if total_ms > 0 else None

    insights = [
        f"  Slowest block : {slowest_name} ({slowest.mean_ms:.3f} ms mean)",
        f"  Fastest block : {fastest_name} ({fastest.mean_ms:.3f} ms mean)",
    ]
    if total_ms is not None:
        insights.append(f"  Frame time est: {total_ms:.3f} ms ({fps:.1f} FPS)")

    if _rich_available():
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel("\n".join(insights), title="[bold]Key Insights[/bold]", expand=False))
    else:
        print("\nKey Insights:")
        for line in insights:
            print(line)


def print_comparison(results: List[ProfileResult]) -> None:
    """Print a side-by-side comparison table across multiple platform profiles."""
    if len(results) < 2:
        print("[warn] Need at least 2 files for comparison.")
        return

    reference = results[0]
    all_blocks = set()
    for r in results:
        all_blocks.update(r.blocks.keys())
    all_blocks_sorted = sorted(all_blocks)

    # Build header
    headers = ["Block Name"]
    for r in results:
        headers.append(f"{r.platform}\nMean(ms)")
    for r in results[1:]:
        headers.append(f"vs {reference.platform}\nSpeedup")

    rows = []
    for block_name in all_blocks_sorted:
        row = [block_name]
        means = []
        for r in results:
            s = r.blocks.get(block_name)
            if s:
                row.append(f"{s.mean_ms:.3f}")
                means.append(s.mean_ms)
            else:
                row.append("N/A")
                means.append(None)
        ref_mean = means[0]
        for m in means[1:]:
            if ref_mean and m:
                speedup = ref_mean / m
                row.append(f"{speedup:.2f}x")
            else:
                row.append("N/A")
        rows.append(row)

    title = "Cross-Platform Comparison (speedup > 1.0 = faster than reference)"

    if _rich_available():
        from rich.console import Console
        from rich.table import Table

        console = Console()
        # Flatten headers (replace newline)
        flat_headers = [h.replace("\n", " ") for h in headers]
        table = Table(title=title, show_lines=True)
        for h in flat_headers:
            table.add_column(h, justify="right" if h != "Block Name" else "left")
        for row in rows:
            table.add_row(*[str(c) for c in row])
        console.print(table)
    elif _tabulate_available():
        from tabulate import tabulate
        flat_headers = [h.replace("\n", " ") for h in headers]
        print(f"\n{title}")
        print(tabulate(rows, headers=flat_headers, tablefmt="grid"))
    else:
        print(f"\n{title}")
        flat_headers = [h.replace("\n", " ") for h in headers]
        print("\t".join(flat_headers))
        for row in rows:
            print("\t".join(str(c) for c in row))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(results: List[ProfileResult], output_path: str) -> None:
    """Export all results to a CSV file."""
    rows = []
    for r in results:
        for block_name, s in sorted(r.blocks.items()):
            rows.append({
                "filename": r.filename,
                "platform": r.platform,
                "algorithm": r.algorithm,
                "block": block_name,
                "count": s.count,
                "mean_ms": s.mean_ms,
                "median_ms": s.median_ms,
                "min_ms": s.min_ms,
                "max_ms": s.max_ms,
                "std_ms": s.std_ms,
                "p95_ms": s.p95_ms,
                "p99_ms": s.p99_ms,
                "total_ms": s.total_ms,
            })

    if not rows:
        print("[warn] No data to export.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported CSV -> {output_path}")


def export_json_file(results: List[ProfileResult], output_path: str) -> None:
    """Export all results to a JSON file."""
    output = []
    for r in results:
        entry = {
            "filename": r.filename,
            "platform": r.platform,
            "algorithm": r.algorithm,
            "blocks": {
                name: asdict(stats) for name, stats in r.blocks.items()
            },
        }
        output.append(entry)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Exported JSON -> {output_path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(results: List[ProfileResult], output_path: str) -> None:
    """Generate a bar chart comparing mean execution times across platforms."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[error] matplotlib/numpy not available. Install with: pip install matplotlib numpy")
        return

    if not results:
        return

    # Collect all block names present in all results
    all_blocks = set()
    for r in results:
        all_blocks.update(r.blocks.keys())
    all_blocks_sorted = sorted(all_blocks)

    if not all_blocks_sorted:
        print("[warn] No SLAM/* blocks found for plotting.")
        return

    n_blocks = len(all_blocks_sorted)
    n_platforms = len(results)
    x = np.arange(n_blocks)
    bar_width = 0.8 / n_platforms

    fig, ax = plt.subplots(figsize=(max(12, n_blocks * 1.5), 6))

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, r in enumerate(results):
        means = [
            r.blocks[b].mean_ms if b in r.blocks else 0.0
            for b in all_blocks_sorted
        ]
        offset = (i - n_platforms / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, label=r.platform, color=colors[i % len(colors)])

    ax.set_xlabel("Block Name")
    ax.set_ylabel("Mean Execution Time (ms)")
    ax.set_title("SLAM Block Execution Time Comparison Across Platforms")
    ax.set_xticks(x)
    ax.set_xticklabels(all_blocks_sorted, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze easy_profiler JSON output for SLAM algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="JSON (or .prof) profiler files to analyze.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show side-by-side comparison table. First file is the reference.",
    )
    parser.add_argument(
        "--export",
        choices=["csv", "json"],
        metavar="FORMAT",
        help="Export stats to CSV or JSON (use with --output).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Output file path for --export.",
    )
    parser.add_argument(
        "--plot",
        metavar="IMAGE_PATH",
        help="Generate a bar-chart comparison and save to IMAGE_PATH (PNG/PDF).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve inputs (auto-convert .prof if needed)
    resolved_paths = []
    for f in args.files:
        p = resolve_input(f)
        if p:
            resolved_paths.append(p)

    if not resolved_paths:
        print("[error] No valid input files.")
        return 1

    # Parse all files
    results: List[ProfileResult] = []
    for path in resolved_paths:
        try:
            result = parse_json(path)
            results.append(result)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"[error] Failed to parse '{path}': {exc}")

    if not results:
        print("[error] No results parsed.")
        return 1

    # Export mode
    if args.export:
        output_path = args.output
        if not output_path:
            ext = args.export
            output_path = f"profiler_results.{ext}"
        if args.export == "csv":
            export_csv(results, output_path)
        else:
            export_json_file(results, output_path)
        return 0

    # Plot mode
    if args.plot:
        plot_comparison(results, args.plot)
        if not args.compare and len(results) == 1:
            return 0

    # Comparison mode
    if args.compare or len(results) > 1:
        print_comparison(results)
    else:
        # Single file summary (default)
        print_summary(results[0])

    return 0


if __name__ == "__main__":
    sys.exit(main())
