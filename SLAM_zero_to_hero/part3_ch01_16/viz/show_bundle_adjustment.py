#!/usr/bin/env python3
"""Visualize a BAL bundle-adjustment *optimization* in 3D with rerun.

Reads the multi-step ``bundle_adjustment.txt`` dump (one block of landmarks +
camera centers per solver iteration) and logs every iteration under an
``iteration`` timeline, so you can scrub through the optimization and watch the
landmarks converge. The initial cloud is also logged once as a static grey
reference.

By default it writes a self-contained ``bundle_adjustment.rrd`` (open later with
``rerun bundle_adjustment.rrd``). Use ``--spawn`` to open a live viewer, or
``--connect [URL]`` to stream to an already-running viewer (default
``rerun+http://127.0.0.1:9876/proxy``).

Data format::

    points <Np>
    cameras <Nc>
    steps <K>
    step 0 <total_reprojection_error>
    <x y z>   x Np      # landmarks at iteration 0 (initial)
    <x y z>   x Nc      # camera centers at iteration 0
    step 1 <total_reprojection_error>
    ...                 # K blocks total
"""
import argparse
import sys

import numpy as np
import rerun as rr


def parse(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    Np = Nc = None
    idx = 0
    while idx < len(lines) and lines[idx].split()[0] != "step":
        tok = lines[idx].split()
        if tok[0] == "points":
            Np = int(tok[1])
        elif tok[0] == "cameras":
            Nc = int(tok[1])
        idx += 1
    steps = []
    while idx < len(lines):
        tok = lines[idx].split()
        assert tok[0] == "step", lines[idx]
        err = float(tok[2]) if len(tok) > 2 else None  # total reprojection error
        idx += 1
        pts = np.array([[float(v) for v in lines[idx + i].split()] for i in range(Np)])
        idx += Np
        cams = np.array([[float(v) for v in lines[idx + i].split()] for i in range(Nc)])
        idx += Nc
        steps.append((pts, cams, err))
    return Np, Nc, steps


def log_scalar(path, value):
    """Log one scalar sample (rerun renamed Scalar -> Scalars across versions)."""
    if hasattr(rr, "Scalars"):
        rr.log(path, rr.Scalars(value))
    else:
        rr.log(path, rr.Scalar(value))


def set_iter(k):
    """Set the 'iteration' timeline value (works across rerun 0.2x/0.3x)."""
    if hasattr(rr, "set_time"):
        try:
            rr.set_time("iteration", sequence=k)
            return
        except TypeError:
            pass
    rr.set_time_sequence("iteration", k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="bundle_adjustment.txt")
    ap.add_argument("-o", "--out", default="bundle_adjustment.rrd")
    ap.add_argument("--spawn", action="store_true",
                    help="open a live viewer instead of saving a .rrd file")
    ap.add_argument("--connect", nargs="?", const="rerun+http://127.0.0.1:9876/proxy",
                    default=None, metavar="URL",
                    help="stream to an already-running rerun viewer at this gRPC "
                         "URL (default rerun+http://127.0.0.1:9876/proxy)")
    ap.add_argument("--name", default="part3_bundle_adjustment",
                    help="rerun application/recording name (shown in the viewer)")
    args = ap.parse_args()

    try:
        Np, Nc, steps = parse(args.data)
    except FileNotFoundError:
        sys.exit(f"[show_bundle_adjustment] cannot open {args.data}; run the "
                 f"bundle_adjustment example first to generate it.")

    rr.init(args.name, spawn=args.spawn)
    if args.connect:
        rr.connect_grpc(args.connect)
    elif not args.spawn:
        rr.save(args.out)

    # Static grey reference: the initial landmark cloud, shown at every step.
    rr.log("world/initial_points",
           rr.Points3D(steps[0][0], colors=[120, 120, 120], radii=0.012),
           static=True)

    # Per-iteration landmarks (green), camera centers (blue), and the total
    # reprojection error as a scalar time series (shown as a graph in rerun).
    # All landmarks are logged (no filtering).
    for k, (pts, cams, err) in enumerate(steps):
        set_iter(k)
        rr.log("world/landmarks",
               rr.Points3D(pts, colors=[80, 200, 120], radii=0.02))
        rr.log("world/cameras", rr.Points3D(cams, colors=[0, 120, 255], radii=0.12))
        if err is not None:
            log_scalar("reprojection_error", err)

    if args.connect:
        try:
            rr.flush()
        except Exception:
            import time
            time.sleep(2.0)
        print(f"[show_bundle_adjustment] streamed {len(steps)} iterations to "
              f"viewer at {args.connect}")
    elif args.spawn:
        print(f"[show_bundle_adjustment] streaming {len(steps)} iterations to live viewer")
    else:
        print(f"[show_bundle_adjustment] wrote {args.out} with {len(steps)} "
              f"iterations (open with: rerun {args.out}; scrub the 'iteration' timeline)")


if __name__ == "__main__":
    main()
