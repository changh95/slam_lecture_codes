#!/usr/bin/env python3
"""
Send the viewer layout (blueprint) for the live-streaming PnP demos.

The rerun C++ SDK cannot author blueprints, so the demos stream data only and
this script pushes the layout to the running viewer: the 3D world and the
camera view side by side on top, and one consolidated time-series plot per
metric (all PnP methods in the same plot) below.

Run it once before (or after) the demos - the layout applies to every
recording with the demos' application id:

    python3 blueprint.py                       # viewer on this machine
    python3 blueprint.py --connect-url rerun+http://HOST:PORT/proxy
    python3 blueprint.py --save layout.rbl     # write a blueprint file instead
"""

import argparse
import socket
import sys
from urllib.parse import urlparse

import rerun as rr
import rerun.blueprint as rrb

APP_ID = "apriltag_pnp"  # must match the C++ demos
DEFAULT_CONNECT_URL = "rerun+http://127.0.0.1:9876/proxy"


def make_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="World (tag 10 frame)"),
                rrb.Spatial2DView(origin="world/camera/image",
                                  name="Camera + detections"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="plots/reproj_err_px",
                                   name="Reprojection error (px)"),
                rrb.TimeSeriesView(origin="plots/solve_ms",
                                   name="PnP solve time (ms)"),
            ),
            row_shares=[3, 1],
        ),
    )


def viewer_reachable(url: str) -> bool:
    """The gRPC connection never fails fast on an absent viewer - probe first."""
    parsed = urlparse(url.replace("rerun+http", "http", 1))
    if parsed.hostname is None or parsed.port is None:
        return True  # unparsable - let connect_grpc try
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=2):
            return True
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect-url", default=DEFAULT_CONNECT_URL, metavar="URL",
                    help=f"gRPC URL of the running viewer (default {DEFAULT_CONNECT_URL})")
    ap.add_argument("--save", default=None, metavar="PATH.rbl",
                    help="write the blueprint to a .rbl file instead of sending it")
    args = ap.parse_args()

    blueprint = make_blueprint()

    if args.save:
        blueprint.save(APP_ID, args.save)
        print(f"Blueprint saved to {args.save} - load it with: rerun {args.save}")
        return 0

    if not viewer_reachable(args.connect_url):
        print(f"Warning: no rerun viewer reachable at {args.connect_url} - "
              "blueprint not sent.", file=sys.stderr)
        return 0  # do not fail a demo pipeline over the layout

    rr.init(APP_ID)
    rr.connect_grpc(args.connect_url)
    rr.send_blueprint(blueprint)
    print(f"Viewer layout sent to {args.connect_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
