#!/usr/bin/env python3
"""
Send the viewer layout (blueprint) for frontend_rerun.

The rerun C++ SDK cannot author blueprints, so frontend_rerun logs data only and
runs this script to push the layout: the CPU and the GPU front end side by side
on top, the per-frame plots (frames done, time per frame, tracks) and the summary
below, with the time panel on the frontend_time timeline, where both runs start
at 0.

frontend_rerun runs it by itself. By hand:

    python3 blueprint.py                                  # viewer on this machine
    python3 blueprint.py --connect-url rerun+http://HOST:PORT/proxy
    python3 blueprint.py --save frontend_rerun.rbl        # write a blueprint file instead
"""

import argparse
import socket
import sys
from urllib.parse import urlparse

import rerun as rr
import rerun.blueprint as rrb

APP_ID = "vilib_frontend"  # must match frontend_rerun.cpp
DEFAULT_CONNECT_URL = "rerun+http://127.0.0.1:9876/proxy"


def make_blueprint(playback_speed: float) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="cpu/image", name="CPU: OpenCV KLT + Rosten FAST"),
                rrb.Spatial2DView(origin="gpu/image", name="GPU: vilib (CUDA), with upload"),
            ),
            rrb.Horizontal(
                # Legends in the corner the curves leave empty
                rrb.TimeSeriesView(origin="frames_done", name="Frames done",
                                   plot_legend=rrb.PlotLegend(corner="RightBottom")),
                rrb.TimeSeriesView(origin="frame_time_ms", name="Time per frame (ms)",
                                   plot_legend=rrb.PlotLegend(corner="RightTop")),
                rrb.TimeSeriesView(origin="tracks", name="Tracks",
                                   plot_legend=rrb.PlotLegend(corner="RightBottom")),
                rrb.TextDocumentView(origin="summary", name="Summary"),
            ),
            row_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline="frontend_time", playback_speed=playback_speed),
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
    ap.add_argument("--playback-speed", type=float, default=0.1,
                    help="playback speed of the frontend_time timeline (default 0.1)")
    args = ap.parse_args()

    blueprint = make_blueprint(args.playback_speed)

    if args.save:
        blueprint.save(APP_ID, args.save)
        print(f"Blueprint saved to {args.save}")
        return 0

    if not viewer_reachable(args.connect_url):
        print(f"Warning: no rerun viewer reachable at {args.connect_url} - "
              "blueprint not sent.", file=sys.stderr)
        return 0  # do not fail the demo over the layout

    rr.init(APP_ID)
    rr.connect_grpc(args.connect_url)
    rr.send_blueprint(blueprint)
    print(f"Viewer layout sent to {args.connect_url} "
          f"(frontend_time timeline, playback speed {args.playback_speed}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
