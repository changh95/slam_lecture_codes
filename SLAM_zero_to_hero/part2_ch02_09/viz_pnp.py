#!/usr/bin/env python3
"""
Rerun viewer for the AprilTag + PnP camera localization demos.

Reads one or more JSON files produced by pnp_opencv / pnp_poselib / pnp_opengv
and logs:
  - the video frames with the detected tag corners overlaid (2D),
  - a 3D camera frustum + camera trajectory per PnP method,
  - the mapped AprilTag squares in 3D (world frame = tag 10),
  - per-frame reprojection error and solve time as scalar plots.

Usage:
    python3 viz_pnp.py pnp_opencv.json [pnp_poselib.json pnp_opengv.json ...]
    python3 viz_pnp.py --save out.rrd pnp_opencv.json
    python3 viz_pnp.py --connect pnp_opencv.json     # stream to a running viewer
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

DEFAULT_CONNECT_URL = "rerun+http://127.0.0.1:9876/proxy"

# Distinct color per PnP method.
METHOD_COLORS = {
    "opencv":  (0, 200, 0),      # green
    "poselib": (0, 120, 255),    # blue
    "opengv":  (220, 60, 60),    # red
}
FALLBACK_COLOR = (200, 200, 0)

MM_TO_M = 1.0e-3


def set_frame(i: int):
    """rerun 0.23+ uses rr.set_time; older SDKs use rr.set_time_sequence."""
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=i)
    else:
        rr.set_time_sequence("frame", i)


def log_scalar(entity: str, value: float):
    if hasattr(rr, "Scalars"):
        rr.log(entity, rr.Scalars(value))
    else:
        rr.log(entity, rr.Scalar(value))


def log_series_style(entity: str, name: str, color):
    """Static line style so each method gets a named, colored series."""
    if hasattr(rr, "SeriesLines"):
        rr.log(entity, rr.SeriesLines(colors=[color], names=[name], widths=[1.5]),
               static=True)
    elif hasattr(rr, "SeriesLine"):
        rr.log(entity, rr.SeriesLine(color=color, name=name, width=1.5), static=True)


def send_blueprint(img_method: str):
    """Viewer layout: 3D view + camera view on top, one consolidated
    time-series plot per metric (all methods in the same plot) below."""
    try:
        import rerun.blueprint as rrb

        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="World (tag 10 frame)"),
                    rrb.Spatial2DView(origin=f"world/{img_method}/cam/image",
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
        rr.send_blueprint(blueprint)
    except Exception as e:  # blueprint API varies across SDK versions
        print(f"Warning: could not send blueprint ({e}) - using default layout")


def log_tag_map(method: str, color, tag_map):
    """Static 3D squares + id labels for the mapped tags."""
    strips, labels, centers = [], [], []
    for tag in tag_map:
        c = np.array(tag["corners"], dtype=np.float64) * MM_TO_M
        strips.append(np.vstack([c, c[:1]]))  # closed square
        labels.append(f"tag {tag['id']}")
        centers.append(c.mean(axis=0))
    if not strips:
        return
    rr.log(f"world/tags/{method}", rr.LineStrips3D(strips, colors=[color]), static=True)
    rr.log(f"world/tags/{method}/ids",
           rr.Points3D(centers, radii=0.004, colors=[color], labels=labels),
           static=True)


def log_detections_2d(entity: str, detections, scale: float, color, transform=None):
    """Tag outlines + ids on top of the video frame."""
    strips, labels = [], []
    for det in detections:
        px = np.array(det["px"], dtype=np.float64)
        if transform is not None:
            px = transform(px)
        px = px * scale
        strips.append(np.vstack([px, px[:1]]))
        labels.append(f"id {det['id']}")
    if strips:
        rr.log(entity, rr.LineStrips2D(strips, colors=[color], labels=labels))
    else:
        rr.log(entity, rr.Clear(recursive=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_files", nargs="+", help="JSON files from the C++ demos")
    ap.add_argument("--video", default=None,
                    help="video path (default: taken from the first JSON)")
    ap.add_argument("--img-stride", type=int, default=1,
                    help="log every Nth video frame image (default: 1 = every frame)")
    ap.add_argument("--img-scale", type=float, default=0.5,
                    help="downscale factor for logged images (default: 0.5)")
    ap.add_argument("--no-images", action="store_true", help="do not log video frames")
    ap.add_argument("--no-undistort", action="store_true",
                    help="show raw distorted frames instead of undistorting them "
                         "to match the pinhole frustum model")
    ap.add_argument("--save", default=None, help="save to .rrd instead of spawning a viewer")
    ap.add_argument("--connect", action="store_true",
                    help="stream to an already-running rerun viewer "
                         f"at {DEFAULT_CONNECT_URL} (override with --connect-url)")
    ap.add_argument("--connect-url", default=None, metavar="URL",
                    help=f"gRPC URL of the running viewer (default {DEFAULT_CONNECT_URL}); "
                         "implies --connect")
    args = ap.parse_args()

    # Any supplied --connect-url implies --connect; otherwise fall back to default.
    connect = args.connect or args.connect_url is not None
    connect_url = args.connect_url or DEFAULT_CONNECT_URL

    results = []
    for path in args.json_files:
        with open(path) as f:
            data = json.load(f)
        results.append(data)
        print(f"{path}: method={data['method']}, "
              f"{len(data['frames'])} frames, {len(data['tag_map'])} tags mapped")

    # Undistortion: the frustum uses an ideal pinhole model, so raw (distorted)
    # frames and corner overlays are warped to match it. Requires dist_coeffs
    # in the JSON (written by the C++ demos).
    cam0 = results[0]["camera"]
    K = np.array([[cam0["fx"], 0.0, cam0["cx"]],
                  [0.0, cam0["fy"], cam0["cy"]],
                  [0.0, 0.0, 1.0]])
    dist = np.array(cam0.get("dist_coeffs", []), dtype=np.float64)
    undistort = (not args.no_undistort) and dist.size > 0 and np.any(dist != 0.0)
    if not args.no_undistort and not undistort:
        print("Note: no dist_coeffs in the JSON - showing raw frames "
              "(re-run the C++ demo to regenerate the JSON)")

    def undist_pts(px):
        """Distorted pixel coords -> pixel coords in the undistorted image."""
        return cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    pts_transform = undist_pts if undistort else None

    rr.init("apriltag_pnp", spawn=(args.save is None and not connect))
    if connect:
        rr.connect_grpc(connect_url)
    elif args.save:
        rr.save(args.save)

    # World frame = tag 10 (z = tag normal). 0.2 m axes.
    rr.log("world",
           rr.Arrows3D(vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
                       colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                       labels=["x", "y", "z"]),
           static=True)

    # Consolidated plot layout (one shared plot per metric).
    send_blueprint(results[0]["method"])

    # Static per-method data: tag map, pinhole camera, trajectory, plot style.
    scale = args.img_scale
    for data in results:
        method = data["method"]
        color = METHOD_COLORS.get(method, FALLBACK_COLOR)
        cam = data["camera"]

        log_series_style(f"plots/reproj_err_px/{method}", method, color)
        log_series_style(f"plots/solve_ms/{method}", method, color)

        log_tag_map(method, color, data["tag_map"])

        rr.log(f"world/{method}/cam",
               rr.Pinhole(focal_length=[cam["fx"] * scale, cam["fy"] * scale],
                          principal_point=[cam["cx"] * scale, cam["cy"] * scale],
                          resolution=[int(cam["width"] * scale), int(cam["height"] * scale)],
                          camera_xyz=rr.ViewCoordinates.RDF,
                          image_plane_distance=0.15),
               static=True)

        traj = [np.array(f["T_wc"], dtype=np.float64).reshape(4, 4)[:3, 3] * MM_TO_M
                for f in data["frames"] if f["valid"]]
        if traj:
            rr.log(f"world/{method}/trajectory",
                   rr.LineStrips3D([np.array(traj)], colors=[color]),
                   static=True)

    # Video (images are identical for all methods -> log under the first one).
    video_path = args.video or results[0]["video"]
    cap = None
    if not args.no_images:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: cannot open video {video_path} - skipping images "
                  f"(pass --video or run from the demo's working directory)")
            cap = None
    img_method = results[0]["method"]
    img_color = METHOD_COLORS.get(img_method, FALLBACK_COLOR)

    # Per-frame data, driven by the first JSON's frame list.
    n_frames = max(len(d["frames"]) for d in results)
    video_frame_idx = -1
    frame_img = None

    for i in range(n_frames):
        set_frame(i)

        # Camera pose + scalars per method.
        for data in results:
            method = data["method"]
            frames = data["frames"]
            if i >= len(frames):
                continue
            f = frames[i]
            if f["valid"]:
                T = np.array(f["T_wc"], dtype=np.float64).reshape(4, 4)
                rr.log(f"world/{method}/cam",
                       rr.Transform3D(translation=T[:3, 3] * MM_TO_M, mat3x3=T[:3, :3]))
                log_scalar(f"plots/reproj_err_px/{method}", f["reproj_err_px"])
                log_scalar(f"plots/solve_ms/{method}", f["solve_ms"])

        # 2D detections (tag outlines + ids) on every frame.
        log_detections_2d(f"world/{img_method}/cam/image/detections",
                          results[0]["frames"][i]["detections"]
                          if i < len(results[0]["frames"]) else [],
                          scale, img_color, transform=pts_transform)

        # Video frame (only every Nth frame, downscaled).
        if cap is not None and i % args.img_stride == 0:
            while video_frame_idx < i:
                ok, frame_img = cap.read()
                if not ok:
                    cap = None
                    frame_img = None
                    break
                video_frame_idx += 1
            if frame_img is not None:
                shown = cv2.undistort(frame_img, K, dist) if undistort else frame_img
                rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
                if scale != 1.0:
                    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
                img = rr.Image(rgb)
                if hasattr(img, "compress"):
                    img = img.compress(jpeg_quality=75)
                rr.log(f"world/{img_method}/cam/image", img)

    print(f"Logged {n_frames} frames for {len(results)} method(s).")
    if args.save:
        print(f"Saved to {args.save} - open with: rerun {args.save}")


if __name__ == "__main__":
    main()
