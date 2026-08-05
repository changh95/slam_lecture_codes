#!/usr/bin/env python3
# coding: utf-8
"""Run MonoLaneMapping on one OpenLane rosbag.

Same pipeline as upstream examples/demo_mapping.py -- LaneMapping.process()
does odometry, lane association, spline map update and iSAM2 optimisation --
but the bag is a CLI argument instead of a hardcoded path, and the final map
can be written to PNG instead of only being thrown at an interactive window.
"""
import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.environ.get("MONOLANE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from misc.config import cfg, cfg_from_yaml_file       # noqa: E402
from misc.utils import mkdir_if_missing               # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="MonoLaneMapping on an OpenLane rosbag")
    p.add_argument("--bag", default=os.path.join(
        ROOT_DIR, "examples/data/"
        "segment-14486517341017504003_3406_349_3426_349_with_camera_labels.bag"),
        help="rosbag with /gt_pose_wc, /lanes_gt, /lanes_predict")
    p.add_argument("--cfg_file", default=os.path.join(ROOT_DIR, "config/lane_mapping_docker.yaml"))
    p.add_argument("--output_dir", default="/out",
                   help="where stats.npy, the map .npy and screenshots go")
    p.add_argument("--gui", action="store_true",
                   help="open the interactive Open3D map viewer when the run finishes")
    p.add_argument("--screenshot", default=None,
                   help="render the finished map to this PNG (top-down, whole map)")
    p.add_argument("--detail_screenshot", default=None,
                   help="also render a 45 m close-up showing the control points")
    p.add_argument("--detail_at", type=float, default=0.40,
                   help="where along the drive the --detail_screenshot crop starts, "
                        "as a fraction of path length")
    p.add_argument("--from_map", default=None,
                   help="skip mapping and re-render an existing visualization/<seg>/map.npy")
    p.add_argument("--odo_noise", action="store_true",
                   help="corrupt the odometry (0.5 deg yaw, 0.5 m xy per frame) so the "
                        "pose half of the optimiser has something to correct; without "
                        "it the bag's poses are ground truth and every RPE is 0")
    p.add_argument("--eval_pose", action="store_true",
                   help="pose benchmark only -- skip map save and per-frame json output")
    p.add_argument("--all_segments", default=None, metavar="DIR",
                   help="run every *.bag in DIR and aggregate the pose benchmark "
                        "(upstream examples/mapping_bm.py, minus the json output "
                        "that needs the original OpenLane annotations)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cap --all_segments to N bags")
    return p.parse_args()


def _road_axis(points):
    """Unit heading of the road in the xy plane (first PCA component)."""
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    axis = np.linalg.svd(xy, full_matrices=False)[2][0]
    return np.array([axis[0], axis[1], 0.0]) / np.linalg.norm(axis[:2])


def _lane_geometries(lanes_in_map, crop=None):
    """The three layers LaneUI.visualize_map() draws, as Open3D geometries.

    Grey raw measurements, a coloured Catmull-Rom polyline per lane, and red
    spheres on the control points that the factor graph actually optimises.
    Upstream draws the spline as a dense point cloud; a LineSet reads better
    at map scale because it stays continuous however far the camera is.
    """
    import open3d as o3d
    from misc.plot_utils import colors_list
    from misc.curve.catmull_rom import CatmullRomSplineList

    geoms, fitted_all, ctrl_all = [], [], []
    for i, (_lane_id, lane) in enumerate(sorted(lanes_in_map.items())):
        raw = np.asarray(lane["xyz_raw"])[:, :3]
        ctrl = np.asarray(lane["ctrl_pts"])[:, :3]
        if len(ctrl) < 4:
            continue
        fitted = np.asarray(CatmullRomSplineList(ctrl).get_points(30))[:, :3]

        if crop is not None:
            keep = crop(fitted)
            if keep.sum() < 2:
                continue
            fitted = fitted[keep]
            raw = raw[crop(raw)]
            ctrl = ctrl[crop(ctrl)]

        colour = colors_list[i % len(colors_list)]
        if len(raw):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw)
            pcd.paint_uniform_color([0.62, 0.62, 0.62])
            geoms.append(pcd)

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(fitted)
        line.lines = o3d.utility.Vector2iVector(
            np.stack([np.arange(len(fitted) - 1), np.arange(1, len(fitted))], axis=1))
        line.colors = o3d.utility.Vector3dVector(np.tile(colour, (len(fitted) - 1, 1)))
        geoms.append(line)

        if len(ctrl):
            from misc.plot_utils import pointcloud_to_spheres
            geoms.append(pointcloud_to_spheres(ctrl, color=[0.85, 0.1, 0.1], sphere_size=0.32))
            ctrl_all.append(ctrl)
        fitted_all.append(fitted)

    if not fitted_all:
        return [], None, None
    return geoms, np.concatenate(fitted_all), (
        np.concatenate(ctrl_all) if ctrl_all else np.empty((0, 3)))


def _fit_camera(vis, points, right, up, width, height, fov_deg=55.0, margin=1.06):
    """Aim the camera along `right` x `up` so `points` just fill the canvas.

    Open3D's set_front/set_up/set_zoom fits a bounding *sphere*, which leaves a
    20 x 325 m ribbon of road as a hairline across an ocean of white. Writing
    the pinhole extrinsic directly is the only way to frame something this
    anisotropic, and because perspective scale depends on the distance being
    solved for, the fit is iterated: project, measure the margin actually used,
    push the camera in or out, repeat. Converges in a handful of rounds.
    """
    import open3d as o3d

    right = np.asarray(right, dtype=float)
    right /= np.linalg.norm(right)
    up = np.asarray(up, dtype=float)
    up /= np.linalg.norm(up)
    forward = np.cross(right, -up)          # camera +z looks into the scene
    forward /= np.linalg.norm(forward)
    rot = np.stack([right, -up, forward])   # world -> camera

    f = (height / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    target = 0.5 * (points.min(axis=0) + points.max(axis=0))
    spans = points @ np.stack([right, up]).T
    distance = max(np.ptp(spans[:, 0]) / 2.0 * f / (width / 2.0),
                   np.ptp(spans[:, 1]) / 2.0 * f / (height / 2.0), 1.0)

    for _ in range(50):
        eye = target - forward * distance
        cam = (points - eye) @ rot.T
        depth = np.clip(cam[:, 2], 1e-3, None)
        u, v = f * cam[:, 0] / depth, f * cam[:, 1] / depth
        target = (target
                  + right * (0.5 * (u.max() + u.min()) * distance / f)
                  - up * (0.5 * (v.max() + v.min()) * distance / f))
        scale = max((u.max() - u.min()) / (width / margin),
                    (v.max() - v.min()) / (height / margin))
        if 0.99 < scale <= 1.0:
            break
        distance *= scale

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = -rot @ (target - forward * distance)

    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, f, f, width / 2.0 - 0.5, height / 2.0 - 0.5)
    params.extrinsic = extrinsic
    vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def render_map(lanes_in_map, png_path, view="bev", title="", detail_at=0.40):
    """Render the finished map offscreen (works headless under Xvfb)."""
    import open3d as o3d

    geoms, fitted, _ = _lane_geometries(lanes_in_map)
    if fitted is None:
        print("nothing in the map to render")
        return False
    axis = _road_axis(fitted)

    if view == "detail":
        # A 45 m window from the middle of the drive, close enough that the
        # 3 m control-point chord and the raw measurements are separable.
        mid = fitted.mean(axis=0)
        start = np.quantile((fitted - mid) @ axis, detail_at)

        def crop(pts):
            t = (pts - mid) @ axis
            return (t >= start) & (t <= start + 45.0)

        geoms, fitted, _ = _lane_geometries(lanes_in_map, crop=crop)
        if fitted is None:
            print("nothing in the map to render")
            return False
        # Re-fit the axis to just the crop. The global PCA axis is a poor local
        # frame on a curving road -- using it tilts the close-up and throws the
        # lanes diagonally across the canvas.
        axis = _road_axis(fitted)
        width, height = 1600, 620
    else:
        # The map is ~17x longer than it is wide, so the canvas is a strip.
        width, height = 2400, 520

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=width, height=height, visible=False)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = 2.0
    opt.line_width = 4.0

    across = np.array([-axis[1], axis[0], 0.0])
    if view == "detail":
        # Tilt over the *side* of the road rather than down its length: the road
        # stays horizontal and full width, and the oblique gives the control
        # spheres enough parallax to sit visibly above the raw point ribbon.
        eye_dir = np.array([0.0, 0.0, 0.88]) + across * 0.47
        eye_dir /= np.linalg.norm(eye_dir)
        _fit_camera(vis, fitted, axis, np.cross(eye_dir, axis), width, height)
    else:
        _fit_camera(vis, fitted, axis, across, width, height)

    vis.poll_events()
    vis.update_renderer()
    mkdir_if_missing(os.path.dirname(os.path.abspath(png_path)))
    vis.capture_screen_image(png_path, do_render=True)
    vis.destroy_window()
    print("wrote {}  ({:.0f} m along the road, {:.0f} m across)".format(
        png_path, np.ptp(fitted @ axis), np.ptp(fitted @ across)))
    return True


def _run_one(bag):
    """One segment, pose benchmark only. Top level so Pool can pickle it."""
    np.random.seed(666)
    from lane_slam.system.lane_mapping import LaneMapping
    mapper = LaneMapping(bag, save_result=False)
    return mapper.segment, mapper.process()


def run_all_segments(bag_dir, workers, limit):
    import glob
    from multiprocessing import Pool

    bags = sorted(glob.glob(os.path.join(bag_dir, "*.bag")))
    if limit:
        bags = bags[:limit]
    if not bags:
        print("no .bag files under {}".format(bag_dir))
        return
    print("running {} segments on {} workers".format(len(bags), workers))

    with Pool(workers) as pool:
        results = pool.map(_run_one, bags)

    intervals = cfg.evaluation.intervals
    pooled = {i: {k: [] for k in ("error_rot", "error_rot_raw",
                                  "error_trans", "error_trans_raw")} for i in intervals}
    # Pool.map re-raises a worker exception, so anything here ran to completion.
    length, sizes = 0.0, []
    for _segment, stats in results:
        length += stats["path_length"]
        sizes.append(stats["map_size"])
        for i in intervals:
            for k in pooled[i]:
                pooled[i][k].extend(stats[i][k])

    print("\n=== {} segments, {:.1f} km of driving ===".format(len(sizes), length / 1000.0))
    print("control points per segment map: mean {:.0f}".format(np.mean(sizes)))
    print("\nrelative pose error, pooled over all segments (optimised / raw odometry):")
    for i in intervals:
        v = pooled[i]
        if not v["error_rot"]:
            continue
        print("  {:>2d} m : rot {:.3f} / {:.3f} deg   trans {:.3f} / {:.3f} m   ({} pairs)".format(
            i, np.mean(v["error_rot"]), np.mean(v["error_rot_raw"]),
            np.mean(v["error_trans"]), np.mean(v["error_trans_raw"]), len(v["error_rot"])))


def main():
    args = parse_args()
    np.random.seed(666)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.visualization = False          # we drive the viewer ourselves, below
    cfg.eval_pose = args.eval_pose
    cfg.pose_update.add_odo_noise = args.odo_noise
    cfg.output_dir = args.output_dir
    for sub in ("logs", "visualization", "results", "results_det", "eval_results"):
        mkdir_if_missing(os.path.join(cfg.output_dir, sub))

    if args.all_segments:
        cfg.eval_pose = True        # no map save / json dump for 200+ segments
        run_all_segments(args.all_segments, args.workers, args.limit)
        return

    if args.from_map:
        lanes = np.load(args.from_map, allow_pickle=True).item()["lanes_in_map"]
        if args.screenshot:
            render_map(lanes, args.screenshot, "bev", os.path.basename(args.from_map))
        if args.detail_screenshot:
            render_map(lanes, args.detail_screenshot, "detail", os.path.basename(args.from_map), args.detail_at)
        return

    # save_result=True needs the original OpenLane json annotations for the
    # camera extrinsics; the rosbag-only quick start does not have them.
    from lane_slam.system.lane_mapping import LaneMapping   # noqa: E402
    print("bag:    {}".format(args.bag))
    print("config: {}".format(args.cfg_file))
    mapper = LaneMapping(args.bag, save_result=False)
    print("frames: {}".format(len(mapper.frames_data)))

    stats = mapper.process()

    np.save(os.path.join(cfg.output_dir, "stats.npy"), {mapper.segment: stats})

    print("\n=== {} ===".format(mapper.segment))
    print("path length            : {:.1f} m".format(stats["path_length"]))
    print("lanes in map           : {}".format(len(mapper.lanes_in_map)))
    # upstream's "map size" metric, sampled before post_merge_lane() drops the
    # overlapping landmarks -- so it is a little above the saved map's count
    print("control points         : {} (upstream map_size metric)".format(stats["map_size"]))
    print("graph build            : {:.2f} ms/frame".format(stats["graph"]))
    print("iSAM2 / LM solve       : {:.2f} ms/frame".format(stats["isam"]))
    print("whole pipeline         : {:.2f} ms/frame".format(mapper.whole_timer.avg * 1000))
    print("  odometry             : {:.2f} ms/frame".format(mapper.odo_timer.avg * 1000))
    print("  lane association     : {:.2f} ms/frame".format(mapper.assoc_timer.avg * 1000))
    print("\nrelative pose error (optimised / raw odometry):")
    for interval in cfg.evaluation.intervals:
        v = stats.get(interval)
        if not v or not v["error_rot"]:
            continue
        print("  {:>2d} m : rot {:.3f} / {:.3f} deg   trans {:.3f} / {:.3f} m".format(
            interval, np.mean(v["error_rot"]), np.mean(v["error_rot_raw"]),
            np.mean(v["error_trans"]), np.mean(v["error_trans_raw"])))

    map_npy = os.path.join(cfg.output_dir, "visualization", mapper.segment, "map.npy")
    lanes_in_map = np.load(map_npy, allow_pickle=True).item()["lanes_in_map"]

    # The point of the Catmull-Rom parameterisation: the map is the control
    # points, not the measurements that produced them.
    raw = sum(len(v["xyz_raw"]) for v in lanes_in_map.values())
    ctrl = sum(len(v["ctrl_pts"]) for v in lanes_in_map.values())
    print("\nsaved map              : {} lanes, {} control points".format(len(lanes_in_map), ctrl))
    print("  vs raw measurements  : {} points -> {:.0f}x fewer ({:.1f} kB vs {:.0f} kB as f32)"
          .format(raw, raw / max(ctrl, 1), ctrl * 12 / 1024.0, raw * 12 / 1024.0))
    print("  written to           : {}".format(map_npy))
    if args.screenshot:
        render_map(lanes_in_map, args.screenshot, "bev", mapper.segment)
    if args.detail_screenshot:
        render_map(lanes_in_map, args.detail_screenshot, "detail", mapper.segment, args.detail_at)
    if args.gui:
        mapper.visualize_map()


if __name__ == "__main__":
    main()
