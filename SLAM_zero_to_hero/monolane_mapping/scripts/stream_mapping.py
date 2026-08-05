#!/usr/bin/env python3
# coding: utf-8
"""Stream MonoLaneMapping's lane map to a Rerun viewer as it is built.

Same pipeline as run_mapping.py -- this only watches. Upstream's process() loop
ends each frame with lane_nms(), so subclassing and wrapping that one method
gives a per-frame callback with fully settled state, with no edit to the
upstream source and no copy of its loop. (Hooking map_update() instead would
fire *before* the NMS prune, so landmarks about to be dropped would flicker
into the map for one frame.)

What lands on the timeline, per frame:
  world/map/lane_NNN     the Catmull-Rom spline + its control points, coloured
                         by landmark id -- so association is visible, and a
                         track that gets re-initialised changes colour
  world/frame/detections this frame's raw detections in world coords, coloured
                         by the landmark they were associated to (white = not
                         associated to anything yet)
  world/vehicle          the estimated camera pose
  world/traj/{est,odom,gt}
  plots/...              map growth, per-frame cost, pose error against GT

The three trajectories only separate when the odometry is corrupted, so run
with --odo_noise to see the pose actually being pulled back onto the lanes.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT_DIR = os.environ.get("MONOLANE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import rerun as rr                                    # noqa: E402
import rerun.blueprint as rrb                         # noqa: E402

from misc.config import cfg, cfg_from_yaml_file       # noqa: E402
from misc.utils import mkdir_if_missing               # noqa: E402

APP_ID = "monolane_mapping"
DEFAULT_BAG = os.path.join(
    ROOT_DIR, "examples/data/"
    "segment-14486517341017504003_3406_349_3426_349_with_camera_labels.bag")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", default=DEFAULT_BAG)
    p.add_argument("--cfg_file", default=os.path.join(ROOT_DIR, "config/lane_mapping_docker.yaml"))
    p.add_argument("--output_dir", default="/out")
    p.add_argument("--odo_noise", action="store_true",
                   help="corrupt the odometry so the pose factors have work to do")

    sink = p.add_argument_group("where to send the stream")
    sink.add_argument("--serve", action="store_true",
                     help="host the web viewer in this container (the default)")
    sink.add_argument("--web_port", type=int, default=9090)
    sink.add_argument("--ws_port", type=int, default=9877)
    sink.add_argument("--connect", metavar="HOST:PORT", default=None,
                     help="stream to a viewer already running elsewhere, e.g. 127.0.0.1:9876")
    sink.add_argument("--rrd", metavar="FILE", default=None,
                     help="record to an .rrd instead of streaming; replay with `rerun FILE`")

    p.add_argument("--rate", type=float, default=10.0,
                   help="frames/s to pace playback at; the bags are 10 Hz, so 10 is "
                        "real time. 0 runs flat out (~13 fps here)")
    p.add_argument("--spline_samples", type=int, default=8,
                   help="samples per 3 m spline span. Upstream's still renders use 30; "
                        "8 looks the same at map scale and re-fits 4x cheaper per frame")
    return p.parse_args()


# Upstream's own palette, so the live view and the README's PNGs agree.
def _palette():
    from misc.plot_utils import colors_list
    return (np.asarray(colors_list) * 255).astype(np.uint8)


class StreamingLaneMapping(object):
    """Mixin factory -- built lazily so `import rerun` cannot fail before argparse."""

    @staticmethod
    def build(bag_file, on_frame):
        from lane_slam.system.lane_mapping import LaneMapping

        class _Streaming(LaneMapping):
            def lane_nms(self, frame):
                super(_Streaming, self).lane_nms(frame)   # let the prune run first
                on_frame(self)

        return _Streaming(bag_file, save_result=False)


class RerunLogger(object):
    def __init__(self, rate_hz, spline_samples):
        self.period = (1.0 / rate_hz) if rate_hz and rate_hz > 0 else 0.0
        self.samples = spline_samples
        self.colors = _palette()
        self.t_sensor0 = None
        self.next_deadline = None
        self.live_ids = set()
        self._prev_ctrl = {}
        self.est, self.odom, self.gt = [], [], []

        from misc.curve.catmull_rom import CatmullRomSplineList
        self._Spline = CatmullRomSplineList

    # ---------------------------------------------------------------- static
    def log_static(self):
        # OpenLane's camera frame is x-forward, y-left, z-up.
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for path, name, colour, width in (
                ("world/traj/est", "estimated", [40, 170, 70], 2.0),
                ("world/traj/odom", "raw odometry", [210, 60, 60], 2.0),
                ("world/traj/gt", "ground truth", [40, 40, 40], 2.0)):
            rr.log(path + "_style", rr.SeriesLine(color=colour, name=name, width=width),
                   static=True)
        for path, name, colour in (
                ("plots/map/landmarks", "lane landmarks", [70, 110, 200]),
                ("plots/map/control_points", "control points", [200, 120, 40]),
                ("plots/timing/total", "whole frame", [40, 40, 40]),
                ("plots/timing/graph_build", "graph build", [70, 110, 200]),
                ("plots/timing/isam", "iSAM2 solve", [200, 120, 40]),
                ("plots/timing/association", "lane association", [40, 170, 70]),
                ("plots/pose/optimised", "optimised vs GT", [40, 170, 70]),
                ("plots/pose/raw_odometry", "raw odometry vs GT", [210, 60, 60])):
            rr.log(path, rr.SeriesLine(color=colour, name=name, width=2.0), static=True)

    # ----------------------------------------------------------- per frame
    def __call__(self, mapper):
        frame = mapper.cur_frame
        if self.t_sensor0 is None:
            self.t_sensor0 = frame.timestamp
        rr.set_time_sequence("frame", frame.frame_id)
        rr.set_time_seconds("sensor_time", frame.timestamp - self.t_sensor0)

        self._log_map(mapper)
        self._log_detections(frame)
        self._log_poses(mapper, frame)
        self._log_plots(mapper)
        self._pace()

    def _log_map(self, mapper):
        seen = set()
        n_ctrl = 0
        for lane_id, lm in mapper.lanes_in_map.items():
            ctrl = np.asarray(lm.get_ctrl_xyz(), dtype=float).reshape(-1, 3)
            n_ctrl += len(ctrl)
            if len(ctrl) < 4:            # a Catmull-Rom span needs four
                continue
            seen.add(lane_id)

            # Only the landmarks in the sliding window move on any given frame,
            # so re-fitting all of them every frame burns time and bloats the
            # .rrd with duplicates. A rerun entity persists on the timeline
            # until it is logged again, so skipping an unchanged landmark leaves
            # the viewer showing exactly the same thing.
            prev = self._prev_ctrl.get(lane_id)
            if prev is not None and prev.shape == ctrl.shape and np.array_equal(prev, ctrl):
                continue
            self._prev_ctrl[lane_id] = ctrl.copy()

            base = "world/map/lane_{:03d}".format(lane_id)
            colour = self.colors[lane_id % len(self.colors)]
            fitted = np.asarray(self._Spline(ctrl).get_points(self.samples))[:, :3]
            rr.log(base + "/spline", rr.LineStrips3D([fitted], colors=[colour], radii=0.10))
            rr.log(base + "/control", rr.Points3D(ctrl, colors=[200, 30, 30], radii=0.28))

        # NMS and the merge pass delete landmarks; without an explicit clear the
        # viewer would keep showing their last known geometry forever.
        for stale in self.live_ids - seen:
            rr.log("world/map/lane_{:03d}".format(stale), rr.Clear(recursive=True))
            self._prev_ctrl.pop(stale, None)
        self.live_ids = seen
        self._n_ctrl = n_ctrl

    def _log_detections(self, frame):
        pts, cols = [], []
        for lf in frame.get_lane_features():
            xyz = np.asarray(lf.get_xyzs(), dtype=float).reshape(-1, 3)
            if not len(xyz):
                continue
            xyz = xyz @ frame.T_wc[:3, :3].T + frame.T_wc[:3, 3]
            pts.append(xyz)
            # white until the associator gives it a landmark, then the map colour
            colour = ([235, 235, 235] if lf.id == -1
                      else self.colors[lf.id % len(self.colors)])
            cols.append(np.tile(colour, (len(xyz), 1)))
        if pts:
            rr.log("world/frame/detections",
                   rr.Points3D(np.concatenate(pts), colors=np.concatenate(cols), radii=0.09))
        else:
            rr.log("world/frame/detections", rr.Clear(recursive=False))

    def _log_poses(self, mapper, frame):
        self.est.append(frame.T_wc[:3, 3].copy())
        self.odom.append(mapper.raw_pose[-1][:3, 3].copy())
        self.gt.append(mapper.gt_pose[-1][:3, 3].copy())

        rr.log("world/vehicle", rr.Transform3D(translation=frame.T_wc[:3, 3],
                                               mat3x3=frame.T_wc[:3, :3]))
        rr.log("world/vehicle/axes",
               rr.Arrows3D(vectors=np.eye(3) * 2.0,
                           colors=[[220, 50, 50], [50, 200, 50], [50, 120, 230]]))

        for path, track, colour in (("world/traj/est", self.est, [40, 170, 70]),
                                    ("world/traj/odom", self.odom, [210, 60, 60]),
                                    ("world/traj/gt", self.gt, [40, 40, 40])):
            if len(track) > 1:
                rr.log(path, rr.LineStrips3D([np.asarray(track)],
                                             colors=[colour], radii=0.16))

    def _log_plots(self, mapper):
        rr.log("plots/map/landmarks", rr.Scalar(len(mapper.lanes_in_map)))
        rr.log("plots/map/control_points", rr.Scalar(self._n_ctrl))
        rr.log("plots/timing/total", rr.Scalar(mapper.whole_timer.val * 1000.0))
        rr.log("plots/timing/graph_build", rr.Scalar(mapper.graph_build_timer.val * 1000.0))
        rr.log("plots/timing/isam", rr.Scalar(mapper.opt_timer.val * 1000.0))
        rr.log("plots/timing/association", rr.Scalar(mapper.assoc_timer.val * 1000.0))
        rr.log("plots/pose/optimised",
               rr.Scalar(float(np.linalg.norm(self.est[-1] - self.gt[-1]))))
        rr.log("plots/pose/raw_odometry",
               rr.Scalar(float(np.linalg.norm(self.odom[-1] - self.gt[-1]))))

    def _pace(self):
        if not self.period:
            return
        now = time.time()
        if self.next_deadline is None:
            self.next_deadline = now
        self.next_deadline += self.period
        slack = self.next_deadline - now
        if slack > 0:
            time.sleep(slack)
        else:
            self.next_deadline = now      # fell behind; don't accumulate debt


def blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Lane map (building)"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="plots/map", name="Map growth"),
                rrb.TimeSeriesView(origin="plots/timing", name="Per-frame cost (ms)"),
                rrb.TimeSeriesView(origin="plots/pose", name="Position error vs GT (m)"),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded"),
        collapse_panels=True,
    )


def main():
    args = parse_args()
    np.random.seed(666)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.visualization = False
    cfg.eval_pose = False
    cfg.pose_update.add_odo_noise = args.odo_noise
    cfg.output_dir = args.output_dir
    for sub in ("logs", "visualization", "results", "results_det", "eval_results"):
        mkdir_if_missing(os.path.join(cfg.output_dir, sub))

    rr.init(APP_ID, default_enabled=True)
    if args.rrd:
        mkdir_if_missing(os.path.dirname(os.path.abspath(args.rrd)))
        rr.send_blueprint(blueprint())
        rr.save(args.rrd)
        print("recording to {}".format(args.rrd))
    elif args.connect:
        rr.connect(args.connect, default_blueprint=blueprint())
        print("streaming to viewer at {}".format(args.connect))
    else:
        rr.serve(open_browser=False, web_port=args.web_port, ws_port=args.ws_port,
                 default_blueprint=blueprint())
        print("\n  open  http://localhost:{}/?url=ws://localhost:{}\n"
              .format(args.web_port, args.ws_port))

    logger = RerunLogger(args.rate, args.spline_samples)
    logger.log_static()

    print("loading {}".format(os.path.basename(args.bag)))
    mapper = StreamingLaneMapping.build(args.bag, logger)
    print("streaming {} frames at {}\n".format(
        len(mapper.frames_data), "{:g} fps".format(args.rate) if args.rate else "full speed"))

    t0 = time.time()
    mapper.process()
    wall = time.time() - t0

    n = len(mapper.frames_data)
    print("\ndone: {} frames in {:.1f} s ({:.1f} fps)".format(n, wall, n / wall))
    # process() has already run post_merge_lane() by now, so this is the same
    # count run_mapping.py reports for the saved map -- not its pre-merge
    # "upstream map_size metric" line.
    print("final map after the merge pass: {} landmarks, {} control points".format(
        len(mapper.lanes_in_map), mapper.map_size()))

    if not args.rrd and not args.connect:
        print("\nviewer is still served; Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    main()
