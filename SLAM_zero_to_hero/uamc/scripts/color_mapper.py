#!/usr/bin/env python3
"""Live camera-coloured map from the Avia + Mid-360 LiDARs, placed by the Mid-360 LIO poses.

FAST-LIVO2 runs Mid-360 LiDAR + Mid-360 IMU LIO and writes one TUM pose per scan (Mid-360 IMU
in its world frame, stamped with the scan time) to TRAJ. This node runs alongside it:
  - tails TRAJ for poses (the /aft_mapped_to_init odometry is stamped with now(), not scan time),
  - batches one Mid-360 scan (0.1 s) with the Avia scans that start inside it, once a pose past
    the batch's last point exists: Avia points kept at 4-40 m range, Mid-360 at 0.1-20 m,
    every point de-skewed with the pose interpolated at its own time,
  - colours the batch from the Oak-D image nearest its middle, projected with the camera pose
    at that image's time; points outside the image are dropped,
  - publishes the first point of each --pub-leaf voxel on /color_cloud, and the pose used
    for the batch on /color_pose and /color_path (camera_init, stamped with the batch time),
  - on SIGINT/SIGTERM writes every coloured point with --raw, else the multi-resolution averaged
    map to OUT (x y z rgb PCD): cells from
    --max-leaf down to --min-leaf, split where at least --split-views batches saw them
    (scripts/multires_map.py), so places seen from many views come out denser.

  color_mapper.py TRAJ OUT.pcd [--min-leaf 0.005 --max-leaf 0.16 --split-views 3] [--pub-leaf 0.01]
                  [--avia-range 4 40] [--mid-range 0.1 20] [--img-offset 0.0]

--img-offset is added to the image header stamps to put them on the LiDAR clock: 0 on lvi_ghm_set
(one clock), 1200.293 on lvi_coex_set_2 (camera on the host clock; measured 2026-10-26 by
correlating the image rotation rate with the Mid-360 gyro, 1200.288 / 1200.298 s in two windows).
"""
import argparse
import collections
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from livox_ros_driver2.msg import CustomMsg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multires_map import MultiResMap  # noqa: E402
from pcd_io import DT, write_pcd  # noqa: E402
import uamc_geom  # noqa: E402
from uamc_geom import CAM_H, CAM_W, DIST, K, R_CI, T_CI, interp_pose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("traj")
ap.add_argument("out")
ap.add_argument("--raw", action="store_true", help="save every coloured point, no averaging (else multi-resolution)")
ap.add_argument("--min-leaf", type=float, default=0.005, help="finest cell of the saved map (m)")
ap.add_argument("--max-leaf", type=float, default=0.16, help="coarsest cell of the saved map (m)")
ap.add_argument("--split-views", type=int, default=3, help="batches that must see a cell before it is split")
ap.add_argument("--pub-leaf", type=float, default=0.01, help="/color_cloud sends the first point per voxel of this size (m); 0 = every point")
ap.add_argument("--avia-range", type=float, nargs=2, default=[4.0, 40.0], metavar=("MIN", "MAX"))
ap.add_argument("--mid-range", type=float, nargs=2, default=[0.1, 20.0], metavar=("MIN", "MAX"))
ap.add_argument("--max-dt", type=float, default=0.05, help="max |image - batch| time to colour a batch (s)")
ap.add_argument("--img-offset", type=float, default=0.0, help="added to image stamps to reach the LiDAR clock (s)")
ap.add_argument("--avia", default=uamc_geom.AVIA_TOPIC)
ap.add_argument("--mid", default=uamc_geom.MID_TOPIC)
ap.add_argument("--image", default=uamc_geom.IMAGE_TOPIC)
a = ap.parse_args()

def parse_avia(buf):
    return uamc_geom.parse_avia(buf, a.avia_range)


def parse_mid(m):
    return uamc_geom.parse_mid(m, a.mid_range)


B = 1 << 20


def vkey(xyz, leaf):
    k = np.floor(xyz / leaf).astype(np.int64) + B
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def to_pcd(xyz, bgr):
    """x y z rgb records (pcd_io.DT) from world points and BGR colours."""
    c = np.zeros(len(xyz), dtype=DT)
    c["x"], c["y"], c["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    b, g, r = (bgr[:, i].astype(np.uint32) for i in range(3))
    c["rgb"] = (r << 16) | (g << 8) | b
    return c


def stamp(msg, t):
    msg.sec, msg.nanosec = int(t), int((t % 1) * 1e9)


class ColorMapper:
    def __init__(self, node):
        self.node = node
        self.tt = np.zeros(0); self.tp = np.zeros((0, 3)); self.tq = np.zeros((0, 4))
        self.traj_pos = 0
        self.mid = collections.deque()                     # (t_first, t_last, times, xyz_imu)
        self.avia = collections.deque()
        self.images = collections.deque(maxlen=60)          # (t, BGR) ~2 s at 30 Hz
        self.img_lock = threading.Lock()                    # callbacks run on the spin thread
        self.known = []                                     # published pub-leaf voxels: sorted arrays,
                                                            # merged when sizes meet (log-structured)
        self.map = None if a.raw else MultiResMap(a.min_leaf, a.max_leaf, a.split_views)
        self.raw = []                                       # --raw: x y z rgb chunks, as they come
        self.n_batch = self.n_pts = self.n_col = self.n_drop = self.n_pub = 0
        self.n_src = {"avia": 0, "mid": 0}
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(CustomMsg, a.avia, self.on_avia, qos, raw=True)
        node.create_subscription(PointCloud2, a.mid, self.on_mid, qos)
        node.create_subscription(Image, a.image, self.on_image, QoSProfile(depth=30, reliability=ReliabilityPolicy.RELIABLE))
        self.pub = node.create_publisher(PointCloud2, "/color_cloud", 10)
        self.pub_pose = node.create_publisher(Odometry, "/color_pose", 10)
        self.pub_path = node.create_publisher(Path, "/color_path", 10)
        self.path = Path(); self.path.header.frame_id = "camera_init"

    def on_avia(self, buf):
        t, x = parse_avia(bytes(buf))
        if len(t):
            self.avia.append((t[0], t[-1], t, x))

    def on_mid(self, m):
        t, x = parse_mid(m)
        if len(t):
            self.mid.append((t[0], t[-1], t, x))

    def on_image(self, m):
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[..., :3]
        if m.encoding == "rgb8":
            im = im[..., ::-1]
        with self.img_lock:
            self.images.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 + a.img_offset, np.ascontiguousarray(im)))

    def read_traj(self):
        if not os.path.exists(a.traj):
            return
        with open(a.traj) as f:
            f.seek(self.traj_pos)
            data = f.read()
        lines = data.split("\n")
        tail = lines.pop()                                  # incomplete last line stays for next time
        self.traj_pos += len(data) - len(tail)
        rows = np.array([[float(x) for x in r.split()] for r in lines if len(r.split()) == 8]).reshape(-1, 8)
        if len(rows):
            self.tt = np.concatenate([self.tt, rows[:, 0]])
            self.tp = np.concatenate([self.tp, rows[:, 1:4]])
            self.tq = np.concatenate([self.tq, rows[:, 4:8]])

    def pose_at(self, t):
        return interp_pose(self.tt, self.tp, self.tq, t)

    def step(self):
        self.read_traj()
        if len(self.tt) < 2:
            return
        # A batch is one Mid-360 scan plus the Avia scans that start before it ends; it is
        # processed once the newest pose covers every point in it.
        while self.mid:
            m0, m1, mt, mx = self.mid[0]
            av = [s for s in self.avia if s[0] < m1]
            if max([m1] + [s[1] for s in av]) > self.tt[-1]:
                break
            self.mid.popleft()
            for _ in av:
                self.avia.popleft()
            if m0 < self.tt[0]:
                self.n_drop += 1                            # before the LIO started
                continue
            self.n_src["mid"] += len(mt)
            self.n_src["avia"] += sum(len(s[2]) for s in av)
            t = np.concatenate([mt] + [s[2] for s in av])
            x = np.concatenate([mx] + [s[3] for s in av])
            self.process(m0, m1, t, x)

    def process(self, m0, m1, t_pt, p_imu):
        Rw, pw, _ = self.pose_at(t_pt)
        xyz = np.einsum("nij,nj->ni", Rw, p_imu) + pw      # de-skewed world points
        self.n_batch += 1; self.n_pts += len(xyz)
        t_mid = 0.5 * (m0 + m1)
        _, pb, qb = self.pose_at(np.array([m1]))
        self.publish_pose(m1, pb[0], qb[0])

        with self.img_lock:
            images = list(self.images)
        if not images:
            return
        t_img, im = min(images, key=lambda x: abs(x[0] - t_mid))
        if abs(t_img - t_mid) > a.max_dt or t_img > self.tt[-1] or t_img < self.tt[0]:
            return
        Rc, pc, _ = self.pose_at(np.array([t_img]))
        Rwc = Rc[0] @ R_CI.T                                # camera orientation in world
        pwc = pc[0] - Rwc @ T_CI
        p_cam = (xyz - pwc) @ Rwc                           # Rwc^T (x - pwc)
        front = np.flatnonzero(p_cam[:, 2] > 0.05)
        if len(front) == 0:
            return
        uv, _ = cv2.projectPoints(p_cam[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)
        uv = uv.reshape(-1, 2)
        u = np.round(uv[:, 0] * im.shape[1] / CAM_W).astype(int)
        v = np.round(uv[:, 1] * im.shape[0] / CAM_H).astype(int)
        inside = (u >= 0) & (u < im.shape[1]) & (v >= 0) & (v < im.shape[0])
        idx = front[inside]
        xyz, bgr = xyz[idx], im[v[inside], u[inside]].astype(np.float64)
        self.n_col += len(idx)

        if a.raw:
            self.raw.append(to_pcd(xyz, bgr))
        else:
            self.map.add(xyz, bgr)
        if a.pub_leaf > 0:
            # publish only the first point of each pub-leaf voxel (keeps RViz from drowning)
            uk, first = np.unique(vkey(xyz, a.pub_leaf), return_index=True)
            new = np.ones(len(uk), bool)
            for lv in self.known:
                j = np.minimum(np.searchsorted(lv, uk), len(lv) - 1)
                new &= lv[j] != uk
            self.add_known(uk[new])
            sel = first[new]
        else:
            sel = slice(None)
        self.n_pub += len(xyz[sel])
        self.publish_cloud(xyz[sel], bgr[sel], m1)
        if self.n_batch % 200 == 0:
            self.node.get_logger().info(
                f"{self.n_batch} batches, t = {m1 - self.tt[0]:.0f} s, points avia {self.n_src['avia']} mid {self.n_src['mid']}, "
                f"{100 * self.n_col / self.n_pts:.0f} % coloured, "
                f"{sum(map(len, self.raw)) if a.raw else len(self.map.sums.k) + self.map.sums.npend} "
                f"{'points kept' if a.raw else 'finest cells'}, {self.n_pub} points published, "
                f"{len(self.mid)} mid / {len(self.avia)} avia scans queued")

    def add_known(self, k):
        if len(k) == 0:
            return
        self.known.append(k)                                # k is sorted (np.unique) and new
        while len(self.known) > 1 and len(self.known[-1]) * 2 >= len(self.known[-2]):
            b = self.known.pop(); c = self.known.pop()
            self.known.append(np.union1d(c, b))

    def publish_pose(self, t, p, q):
        o = Odometry()
        o.header.frame_id, o.child_frame_id = "camera_init", "mid360_imu"
        stamp(o.header.stamp, t)
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, p)
        (o.pose.pose.orientation.x, o.pose.pose.orientation.y,
         o.pose.pose.orientation.z, o.pose.pose.orientation.w) = map(float, q)
        self.pub_pose.publish(o)
        ps = PoseStamped(); ps.header = o.header; ps.pose = o.pose.pose
        self.path.poses.append(ps); self.path.header.stamp = o.header.stamp
        if self.n_batch % 10 == 0:
            self.pub_path.publish(self.path)

    def publish_cloud(self, xyz, bgr, t):
        if len(xyz) == 0:
            return
        c = to_pcd(xyz, bgr)
        m = PointCloud2()
        m.header.frame_id = "camera_init"
        stamp(m.header.stamp, t)
        m.height, m.width = 1, len(c)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        m.fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
        m.is_bigendian, m.point_step, m.is_dense = False, 16, True
        m.row_step = 16 * len(c)
        m.data = c.tobytes()
        self.pub.publish(m)

    def save(self):
        t0 = time.time()
        if a.raw:
            out = np.concatenate(self.raw) if self.raw else np.zeros(0, dtype=DT)
            how = "every coloured point"
        else:
            xyz, bgr, lev, _ = self.map.build()
            out = to_pcd(xyz, np.round(bgr))
            how = "multi-resolution: " + ", ".join(f"{a.min_leaf * 2 ** l * 100:g} cm {c}" for l, c in
                                                   enumerate(np.bincount(lev, minlength=self.map.L + 1))) + \
                  f"; split at {a.split_views} views"
        write_pcd(a.out, out)
        print(f"[color_mapper] {self.n_batch} batches, points avia {self.n_src['avia']} mid {self.n_src['mid']}, "
              f"{100 * self.n_col / max(self.n_pts, 1):.1f} % coloured, {self.n_drop} scans before the first pose, "
              f"{self.n_pub} points published (pub-leaf {a.pub_leaf} m), "
              f"{len(self.mid)} mid / {len(self.avia)} avia scans left -> {len(out)} points ({how}; "
              f"{time.time() - t0:.0f} s) -> {a.out}", flush=True)


rclpy.init(args=None, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
node = rclpy.create_node("color_mapper")
mapper = ColorMapper(node)
stop = []
signal.signal(signal.SIGINT, lambda *_: stop.append(1))
signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
# Callbacks only buffer messages, on their own thread, so a long batch or a big accumulator
# merge here never stalls reception (the subscriptions keep 50 scans / 30 images).
executor = rclpy.executors.SingleThreadedExecutor()
executor.add_node(node)
spin = threading.Thread(target=executor.spin, daemon=True)
spin.start()
while not stop:
    mapper.step()
    time.sleep(0.02)
executor.shutdown(timeout_sec=1.0)
mapper.step()                                           # process what the final poses cover
mapper.save()
node.destroy_node()
rclpy.shutdown()
