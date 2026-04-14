#!/usr/bin/env python3
"""
ROS1 -> Rerun bridge for Voxblox (cow_and_lady).

Renders 5 views:
    1. Camera frustum       (slam/pose/cam, Pinhole)
    2. Camera trajectory    (slam/trajectory, LineStrips3D)
    3. Local map            (slam/local_map, per-frame surface cloud)
    4. Global map           (slam/global_map, accumulated TSDF cloud)
    5. RGB + Depth image    (slam/pose/cam/rgb, slam/pose/cam/depth)
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Empty
from rospy import AnyMsg
import tf.transformations as tft
import struct


# Kinect v1 RGB intrinsics (cow_and_lady)
KINECT_FX, KINECT_FY = 525.0, 525.0
KINECT_CX, KINECT_CY = 319.5, 239.5
KINECT_W,  KINECT_H  = 640, 480

# --- Relative time helper (bag timestamps from 2016 look weird in Rerun) ---
_t0 = [None]
def _rel_time(stamp):
    ts = stamp.to_sec()
    if _t0[0] is None:
        _t0[0] = ts
    return ts - _t0[0]


# Vicon -> Camera extrinsic from voxblox/voxblox_ros/cfg/cow_and_lady.yaml
# (yaml comment says: "actually T_V_C, C=cam0, V=vicon"). This places the
# Kinect optical frame at the correct pose relative to the Vicon marker.
_T_V_C = np.array([
    [ 0.971048, -0.120915,  0.206023,  0.00114049],
    [ 0.15701,   0.973037, -0.168959,  0.0450936 ],
    [-0.180038,  0.196415,  0.96385,   0.0430765 ],
    [ 0.0,       0.0,       0.0,       1.0       ],
], dtype=np.float64)
_T_V_C_TRANS = _T_V_C[:3, 3].tolist()
_T_V_C_QUAT  = tft.quaternion_from_matrix(_T_V_C)  # xyzw


# ---------------------------------------------------------------------------
# Pose + trajectory
# ---------------------------------------------------------------------------

_trajectory: list = []

def transform_cb(msg: TransformStamped):
    t = msg.transform.translation
    q = msg.transform.rotation
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))

    # Raw Vicon pose (world <- vicon body). The fixed vicon->cam extrinsic
    # lives on slam/pose/cam as a static child transform.
    rr.log("slam/pose", rr.Transform3D(
        translation=[t.x, t.y, t.z],
        quaternion=[q.x, q.y, q.z, q.w],
    ))

    pos = [t.x, t.y, t.z]
    _trajectory.append(pos)
    if len(_trajectory) % 5 == 0 and len(_trajectory) >= 10:
        pts = np.array(_trajectory[::5], dtype=np.float32)
        rr.log("slam/trajectory", rr.LineStrips3D(
            [pts], colors=[[0, 200, 255]],
        ))


# ---------------------------------------------------------------------------
# Current frame: RGB + depth reconstructed from organized depth_registered/points
# ---------------------------------------------------------------------------

_depth_frame_i = [0]

def depth_pc_cb(msg: PointCloud2):
    """Decode RGB + depth + local 3D cloud from the organized XYZRGB
    pointcloud. rospy's Python PointCloud2 deserializer itself is the
    bottleneck (~4 Hz max on aarch64), so any per-call optimization
    barely matters — just do the work every message."""
    _depth_frame_i[0] += 1
    if _depth_frame_i[0] % 10 == 0:
        rospy.loginfo(f"[bridge] depth_pc_cb #{_depth_frame_i[0]}")
    if msg.height < 2 or msg.width < 2:
        return
    rgb_off = 16   # fixed for this dataset

    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))

    h, w = msg.height, msg.width
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, msg.point_step)
    # Slices of a point_step=32 buffer are non-contiguous: force a copy
    # before calling .view() or it raises ValueError (silently caught by
    # rospy, leaves the image frozen).
    xyz = np.ascontiguousarray(raw[..., 0:12]).view(np.float32).reshape(h, w, 3)
    z = xyz[..., 2].copy()
    np.nan_to_num(z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_u32 = np.ascontiguousarray(raw[..., rgb_off:rgb_off + 4]).view(np.uint32).reshape(h, w)
    r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_u32 >> 8)  & 0xFF).astype(np.uint8)
    b = ( rgb_u32        & 0xFF).astype(np.uint8)
    rgb_img = np.dstack([r, g, b])

    rr.log("slam/pose/cam/rgb",   rr.Image(rgb_img))
    rr.log("slam/pose/cam/depth", rr.DepthImage(z, meter=1.0))

    # Local map: aggressive stride (~10k points) so the Points3D log is cheap.
    pts = xyz[::8, ::8].reshape(-1, 3)
    cols = rgb_img[::8, ::8].reshape(-1, 3)
    finite_z = (pts[:, 2] > 0.1) & (pts[:, 2] < 5.0)
    if finite_z.any():
        rr.log("slam/pose/cam/local_map",
               rr.Points3D(pts[finite_z], colors=cols[finite_z], radii=0.01))


# ---------------------------------------------------------------------------
# Global map: voxblox /surface_pointcloud is the zero-crossing extraction of
# the fused TSDF, i.e. the optimized reconstructed surface of the whole map.
# This is what voxblox considers the "map", not the raw voxel distance cloud.
# ---------------------------------------------------------------------------

def _height_colors(points: np.ndarray) -> np.ndarray:
    h = points[:, 2]
    lo, hi = float(h.min()), float(h.max())
    n = (h - lo) / max(hi - lo, 1e-6)
    r = (n * 255).astype(np.uint8)
    g = ((1 - n) * 128).astype(np.uint8)
    b = ((1 - n) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


_VOXEL_SIZE = 0.05   # must match /voxblox_node/tsdf_voxel_size

def surface_cloud_cb(msg: PointCloud2):
    """voxblox surface_pointcloud: one point per surface voxel, with
    integrated per-voxel color in the `rgb` field (offset 16, float32
    storage of packed uint32). Decode directly from the raw buffer for
    speed and render with radii == voxel_size/2 so the points look like
    cubes ("voxelized")."""
    n = msg.width * msg.height
    if n == 0:
        return
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    pts = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    rgb = raw[:, 16:20].copy().view(np.uint32).reshape(n)
    r = ((rgb >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb >>  8) & 0xFF).astype(np.uint8)
    b = ( rgb        & 0xFF).astype(np.uint8)
    colors = np.stack([r, g, b], axis=1)
    rr.log("slam/global_map", rr.Points3D(
        pts, colors=colors, radii=_VOXEL_SIZE / 2.0,
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D",
                origin="/",
                contents=[
                    "+ $origin/**",
                ],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="RGB",   origin="slam/pose/cam/rgb"),
                rrb.Spatial2DView(name="Depth", origin="slam/pose/cam/depth"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def main():
    rospy.init_node("voxblox_rerun_bridge", anonymous=True)

    rr.init("voxblox")
    # Connect to an out-of-process `rerun --serve-web` running on 127.0.0.1:9876.
    # This replaces in-process serve_web(): log calls become non-blocking TCP
    # sends, so the viewer's serialization can't back-pressure the ROS
    # callback thread.
    rr.connect_tcp("127.0.0.1:9876", default_blueprint=_build_blueprint())
    rospy.loginfo(
        "Bridge connected to rerun TCP sink at 127.0.0.1:9876. "
        "Open http://localhost:9090/?url=ws://localhost:9877"
    )

    # Static world axes
    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    # Static Vicon->Cam extrinsic (T_V_C from cow_and_lady.yaml), logged as
    # a child transform under slam/pose so /cam sits at the optical frame.
    rr.log("slam/pose/cam", rr.Transform3D(
        translation=_T_V_C_TRANS,
        quaternion=_T_V_C_QUAT,
    ), static=True)
    # Static pinhole. The child frame /cam IS the kinect optical frame
    # (x-right, y-down, z-forward), so camera_xyz=RDF is correct.
    rr.log("slam/pose/cam", rr.Pinhole(
        focal_length=[KINECT_FX, KINECT_FY],
        principal_point=[KINECT_CX, KINECT_CY],
        resolution=[KINECT_W, KINECT_H],
        image_plane_distance=0.3,
        camera_xyz=rr.ViewCoordinates.RDF,
    ), static=True)

    # Pose (cow_and_lady Vicon ground truth)
    rospy.Subscriber(
        "/kinect/vrpn_client/estimated_transform",
        TransformStamped, transform_cb, queue_size=50,
    )
    # Current RGB + depth + local-map points. buff_size=16 MB is required
    # because the default 64 KB silently drops the 9.4 MB pointcloud
    # messages; rospy's Python PointCloud2 deserializer is still the
    # rate-limiter (~4 Hz max on aarch64), but at least messages flow.
    rospy.Subscriber(
        "/camera/depth_registered/points",
        PointCloud2, depth_pc_cb,
        queue_size=1, tcp_nodelay=True, buff_size=2**24,
    )
    # Global optimized map = voxblox surface pointcloud (fused TSDF
    # zero-crossing over the whole integrated volume).
    rospy.Subscriber(
        "/voxblox_node/surface_pointcloud",
        PointCloud2, surface_cloud_cb, queue_size=2,
    )

    rospy.loginfo(
        "Subscribed:\n"
        "  /kinect/vrpn_client/estimated_transform -> slam/pose\n"
        "  /camera/depth_registered/points         -> slam/pose/cam/{rgb,depth,local_map}\n"
        "  /voxblox_node/surface_pointcloud        -> slam/global_map"
    )

    # Voxblox only emits tsdf/surface pointclouds when asked. The
    # cow_and_lady launch file uses clear_params=true so we can't set
    # publish_pointclouds_on_update as a pre-launch param. Instead, wait
    # for the service and then poke it on a timer.
    rospy.loginfo("Waiting for /voxblox_node/publish_pointclouds service...")
    rospy.wait_for_service("/voxblox_node/publish_pointclouds")
    _publish_clouds = rospy.ServiceProxy(
        "/voxblox_node/publish_pointclouds", Empty,
    )
    _pub_counter = [0]

    def _trigger_clouds(_evt):
        try:
            _publish_clouds()
            _pub_counter[0] += 1
            if _pub_counter[0] <= 3 or _pub_counter[0] % 10 == 0:
                rospy.loginfo(
                    f"[bridge] publish_pointclouds call #{_pub_counter[0]}"
                )
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(10, f"publish_pointclouds failed: {e}")

    rospy.Timer(rospy.Duration(1.0), _trigger_clouds)

    rospy.spin()


if __name__ == "__main__":
    main()
