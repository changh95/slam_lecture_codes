#!/usr/bin/env python3
"""
ROS1 -> Rerun bridge for Voxblox.

Subscribes to Voxblox's published topics and forwards them to a Rerun
gRPC server. The Rerun web viewer is served on port 9090 so you can view
the mapping output live in a browser.

Topics consumed:
    /voxblox_node/tsdf_pointcloud   (sensor_msgs/PointCloud2) -> slam/tsdf_cloud
    /voxblox_node/esdf_pointcloud   (sensor_msgs/PointCloud2) -> slam/esdf_cloud
    /voxblox_node/surface_pointcloud (sensor_msgs/PointCloud2) -> slam/surface_cloud
    /voxblox_node/mesh              (voxblox_msgs/Mesh)        -> slam/mesh (as Points3D)

Usage:
    python3 ros_rerun_bridge.py
"""

import numpy as np
import rerun as rr
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

# voxblox_msgs/Mesh is only available if voxblox is built; fall back gracefully
try:
    from voxblox_msgs.msg import Mesh as VoxbloxMesh
    _HAVE_MESH_MSG = True
except ImportError:
    _HAVE_MESH_MSG = False
    rospy.logwarn_once(
        "voxblox_msgs not importable; /voxblox_node/mesh will not be subscribed."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAP_VOXEL_SIZE = 0.1   # metres – voxel dedup grid
_MAP_MAX_POINTS = 3_000_000
_MAP_LOG_EVERY  = 5     # re-log accumulated map every N scans

_tsdf_pts: list  = []
_tsdf_count: int = 0
_esdf_pts: list  = []
_esdf_count: int = 0


def _voxel_dedup(points: np.ndarray, voxel: float) -> np.ndarray:
    """One point per occupied voxel (fast hash dedup)."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    k = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, idx = np.unique(k, return_index=True)
    return points[idx]


def _height_colors(points: np.ndarray) -> np.ndarray:
    """Colour points by height (z) using a blue-to-red ramp."""
    h = points[:, 2]
    h_norm = (h - h.min()) / max(h.max() - h.min(), 1e-6)
    r = (h_norm * 255).astype(np.uint8)
    g = ((1 - h_norm) * 128).astype(np.uint8)
    b = ((1 - h_norm) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def tsdf_cloud_cb(msg: PointCloud2):
    global _tsdf_pts, _tsdf_count

    pts = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )
    if len(pts) == 0:
        return

    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    rr.log("slam/tsdf_cloud", rr.Points3D(pts, colors=_height_colors(pts), radii=0.04))

    # Accumulate global TSDF map
    _tsdf_pts.append(pts[::10])  # keep 1/10 for map
    _tsdf_count += 1
    if _tsdf_count % _MAP_LOG_EVERY == 0:
        all_pts = np.concatenate(_tsdf_pts, axis=0)
        all_pts = _voxel_dedup(all_pts, _MAP_VOXEL_SIZE)
        if len(all_pts) > _MAP_MAX_POINTS:
            idx = np.random.choice(len(all_pts), _MAP_MAX_POINTS, replace=False)
            all_pts = all_pts[idx]
        colors = _height_colors(all_pts)
        rr.log("slam/tsdf_map", rr.Points3D(all_pts, colors=colors, radii=0.05))
        _tsdf_pts = [all_pts]


def esdf_cloud_cb(msg: PointCloud2):
    global _esdf_pts, _esdf_count

    pts = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )
    if len(pts) == 0:
        return

    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    # ESDF: colour by distance-to-surface stored in intensity field if present
    try:
        pts_d = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"),
                                 skip_nans=True)),
            dtype=np.float32,
        )
        dist = pts_d[:, 3]
        d_norm = np.clip(dist / (dist.max() + 1e-6), 0, 1)
        colors = np.stack([
            (d_norm * 255).astype(np.uint8),
            ((1 - d_norm) * 200).astype(np.uint8),
            np.zeros(len(pts), dtype=np.uint8),
        ], axis=1)
    except Exception:
        colors = _height_colors(pts)

    rr.log("slam/esdf_cloud", rr.Points3D(pts, colors=colors, radii=0.04))

    _esdf_pts.append(pts[::10])
    _esdf_count += 1
    if _esdf_count % _MAP_LOG_EVERY == 0:
        all_pts = np.concatenate(_esdf_pts, axis=0)
        all_pts = _voxel_dedup(all_pts, _MAP_VOXEL_SIZE)
        rr.log("slam/esdf_map", rr.Points3D(all_pts, colors=_height_colors(all_pts), radii=0.05))
        _esdf_pts = [all_pts]


def surface_cloud_cb(msg: PointCloud2):
    pts = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )
    if len(pts) == 0:
        return
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    # Surface cloud: white/grey
    colors = np.full((len(pts), 3), 200, dtype=np.uint8)
    rr.log("slam/surface_cloud", rr.Points3D(pts, colors=colors, radii=0.03))


def mesh_cb(msg):
    """
    voxblox_msgs/Mesh contains per-block triangle meshes.
    Extract vertex positions and log as Points3D (triangles would need
    rr.Mesh3D which requires index arrays; vertex cloud is a lightweight proxy).
    """
    try:
        all_verts = []
        for block in msg.mesh_blocks:
            if len(block.x) == 0:
                continue
            verts = np.stack(
                [np.array(block.x, dtype=np.float32),
                 np.array(block.y, dtype=np.float32),
                 np.array(block.z, dtype=np.float32)],
                axis=1,
            )
            all_verts.append(verts)
        if not all_verts:
            return
        verts = np.concatenate(all_verts, axis=0)
        rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
        colors = np.full((len(verts), 3), [180, 180, 200], dtype=np.uint8)
        rr.log("slam/mesh", rr.Points3D(verts, colors=colors, radii=0.02))
    except Exception as e:
        rospy.logwarn_throttle(10, f"mesh_cb error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rospy.init_node("voxblox_rerun_bridge", anonymous=True)

    rr.init("voxblox")
    # rerun 0.21: single call serves both web viewer (HTTP) and WebSocket data channel
    rr.serve_web(open_browser=False, web_port=9090, ws_port=9877)
    rospy.loginfo("Rerun web viewer ready at http://localhost:9090/?url=ws://localhost:9877")

    # World coordinate axes (static)
    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    rospy.Subscriber(
        "/voxblox_node/tsdf_pointcloud", PointCloud2, tsdf_cloud_cb, queue_size=5
    )
    rospy.Subscriber(
        "/voxblox_node/esdf_pointcloud", PointCloud2, esdf_cloud_cb, queue_size=5
    )
    rospy.Subscriber(
        "/voxblox_node/surface_pointcloud", PointCloud2, surface_cloud_cb, queue_size=5
    )

    if _HAVE_MESH_MSG:
        rospy.Subscriber("/voxblox_node/mesh", VoxbloxMesh, mesh_cb, queue_size=2)
        rospy.loginfo(
            "Subscribed to /voxblox_node/{tsdf_pointcloud,esdf_pointcloud,"
            "surface_pointcloud,mesh}"
        )
    else:
        rospy.loginfo(
            "Subscribed to /voxblox_node/{tsdf_pointcloud,esdf_pointcloud,"
            "surface_pointcloud} (mesh skipped: voxblox_msgs not available)"
        )

    rospy.spin()


if __name__ == "__main__":
    main()
