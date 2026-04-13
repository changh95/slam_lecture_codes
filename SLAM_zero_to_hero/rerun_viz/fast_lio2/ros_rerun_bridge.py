#!/usr/bin/env python3
"""
ROS1 -> Rerun bridge for FAST-LIO2.

Subscribes to FAST-LIO2's published topics and forwards them to a Rerun
gRPC server. The Rerun web viewer is served on port 9090 so you can view
the SLAM output live in a browser.

Topics consumed:
    /Odometry          (nav_msgs/Odometry)       -> slam/pose
    /cloud_registered  (sensor_msgs/PointCloud2) -> slam/cloud
    /path              (nav_msgs/Path)           -> slam/trajectory

Usage:
    python3 ros_rerun_bridge.py
"""

import numpy as np
import rerun as rr
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2


def odom_cb(msg: Odometry):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    rr.log(
        "slam/pose",
        rr.Transform3D(
            translation=[p.x, p.y, p.z],
            quaternion=[q.x, q.y, q.z, q.w],
        ),
    )
    # Sensor box marker at the pose
    rr.log("slam/pose/body", rr.Boxes3D(centers=[[0, 0, 0]], sizes=[[0.3, 0.3, 0.1]]))


_map_points = []          # accumulated voxelized map
_map_frame_count = 0
MAP_DOWNSAMPLE = 30       # keep 1/N points from each scan
MAP_LOG_EVERY = 5         # re-log accumulated map every N scans
MAP_VOXEL_SIZE = 0.2      # voxel size for dedup (meters)
MAP_MAX_POINTS = 2_000_000


def _voxel_dedup(points: np.ndarray, voxel: float) -> np.ndarray:
    """Downsample with a voxel grid: one point per occupied voxel."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    # Pack 3D key into 1D for uniqueness check
    k = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, idx = np.unique(k, return_index=True)
    return points[idx]


def cloud_cb(msg: PointCloud2):
    global _map_points, _map_frame_count

    points = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )
    if len(points) == 0:
        return
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())

    # Current-scan visualization (replaces each frame)
    heights = points[:, 2]
    h_min, h_max = heights.min(), heights.max()
    h_norm = (heights - h_min) / max(h_max - h_min, 1e-6)
    colors = (np.stack(
        [h_norm * 255, (1 - h_norm) * 128, (1 - h_norm) * 255], axis=1
    )).astype(np.uint8)
    rr.log("slam/cloud", rr.Points3D(points, colors=colors, radii=0.03))

    # Accumulated map: downsample, append, dedup
    sampled = points[::MAP_DOWNSAMPLE]
    if len(sampled):
        _map_points.append(sampled)
        _map_frame_count += 1
        if _map_frame_count % MAP_LOG_EVERY == 0:
            all_pts = np.concatenate(_map_points, axis=0)
            all_pts = _voxel_dedup(all_pts, MAP_VOXEL_SIZE)
            if len(all_pts) > MAP_MAX_POINTS:
                idx = np.random.choice(len(all_pts), MAP_MAX_POINTS, replace=False)
                all_pts = all_pts[idx]
            # Re-color by height
            h = all_pts[:, 2]
            hn = (h - h.min()) / max(h.max() - h.min(), 1e-6)
            map_colors = (np.stack(
                [hn * 180, (1 - hn) * 180, (1 - hn) * 255], axis=1
            )).astype(np.uint8)
            rr.log("slam/map", rr.Points3D(all_pts, colors=map_colors, radii=0.05))
            # Replace buffer with deduped version to cap memory
            _map_points = [all_pts]


def path_cb(msg: Path):
    if not msg.poses:
        return
    pts = np.array(
        [[ps.pose.position.x, ps.pose.position.y, ps.pose.position.z] for ps in msg.poses],
        dtype=np.float32,
    )
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    rr.log("slam/trajectory", rr.LineStrips3D([pts], colors=[[0, 255, 0]]))


def main():
    rospy.init_node("fast_lio2_rerun_bridge", anonymous=True)

    rr.init("fast_lio2")
    # rerun 0.21: single call serves both web viewer (HTTP) and WebSocket data channel
    rr.serve_web(open_browser=False, web_port=9090, ws_port=9877)
    rospy.loginfo("Rerun web viewer ready at http://localhost:9090/?url=ws://localhost:9877")

    # World coordinate axes
    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    rospy.Subscriber("/Odometry", Odometry, odom_cb, queue_size=10)
    rospy.Subscriber("/cloud_registered", PointCloud2, cloud_cb, queue_size=5)
    rospy.Subscriber("/path", Path, path_cb, queue_size=5)

    rospy.loginfo("Subscribed to /Odometry, /cloud_registered, /path")
    rospy.spin()


if __name__ == "__main__":
    main()
