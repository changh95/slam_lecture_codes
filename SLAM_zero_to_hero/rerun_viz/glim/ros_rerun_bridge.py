#!/usr/bin/env python3
"""
ROS1 -> Rerun bridge for GLIM.

Subscribes to GLIM's rviz_viewer extension output topics and forwards them
to a Rerun gRPC server. The Rerun web viewer is served on port 9090.

Topics consumed:
    /glim_ros/odom    (nav_msgs/Odometry)       -> slam/pose
    /glim_ros/points  (sensor_msgs/PointCloud2) -> slam/cloud (current frame)
    /glim_ros/map     (sensor_msgs/PointCloud2) -> slam/map   (accumulated)

Usage:
    python3 ros_rerun_bridge.py
"""

import numpy as np
import rerun as rr
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
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
    rr.log("slam/pose/body", rr.Boxes3D(centers=[[0, 0, 0]], sizes=[[0.3, 0.3, 0.1]]))


def points_cb(msg: PointCloud2):
    points = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )
    if len(points) == 0:
        return
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    rr.log("slam/cloud", rr.Points3D(points, colors=[[255, 200, 0]], radii=0.03))


def map_cb(msg: PointCloud2):
    points = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )
    if len(points) == 0:
        return
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    heights = points[:, 2]
    h_min, h_max = heights.min(), heights.max()
    h_norm = (heights - h_min) / max(h_max - h_min, 1e-6)
    colors = (np.stack(
        [h_norm * 255, (1 - h_norm) * 200, (1 - h_norm) * 255], axis=1
    )).astype(np.uint8)
    rr.log("slam/map", rr.Points3D(points, colors=colors, radii=0.05))


def main():
    rospy.init_node("glim_rerun_bridge", anonymous=True)

    rr.init("glim")
    # rerun 0.21: single call serves both web viewer (HTTP) and WebSocket data channel
    rr.serve_web(open_browser=False, web_port=9090, ws_port=9877)
    rospy.loginfo("Rerun web viewer ready at http://localhost:9090/?url=ws://localhost:9877")

    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    rospy.Subscriber("/glim_ros/odom", Odometry, odom_cb, queue_size=10)
    rospy.Subscriber("/glim_ros/points", PointCloud2, points_cb, queue_size=5)
    rospy.Subscriber("/glim_ros/map", PointCloud2, map_cb, queue_size=2)

    rospy.loginfo("Subscribed to /glim_ros/odom, /glim_ros/points, /glim_ros/map")
    rospy.spin()


if __name__ == "__main__":
    main()
