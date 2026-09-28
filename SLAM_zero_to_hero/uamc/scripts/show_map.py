#!/usr/bin/env python3
"""Publish a finished FAST-LIVO2 run for RViz: the coloured map (pcd/map.pcd) on /map_cloud
and the trajectory (TUM file) on /map_path, both latched, in frame camera_init.
--zmax drops points above a height, so a top view is not just the ceiling."""
import argparse
import os
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcd_io import read_pcd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pcd")
ap.add_argument("traj")
ap.add_argument("--frame", default="camera_init")
ap.add_argument("--zmax", type=float, default=None, help="drop map points above this z (cut the ceiling)")
a = ap.parse_args()

rclpy.init()
node = rclpy.create_node("show_map")
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 reliability=ReliabilityPolicy.RELIABLE)
stamp = node.get_clock().now().to_msg()

pts = read_pcd(a.pcd)
if a.zmax is not None:
    pts = pts[pts["z"] <= a.zmax]
cloud = PointCloud2()
cloud.header.frame_id, cloud.header.stamp = a.frame, stamp
cloud.height, cloud.width = 1, len(pts)
cloud.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate("xyz")]
cloud.fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
cloud.is_bigendian, cloud.point_step, cloud.is_dense = False, 16, True
cloud.row_step = 16 * len(pts)
cloud.data = pts.tobytes()

path = Path()
path.header.frame_id, path.header.stamp = a.frame, stamp
for row in np.loadtxt(a.traj, ndmin=2):
    ps = PoseStamped()
    ps.header = path.header
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, row[1:4])
    (ps.pose.orientation.x, ps.pose.orientation.y,
     ps.pose.orientation.z, ps.pose.orientation.w) = map(float, row[4:8])
    path.poses.append(ps)

node.create_publisher(PointCloud2, "/map_cloud", qos).publish(cloud)
node.create_publisher(Path, "/map_path", qos).publish(path)
node.get_logger().info(f"published {len(pts)} map points and {len(path.poses)} poses")
rclpy.spin(node)
