#!/usr/bin/env python3
"""Counts what Kimera-Semantics consumed/produced during a run and writes
/out/run_stats.txt on shutdown: GT path length, frames seen, labelled clouds,
mesh updates, and the size of the last published mesh."""
import math
import sys

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from voxblox_msgs.msg import Mesh

out = sys.argv[1] if len(sys.argv) > 1 else "/out/run_stats.txt"
s = dict(depth=0, clouds=0, meshes=0, blocks=0, path=0.0, last=None, t0=None, t1=None)


def odom(m):
    p = m.pose.pose.position
    if s["last"] is not None:
        s["path"] += math.dist((p.x, p.y, p.z), s["last"])
    s["last"] = (p.x, p.y, p.z)
    t = m.header.stamp.to_sec()
    s["t0"] = t if s["t0"] is None else s["t0"]
    s["t1"] = t


def depth(_):
    s["depth"] += 1


def cloud(_):
    s["clouds"] += 1


def mesh(m):
    s["meshes"] += 1
    s["blocks"] = max(s["blocks"], len(m.mesh_blocks))


def dump():
    with open(out, "w") as f:
        f.write(f"gt_duration_s {0 if s['t0'] is None else s['t1'] - s['t0']:.1f}\n")
        f.write(f"gt_path_length_m {s['path']:.2f}\n")
        f.write(f"depth_frames {s['depth']}\n")
        f.write(f"semantic_pointclouds {s['clouds']}\n")
        f.write(f"mesh_messages {s['meshes']}\n")
        f.write(f"max_mesh_blocks_in_update {s['blocks']}\n")


rospy.init_node("kimera_run_monitor")
rospy.Subscriber("/tesse/odom", Odometry, odom, queue_size=1000)
rospy.Subscriber("/tesse/depth_cam/mono/image_raw", Image, depth, queue_size=50)
rospy.Subscriber("/semantic_pointcloud", PointCloud2, cloud, queue_size=50)
rospy.Subscriber("/kimera_semantics_node/mesh", Mesh, mesh, queue_size=10)
rospy.on_shutdown(dump)
rospy.Timer(rospy.Duration(5.0), lambda _: dump())
rospy.spin()
