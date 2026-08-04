#!/usr/bin/env python3
"""Log FAST-LIO2's /Odometry to a TUM-format trajectory file.

TUM format: timestamp tx ty tz qx qy qz qw  (one pose per line, space separated).
Usage: rosrun-free -> python3 odom_to_tum.py /out/traj_tum.txt [odom_topic]
"""
import sys
import rospy
from nav_msgs.msg import Odometry

out_path = sys.argv[1] if len(sys.argv) > 1 else "/out/fastlio_traj_tum.txt"
topic = sys.argv[2] if len(sys.argv) > 2 else "/Odometry"

fh = open(out_path, "w", buffering=1)
count = [0]


def cb(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    fh.write("%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"
             % (msg.header.stamp.to_sec(), p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    count[0] += 1


def shutdown():
    fh.flush()
    fh.close()
    rospy.loginfo("odom_to_tum: wrote %d poses to %s", count[0], out_path)


rospy.init_node("odom_to_tum", anonymous=True, disable_signals=False)
rospy.Subscriber(topic, Odometry, cb, queue_size=2000)
rospy.on_shutdown(shutdown)
rospy.loginfo("odom_to_tum: logging %s -> %s", topic, out_path)
rospy.spin()
