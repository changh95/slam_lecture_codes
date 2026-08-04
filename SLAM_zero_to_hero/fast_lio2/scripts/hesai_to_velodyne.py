#!/usr/bin/env python3
"""Republish Hilti's /hesai/pandar PointCloud2 in the layout FAST-LIO's
Velodyne branch actually wants.

Hilti PandarXT-32 cloud (point_step 48):
    x f4@0, y f4@4, z f4@8, intensity f4@16, timestamp f8@24 (ABSOLUTE unix), ring u2@32

FAST-LIO velodyne_ros::Point wants a *relative* float32 field named `time`
(plus x/y/z/intensity/ring). Without it, given_offset_time=false and FAST-LIO
reconstructs intra-scan time from azimuth. This relay hands it the real
per-point time so the IMU de-skewing uses measured timestamps.

Output topic /velodyne_points (point_step 24):
    x f4@0, y f4@4, z f4@8, intensity f4@12, time f4@16, ring u2@20
=> use lid_topic:"/velodyne_points" and timestamp_unit: 0 (seconds).
"""
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField

IN_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity", "timestamp", "ring"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<f8", "<u2"],
    "offsets": [0, 4, 8, 16, 24, 32],
    "itemsize": 48,
})

OUT_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity", "time", "ring"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
    "offsets": [0, 4, 8, 12, 16, 20],
    "itemsize": 24,
})

OUT_FIELDS = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
    PointField("time", 16, PointField.FLOAT32, 1),
    PointField("ring", 20, PointField.UINT16, 1),
]

pub = None
stats = {"n": 0}


def cb(msg):
    src = np.frombuffer(msg.data, dtype=IN_DTYPE, count=msg.width * msg.height)
    dst = np.empty(src.shape[0], dtype=OUT_DTYPE)
    for f in ("x", "y", "z", "intensity", "ring"):
        dst[f] = src[f]
    t0 = msg.header.stamp.to_sec()
    dst["time"] = (src["timestamp"] - t0).astype("<f4")

    out = PointCloud2()
    out.header = msg.header
    out.height = 1
    out.width = dst.shape[0]
    out.fields = OUT_FIELDS
    out.is_bigendian = False
    out.point_step = OUT_DTYPE.itemsize
    out.row_step = OUT_DTYPE.itemsize * dst.shape[0]
    out.is_dense = True
    out.data = dst.tobytes()
    pub.publish(out)

    stats["n"] += 1
    if stats["n"] % 100 == 1:
        rospy.loginfo("relay %d: %d pts, time offset span %.6f..%.6f s",
                      stats["n"], dst.shape[0],
                      float(dst["time"].min()), float(dst["time"].max()))


rospy.init_node("hesai_to_velodyne")
pub = rospy.Publisher("/velodyne_points", PointCloud2, queue_size=20)
rospy.Subscriber("/hesai/pandar", PointCloud2, cb, queue_size=20,
                 buff_size=2 ** 26)
rospy.loginfo("hesai_to_velodyne: /hesai/pandar -> /velodyne_points")
rospy.spin()
