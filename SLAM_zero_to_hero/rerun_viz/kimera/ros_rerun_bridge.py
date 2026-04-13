#!/usr/bin/env python3
"""
ROS1 -> Rerun bridge for Kimera-VIO (Kimera-VIO-ROS / standalone).

Subscribes to Kimera-VIO-ROS published topics and forwards them to a Rerun
gRPC server. The Rerun web viewer is served on port 9090 so you can view
the VIO output live in a browser.

Topics consumed (Kimera-VIO-ROS defaults):
    /kimera_vio_ros/odometry            (nav_msgs/Odometry)       -> slam/pose
    /kimera_vio_ros/imu_odometry        (nav_msgs/Odometry)       -> slam/imu_pose
    /kimera_vio_ros/optimized_odometry  (nav_msgs/Odometry)       -> slam/optimized_pose
    /kimera_vio_ros/frontend/feature_tracks (sensor_msgs/Image)   -> slam/feature_tracks
    /kimera_vio_ros/mesh                (pcl_msgs/PolygonMesh or
                                         visualization_msgs/Marker) -> slam/mesh (best-effort)
    /kimera_vio_ros/path                (nav_msgs/Path)           -> slam/trajectory
    /kimera_vio_ros/optimized_path      (nav_msgs/Path)           -> slam/optimized_trajectory

Fallback topics (Kimera-VIO standalone with ROS bridge):
    /odometry                           -> slam/pose
    /path                               -> slam/trajectory

Usage:
    python3 ros_rerun_bridge.py
    # Then open http://localhost:9090/?url=ws://localhost:9877 in a browser.
"""

import numpy as np
import rerun as rr
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image

# Optional: mesh support (pcl_msgs may not be installed)
try:
    from visualization_msgs.msg import Marker
    HAS_MARKER = True
except ImportError:
    HAS_MARKER = False

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
    _cv_bridge = CvBridge()
except ImportError:
    HAS_CV_BRIDGE = False


# ---------------------------------------------------------------------------
# Odometry callbacks
# ---------------------------------------------------------------------------

def _log_odometry(msg: Odometry, entity: str):
    """Log a nav_msgs/Odometry message to Rerun."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    rr.log(
        entity,
        rr.Transform3D(
            translation=[p.x, p.y, p.z],
            quaternion=[q.x, q.y, q.z, q.w],
        ),
    )
    # Small box as a sensor body marker
    rr.log(f"{entity}/body", rr.Boxes3D(centers=[[0, 0, 0]], sizes=[[0.2, 0.1, 0.1]]))


def odom_cb(msg: Odometry):
    _log_odometry(msg, "slam/pose")


def imu_odom_cb(msg: Odometry):
    _log_odometry(msg, "slam/imu_pose")


def optimized_odom_cb(msg: Odometry):
    _log_odometry(msg, "slam/optimized_pose")


# ---------------------------------------------------------------------------
# Path callbacks
# ---------------------------------------------------------------------------

def _log_path(msg: Path, entity: str, color):
    if not msg.poses:
        return
    pts = np.array(
        [[ps.pose.position.x, ps.pose.position.y, ps.pose.position.z]
         for ps in msg.poses],
        dtype=np.float32,
    )
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    rr.log(entity, rr.LineStrips3D([pts], colors=[color]))


def path_cb(msg: Path):
    _log_path(msg, "slam/trajectory", [0, 200, 255])


def optimized_path_cb(msg: Path):
    _log_path(msg, "slam/optimized_trajectory", [0, 255, 100])


# ---------------------------------------------------------------------------
# Feature-track image callback
# ---------------------------------------------------------------------------

def feature_tracks_cb(msg: Image):
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    if HAS_CV_BRIDGE:
        try:
            cv_img = _cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            rr.log("slam/feature_tracks", rr.Image(cv_img))
            return
        except Exception:
            pass
    # Fallback: raw bytes as tensor
    h, w = msg.height, msg.width
    channels = len(msg.data) // (h * w) if h * w > 0 else 3
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, channels)
    rr.log("slam/feature_tracks", rr.Image(arr))


# ---------------------------------------------------------------------------
# Mesh / Marker callback (best-effort)
# ---------------------------------------------------------------------------

def marker_cb(msg):
    """Log a visualization_msgs/Marker (TRIANGLE_LIST or LINE_LIST) to Rerun."""
    if not HAS_MARKER:
        return
    rr.set_time_seconds("ros_time", msg.header.stamp.to_sec())
    if msg.type == Marker.TRIANGLE_LIST and len(msg.points) >= 3:
        pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        rr.log("slam/mesh", rr.Points3D(pts, radii=0.02,
                                         colors=[[180, 180, 255]] * len(pts)))
    elif msg.type == Marker.LINE_LIST and len(msg.points) >= 2:
        pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        strips = [pts[i:i+2] for i in range(0, len(pts) - 1, 2)]
        rr.log("slam/mesh", rr.LineStrips3D(strips, colors=[[200, 200, 255]]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rospy.init_node("kimera_rerun_bridge", anonymous=True)

    rr.init("kimera_vio")
    # rerun 0.21: serve_web hosts both HTTP viewer and WebSocket data channel
    rr.serve_web(open_browser=False, web_port=9090, ws_port=9877)
    rospy.loginfo("Rerun web viewer ready at http://localhost:9090/?url=ws://localhost:9877")

    # Static world axes
    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    # Kimera-VIO-ROS topics
    rospy.Subscriber("/kimera_vio_ros/odometry", Odometry, odom_cb, queue_size=10)
    rospy.Subscriber("/kimera_vio_ros/imu_odometry", Odometry, imu_odom_cb, queue_size=10)
    rospy.Subscriber("/kimera_vio_ros/optimized_odometry", Odometry, optimized_odom_cb, queue_size=10)
    rospy.Subscriber("/kimera_vio_ros/path", Path, path_cb, queue_size=5)
    rospy.Subscriber("/kimera_vio_ros/optimized_path", Path, optimized_path_cb, queue_size=5)
    rospy.Subscriber("/kimera_vio_ros/frontend/feature_tracks", Image, feature_tracks_cb, queue_size=5)

    if HAS_MARKER:
        rospy.Subscriber("/kimera_vio_ros/mesh", Marker, marker_cb, queue_size=2)

    # Fallback: plain /odometry and /path (standalone Kimera-VIO with minimal ROS bridge)
    rospy.Subscriber("/odometry", Odometry, odom_cb, queue_size=10)
    rospy.Subscriber("/path", Path, path_cb, queue_size=5)

    rospy.loginfo(
        "Subscribed to Kimera-VIO-ROS topics:\n"
        "  /kimera_vio_ros/odometry\n"
        "  /kimera_vio_ros/imu_odometry\n"
        "  /kimera_vio_ros/optimized_odometry\n"
        "  /kimera_vio_ros/path\n"
        "  /kimera_vio_ros/optimized_path\n"
        "  /kimera_vio_ros/frontend/feature_tracks\n"
        "  /kimera_vio_ros/mesh  (Marker, best-effort)\n"
        "Fallback: /odometry, /path"
    )
    rospy.spin()


if __name__ == "__main__":
    main()
