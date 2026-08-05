#!/usr/bin/env python3
"""Turn Cerberus 2.0's pose streams into nav_msgs/Path for rviz, and the mocap too.

src/utils/visualization.cpp in Cerberus 2.0 has every publisher except
pubTrackImage commented out, so a stock run emits no nav_msgs/Path and no
nav_msgs/Odometry -- the only live pose outputs are two
geometry_msgs/PoseWithCovarianceStamped topics. rviz cannot draw a trajectory
from those, so this node accumulates them, and does the same for the mocap topic
so ground truth is on the same screen:

    /vilo/estimate_pose      ->  /vilo/path_viz    fused estimate: stereo + trunk IMU + legs
    /mipo/estimate_pose      ->  /mipo/path_viz    proprioception only: 5 IMUs + joints, no camera
    <gt_topic>               ->  /gt/path_viz      Optitrack, indoor sequences only

It is a viewer aid and nothing more: no filtering, no transform, one Path point
per received pose in the publisher's own frame.

The mocap frame is NOT the estimator's frame -- Optitrack reports poses in the
room frame while the estimator starts at the robot's initial pose -- so
/gt/path_viz is republished twice: once raw, and once rigidly re-anchored so the
first mocap pose sits at the first VILO pose. Only the re-anchored one is
comparable by eye in rviz; the real number comes from plot_trajectory.py, which
solves the full rigid alignment over the whole overlap.
"""
import math

import rospy
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path


class PathBuilder:
    """Accumulate a pose stream into a Path, decimated by distance."""

    def __init__(self, dst_topic, frame_id, min_step):
        # The estimator publishes at its own loop rate (~400 Hz for MIPO), which
        # would grow a Path of hundreds of thousands of poses over a 10 min bag
        # and stall rviz. Decimate by distance instead of by count so the drawn
        # line keeps its shape whether the robot is trotting or standing still.
        self.min_step = min_step
        self.path = Path()
        self.path.header.frame_id = frame_id
        self.last = None
        self.pub = rospy.Publisher(dst_topic, Path, queue_size=1, latch=True)

    def add(self, header, pose):
        p = pose.position
        if self.last is not None:
            d2 = (p.x - self.last[0]) ** 2 + (p.y - self.last[1]) ** 2 + (p.z - self.last[2]) ** 2
            if d2 < self.min_step * self.min_step:
                return False
        self.last = (p.x, p.y, p.z)
        ps = PoseStamped()
        ps.header = header
        ps.header.frame_id = self.path.header.frame_id
        ps.pose = pose
        self.path.poses.append(ps)
        self.path.header.stamp = header.stamp
        self.pub.publish(self.path)
        return True


class Node:
    def __init__(self):
        frame_id = rospy.get_param("~frame_id", "world")
        min_step = float(rospy.get_param("~min_step", 0.02))
        # Same default as launch/cerberus2_bag.launch: the released indoor bags
        # publish mocap here, not on parameters.cpp's /mocap_node/Go1_body/pose.
        gt_topic = rospy.get_param("GT_TOPIC", "/natnet_ros/Shuo_Go1/pose")

        self.vilo = PathBuilder("/vilo/path_viz", frame_id, min_step)
        self.mipo = PathBuilder("/mipo/path_viz", frame_id, min_step)
        self.gt_raw = PathBuilder("/gt/path_viz_raw", frame_id, min_step)
        self.gt = PathBuilder("/gt/path_viz", frame_id, min_step)

        self.first_vilo = None   # first fused pose, the anchor
        self.gt_offset = None    # (dx, dy, dz) applied to every mocap pose

        rospy.Subscriber("/vilo/estimate_pose", PoseWithCovarianceStamped, self.cb_vilo, queue_size=200)
        rospy.Subscriber("/mipo/estimate_pose", PoseWithCovarianceStamped, self.cb_mipo, queue_size=200)
        rospy.Subscriber(gt_topic, PoseStamped, self.cb_gt, queue_size=200)
        rospy.loginfo("pose_to_path: /vilo/path_viz, /mipo/path_viz, /gt/path_viz "
                      "(gt topic %s, frame %s, step %.3f m)", gt_topic, frame_id, min_step)

    def cb_vilo(self, msg):
        if self.first_vilo is None:
            p = msg.pose.pose.position
            if not all(map(math.isfinite, (p.x, p.y, p.z))):
                return
            self.first_vilo = (p.x, p.y, p.z)
        self.vilo.add(msg.header, msg.pose.pose)

    def cb_mipo(self, msg):
        self.mipo.add(msg.header, msg.pose.pose)

    def cb_gt(self, msg):
        self.gt_raw.add(msg.header, msg.pose)
        # Translation-only re-anchor. A full rigid alignment needs the whole
        # trajectory (plot_trajectory.py does that with Umeyama); the point here
        # is only to put the two tracks in the same neighbourhood on screen.
        if self.gt_offset is None:
            if self.first_vilo is None:
                return
            g = msg.pose.position
            self.gt_offset = (self.first_vilo[0] - g.x, self.first_vilo[1] - g.y, self.first_vilo[2] - g.z)
            rospy.loginfo("pose_to_path: mocap re-anchored to the first VILO pose, offset (%.2f %.2f %.2f)",
                          *self.gt_offset)
        shifted = Pose()
        shifted.orientation = msg.pose.orientation
        shifted.position.x = msg.pose.position.x + self.gt_offset[0]
        shifted.position.y = msg.pose.position.y + self.gt_offset[1]
        shifted.position.z = msg.pose.position.z + self.gt_offset[2]
        self.gt.add(msg.header, shifted)


def main():
    rospy.init_node("pose_to_path")
    Node()
    rospy.spin()


if __name__ == "__main__":
    main()
