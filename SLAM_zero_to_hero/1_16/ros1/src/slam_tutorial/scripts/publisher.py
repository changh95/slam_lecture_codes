#!/usr/bin/env python3
"""
ROS1 Publisher Example

Publishes robot pose messages at 10 Hz.
Demonstrates basic ROS1 Python publisher pattern.
"""

import rospy
from geometry_msgs.msg import PoseStamped
import math


def main():
    # Initialize node
    rospy.init_node('pose_publisher', anonymous=True)

    # Create publisher
    pub = rospy.Publisher('/robot/pose', PoseStamped, queue_size=10)

    # Set rate (10 Hz)
    rate = rospy.Rate(10)

    rospy.loginfo("Pose publisher started")

    # Simulation time
    t = 0.0

    while not rospy.is_shutdown():
        # Create message
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        # Simulate circular motion
        msg.pose.position.x = math.cos(t * 0.1) * 5.0
        msg.pose.position.y = math.sin(t * 0.1) * 5.0
        msg.pose.position.z = 0.0

        # Simple quaternion (yaw only)
        yaw = t * 0.1
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)

        # Publish
        pub.publish(msg)

        if int(t) % 10 == 0:
            rospy.loginfo(f"Published pose: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")

        t += 0.1
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
