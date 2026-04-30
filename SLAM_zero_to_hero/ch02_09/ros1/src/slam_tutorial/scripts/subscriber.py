#!/usr/bin/env python3
"""
ROS1 Subscriber Example

Subscribes to robot pose messages.
Demonstrates basic ROS1 Python subscriber pattern.
"""

import rospy
from geometry_msgs.msg import PoseStamped
import math


class PoseSubscriber:
    def __init__(self):
        # Initialize node
        rospy.init_node('pose_subscriber', anonymous=True)

        # Create subscriber
        self.sub = rospy.Subscriber('/robot/pose', PoseStamped, self.callback)

        # Store last position for distance calculation
        self.last_x = None
        self.last_y = None
        self.total_distance = 0.0

        rospy.loginfo("Pose subscriber started")

    def callback(self, msg):
        """Callback function for pose messages."""
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Calculate distance traveled
        if self.last_x is not None:
            dx = x - self.last_x
            dy = y - self.last_y
            distance = math.sqrt(dx * dx + dy * dy)
            self.total_distance += distance

        self.last_x = x
        self.last_y = y

        # Log periodically
        if int(rospy.Time.now().to_sec() * 10) % 50 == 0:
            rospy.loginfo(
                f"Received pose: ({x:.2f}, {y:.2f}), "
                f"total distance: {self.total_distance:.2f}m"
            )

    def run(self):
        """Keep node running."""
        rospy.spin()


def main():
    subscriber = PoseSubscriber()
    subscriber.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
