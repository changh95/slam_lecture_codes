#!/usr/bin/env python3
"""
ROS2 Subscriber Example

Subscribes to robot pose messages and tracks distance traveled.
Demonstrates basic ROS2 Python subscriber pattern.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math


class PoseSubscriber(Node):
    def __init__(self):
        super().__init__('pose_subscriber')

        # Create subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            '/robot/pose',
            self.callback,
            10
        )

        # Track distance
        self.last_x = None
        self.last_y = None
        self.total_distance = 0.0
        self.msg_count = 0

        self.get_logger().info('Pose subscriber started')

    def callback(self, msg):
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
        self.msg_count += 1

        # Log periodically
        if self.msg_count % 50 == 0:
            self.get_logger().info(
                f'Received pose: ({x:.2f}, {y:.2f}), '
                f'total distance: {self.total_distance:.2f}m'
            )


def main(args=None):
    rclpy.init(args=args)

    subscriber = PoseSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
