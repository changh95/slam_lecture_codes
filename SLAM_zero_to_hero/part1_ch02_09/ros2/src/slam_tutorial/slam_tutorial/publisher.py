#!/usr/bin/env python3
"""
ROS2 Publisher Example

Publishes robot pose messages at 10 Hz.
Demonstrates basic ROS2 Python publisher pattern.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math


class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        # Create publisher
        self.publisher = self.create_publisher(PoseStamped, '/robot/pose', 10)

        # Create timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Simulation time
        self.t = 0.0

        self.get_logger().info('Pose publisher started')

    def timer_callback(self):
        # Create message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # Simulate circular motion
        msg.pose.position.x = math.cos(self.t * 0.1) * 5.0
        msg.pose.position.y = math.sin(self.t * 0.1) * 5.0
        msg.pose.position.z = 0.0

        # Simple quaternion (yaw only)
        yaw = self.t * 0.1
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.orientation.w = math.cos(yaw / 2)

        # Publish
        self.publisher.publish(msg)

        if int(self.t) % 10 == 0:
            self.get_logger().info(
                f'Published pose: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}'
            )

        self.t += 0.1


def main(args=None):
    rclpy.init(args=args)

    publisher = PosePublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
