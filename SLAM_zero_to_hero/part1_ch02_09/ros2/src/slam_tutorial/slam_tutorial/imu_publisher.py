#!/usr/bin/env python3
"""
ROS2 IMU Publisher (sensor_data QoS).

Publishes synthetic IMU data on /imu/data at 100 Hz using the
sensor_data QoS preset (best-effort, depth=5). Used by:
  - qos_subscriber.py to demonstrate matching QoS
  - sync_subscriber.py for ApproximateTimeSynchronizer with /robot/pose
"""

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class IMUPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')

        self.publisher = self.create_publisher(
            Imu, '/imu/data', qos_profile_sensor_data
        )
        self.timer = self.create_timer(0.01, self.publish)  # 100 Hz
        self.t = 0.0

        self.get_logger().info(
            'IMU publisher started (100 Hz, sensor_data QoS)'
        )

    def publish(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.1 + random.gauss(0.0, 0.005)

        msg.linear_acceleration.x = (
            -math.cos(self.t * 0.1) * 0.05 + random.gauss(0.0, 0.02)
        )
        msg.linear_acceleration.y = (
            -math.sin(self.t * 0.1) * 0.05 + random.gauss(0.0, 0.02)
        )
        msg.linear_acceleration.z = 9.81 + random.gauss(0.0, 0.02)

        msg.orientation.w = 1.0

        self.publisher.publish(msg)
        self.t += 0.01


def main(args=None):
    rclpy.init(args=args)
    node = IMUPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
