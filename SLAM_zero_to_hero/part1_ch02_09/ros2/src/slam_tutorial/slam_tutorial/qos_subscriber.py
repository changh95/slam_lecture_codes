#!/usr/bin/env python3
"""
ROS2 QoS Subscriber Example.

Subscribes to /imu/data with qos_profile_sensor_data (best-effort,
depth=5), matching the publisher's QoS. If you change the QoS argument
to a plain integer like `10`, you will receive zero messages because
the publisher uses BEST_EFFORT and the default subscriber profile
expects RELIABLE — one of the most common ROS2 footguns.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class QoSSensorSubscriber(Node):
    def __init__(self):
        super().__init__('qos_sensor_subscriber')

        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.callback,
            qos_profile_sensor_data,
        )

        self.count = 0
        self.get_logger().info(
            'IMU subscriber started (sensor_data QoS, matches publisher)'
        )

    def callback(self, msg):
        self.count += 1
        if self.count % 100 == 0:
            self.get_logger().info(
                f'Received {self.count} IMU messages, '
                f'latest az={msg.linear_acceleration.z:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = QoSSensorSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
