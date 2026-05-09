#!/usr/bin/env python3
"""
ROS2 message_filters Synchronizer Example.

Demonstrates ApproximateTimeSynchronizer: receives /robot/pose and
/imu/data together when their timestamps fall within `slop` seconds
of each other. This is exactly how stereo VO, RGB-D SLAM, and
Visual-Inertial systems pair sensor streams.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu


class PoseImuSync(Node):
    def __init__(self):
        super().__init__('pose_imu_sync')

        self.pose_sub = Subscriber(self, PoseStamped, '/robot/pose')
        self.imu_sub = Subscriber(
            self, Imu, '/imu/data', qos_profile=qos_profile_sensor_data
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.pose_sub, self.imu_sub],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self.callback)

        self.matched = 0
        self.get_logger().info(
            'Sync subscriber started (PoseStamped + Imu, slop=50 ms)'
        )

    def callback(self, pose_msg, imu_msg):
        self.matched += 1
        if self.matched % 10 == 0:
            pose_ns = (
                pose_msg.header.stamp.sec * 1_000_000_000
                + pose_msg.header.stamp.nanosec
            )
            imu_ns = (
                imu_msg.header.stamp.sec * 1_000_000_000
                + imu_msg.header.stamp.nanosec
            )
            dt_ms = (pose_ns - imu_ns) / 1e6
            self.get_logger().info(
                f'Synced pair #{self.matched}: '
                f'pose=({pose_msg.pose.position.x:.2f}, '
                f'{pose_msg.pose.position.y:.2f}) '
                f'imu_az={imu_msg.linear_acceleration.z:.2f} '
                f'dt={dt_ms:.1f} ms'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PoseImuSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
