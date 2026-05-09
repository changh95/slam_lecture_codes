#!/usr/bin/env python3
"""
ROS2 TF Broadcaster Example.

Broadcasts the odom -> base_link transform that follows the same
circular motion as publisher.py. Pairs with tf_listener.py and is
launched together with a static map -> odom transform plus
robot_state_publisher in launch/demo.launch.py.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast)
        self.t = 0.0

        self.get_logger().info(
            'odom -> base_link broadcaster started (10 Hz, circular motion)'
        )

    def broadcast(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = math.cos(self.t * 0.1) * 5.0
        t.transform.translation.y = math.sin(self.t * 0.1) * 5.0
        t.transform.translation.z = 0.0

        yaw = self.t * 0.1
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2)
        t.transform.rotation.w = math.cos(yaw / 2)

        self.br.sendTransform(t)
        self.t += 0.1


def main(args=None):
    rclpy.init(args=args)
    node = OdomTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
