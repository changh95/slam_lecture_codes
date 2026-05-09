#!/usr/bin/env python3
"""
ROS2 TF Listener Example.

Looks up map -> base_link once per second and prints the result.
This is the standard way SLAM front-ends ask "where am I right now?"
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException


class MapBaseLinkListener(Node):
    def __init__(self):
        super().__init__('map_base_listener')

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.timer = self.create_timer(1.0, self.lookup)

        self.get_logger().info('Looking up map -> base_link every 1.0 s')

    def lookup(self):
        try:
            tf = self.buffer.lookup_transform(
                'map',           # target
                'base_link',     # source
                Time()           # latest available
            )
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        self.get_logger().info(
            f'base_link in map frame: ({x:.2f}, {y:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MapBaseLinkListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
