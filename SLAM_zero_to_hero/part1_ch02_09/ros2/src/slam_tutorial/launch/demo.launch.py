"""
Demo launch file for slam_tutorial.

Brings up the full Pub/Sub + TF + URDF + QoS + sync demo:

  Pose pub/sub:
    - publisher.py            (PoseStamped on /robot/pose)
    - subscriber.py           (consumes /robot/pose)

  TF:
    - tf_broadcaster.py       (odom -> base_link, follows circular motion)
    - tf_listener.py          (looks up map -> base_link once per second)
    - static_transform_publisher (map -> odom, identity)
    - robot_state_publisher   (URDF: base_link -> camera/lidar/imu)

  IMU + QoS + message_filters:
    - imu_publisher.py        (Imu on /imu/data, sensor_data QoS, 100 Hz)
    - qos_subscriber.py       (best-effort QoS subscriber, matches publisher)
    - sync_subscriber.py      (pairs PoseStamped + Imu by timestamp)

Optional:
    - rviz2 (pass `rviz:=true` to launch with the bundled config)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('slam_tutorial')
    urdf_path = os.path.join(pkg_share, 'urdf', 'slam_robot.urdf')
    rviz_config = os.path.join(pkg_share, 'rviz', 'slam_demo.rviz')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    use_rviz = LaunchConfiguration('rviz')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Whether to launch RViz2 with the bundled config'
    )

    return LaunchDescription([
        rviz_arg,

        # Pose pub/sub demo
        Node(
            package='slam_tutorial', executable='publisher.py',
            name='pose_publisher', output='screen',
        ),
        Node(
            package='slam_tutorial', executable='subscriber.py',
            name='pose_subscriber', output='screen',
        ),

        # TF demo: odom -> base_link follows the same motion as publisher.py
        Node(
            package='slam_tutorial', executable='tf_broadcaster.py',
            name='odom_tf_broadcaster', output='screen',
        ),
        Node(
            package='slam_tutorial', executable='tf_listener.py',
            name='map_base_listener', output='screen',
        ),

        # Static map -> odom (identity)
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='map_to_odom',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'map',
                '--child-frame-id', 'odom',
            ],
        ),

        # robot_state_publisher: base_link -> camera/lidar/imu from URDF
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),

        # IMU + QoS demo
        Node(
            package='slam_tutorial', executable='imu_publisher.py',
            name='imu_publisher', output='screen',
        ),
        Node(
            package='slam_tutorial', executable='qos_subscriber.py',
            name='qos_sensor_subscriber', output='screen',
        ),

        # message_filters sync demo
        Node(
            package='slam_tutorial', executable='sync_subscriber.py',
            name='pose_imu_sync', output='screen',
        ),

        # Optional RViz
        Node(
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', rviz_config], output='screen',
            condition=IfCondition(use_rviz),
        ),
    ])
