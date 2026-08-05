#!/usr/bin/python3
# -- coding: utf-8 --**

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():

    # Find path
    config_file_dir = os.path.join(get_package_share_directory("fast_livo"), "config")
    rviz_config_file = os.path.join(get_package_share_directory("fast_livo"), "rviz_cfg", "fast_livo2.rviz")

    #Load parameters
    mid360_config_cmd = os.path.join(config_file_dir, "mid360_lvi.yaml")
    camera_config_cmd = os.path.join(config_file_dir, "camera_rational.yaml")

    # Param use_rviz
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="False",
        description="Whether to launch Rviz2",
    )

    mid360_config_arg = DeclareLaunchArgument(
        'mid360_params_file',
        default_value=mid360_config_cmd,
        description='Full path to the ROS2 parameters file to use for fast_livo2 nodes',
    )

    camera_config_arg = DeclareLaunchArgument(
        'camera_params_file',
        default_value=camera_config_cmd,
        description='Full path to the ROS2 parameters file to use for vikit_ros nodes',
    )

    use_respawn_arg = DeclareLaunchArgument(
        'use_respawn',
        default_value='True',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.')

    mid360_params_file = LaunchConfiguration('mid360_params_file')
    camera_params_file = LaunchConfiguration('camera_params_file')
    use_respawn = LaunchConfiguration('use_respawn')

    return LaunchDescription([
        use_rviz_arg,
        mid360_config_arg,
        camera_config_arg,
        use_respawn_arg,

        # use parameter_blackboard as global parameters server and load camera params
        Node(
            package='demo_nodes_cpp',
            executable='parameter_blackboard',
            name='parameter_blackboard',
            parameters=[
                camera_params_file,
            ],
            output='screen'
        ),

        # republish compressed image to raw image
        Node(
            package="image_transport",
            executable="republish",
            name="republish",
            arguments=[
                'compressed',
                'raw',
            ],
            remappings=[
                ("in",  "/left_camera/image"),
                ("out", "/left_camera/image")
            ],
            output="screen",
            respawn=use_respawn,
        ),

        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="laserMapping",
            parameters=[
                mid360_params_file,
            ],
            output="screen"
        ),

        Node(
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_file],
            output="screen"
        ),
    ])
