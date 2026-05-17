"""
Full pipeline: live semseg inference + semantic costmap from a raw bag.

The bag must contain /image_raw, /camera_info, /livox/lidar, /livox/imu.
The semantic_segmentation_node runs inference on-the-fly and the
semantic_cost_node consumes the output.

Usage:
  ros2 launch bringup_pkg semantic_full_pipeline.launch.py \\
      bag_path:=/root/data/bags/dtu_lidar_image_20260416_093308_10hz
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    qos_overrides_path = os.path.join(
        get_package_share_directory("bringup_pkg"), "config", "bag_qos_overrides.yaml"
    )

    urdf_path = os.path.join(
        get_package_share_directory("platform_description"), "urdf", "warthog.urdf"
    )
    with open(urdf_path) as f:
        robot_description = f.read()

    bag_path            = LaunchConfiguration("bag_path")
    playback_rate       = LaunchConfiguration("playback_rate")
    cost_mode           = LaunchConfiguration("cost_mode")
    uncertainty_penalty = LaunchConfiguration("uncertainty_penalty")
    lidar_frame         = LaunchConfiguration("lidar_frame")
    camera_frame        = LaunchConfiguration("camera_frame")

    launch_dir = PathJoinSubstitution([FindPackageShare("bringup_pkg"), "launch"])

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value="/root/data/bags/dtu_lidar_image_20260416_093308_10hz",
        ),
        DeclareLaunchArgument("playback_rate",       default_value="0.1"),
        DeclareLaunchArgument("cost_mode",           default_value="penalized_expected_cost"),
        DeclareLaunchArgument("uncertainty_penalty", default_value="0.25"),
        DeclareLaunchArgument("lidar_frame",         default_value="livox_frame"),
        DeclareLaunchArgument("camera_frame",        default_value="default_cam"),

        Node(
            package="perception_pkg",
            executable="semantic_segmentation_node",
            name="semantic_segmentation_node",
            output="screen",
            parameters=[{
                "publish_probabilities": True,
                "publish_segmentation":  False,
                "publish_overlay":       False,
            }],
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([launch_dir, "semantic_cost_node.launch.py"])
            ),
            launch_arguments={
                "publish_cost_image":  "true",
                "cost_mode":           cost_mode,
                "uncertainty_penalty": uncertainty_penalty,
                "lidar_frame":         lidar_frame,
                "camera_frame":        camera_frame,
                "costmap_frame_id":    "base_link",
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([launch_dir, "semantic_cost_rviz.launch.py"])
            ),
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),

        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play",
                bag_path,
                "-r", playback_rate,
                "--qos-profile-overrides-path", qos_overrides_path,
            ],
            output="screen",
        ),
    ])
