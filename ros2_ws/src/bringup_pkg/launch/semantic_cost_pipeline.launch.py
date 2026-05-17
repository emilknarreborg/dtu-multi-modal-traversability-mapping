"""
Semantic costmap pipeline from a pre-segmented rosbag.

The bag must already contain /segmentation/probabilities and
/segmentation/probability_classes (produced by semseg_record_bag.launch.py
or offline_semseg.py).

Usage:
  ros2 launch bringup_pkg semantic_cost_pipeline.launch.py \\
      bag_path:=/root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz
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
    publish_cost_image  = LaunchConfiguration("publish_cost_image")
    cost_mode           = LaunchConfiguration("cost_mode")
    uncertainty_penalty = LaunchConfiguration("uncertainty_penalty")
    lidar_frame         = LaunchConfiguration("lidar_frame")
    camera_frame        = LaunchConfiguration("camera_frame")

    launch_dir = PathJoinSubstitution([FindPackageShare("bringup_pkg"), "launch"])

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value="/root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz",
        ),
        DeclareLaunchArgument("playback_rate",       default_value="1.0"),
        DeclareLaunchArgument("publish_cost_image",  default_value="true"),
        DeclareLaunchArgument("cost_mode",           default_value="penalized_expected_cost"),
        DeclareLaunchArgument("uncertainty_penalty", default_value="0.25"),
        DeclareLaunchArgument("lidar_frame",         default_value="livox_frame"),
        DeclareLaunchArgument("camera_frame",        default_value="default_cam"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([launch_dir, "semantic_cost_node.launch.py"])
            ),
            launch_arguments={
                "publish_cost_image":  publish_cost_image,
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
