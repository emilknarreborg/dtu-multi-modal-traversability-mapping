"""
Run semantic_segmentation_node on a raw bag and record a new bag that
includes the original topics plus /segmentation/probabilities and
/segmentation/probability_classes.

Usage:
  ros2 launch bringup_pkg semseg_record_bag.launch.py \\
      bag_path:=/root/data/bags/dtu_lidar_image_20260416_093308_10hz \\
      bag_output:=/root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory("bringup_pkg")
    qos_overrides_path = os.path.join(bringup_share, "config", "bag_qos_overrides.yaml")

    bag_path      = LaunchConfiguration("bag_path")
    playback_rate = LaunchConfiguration("playback_rate")
    bag_output    = LaunchConfiguration("bag_output")

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value="/root/data/bags/dtu_lidar_image_20260416_093308_10hz",
        ),
        DeclareLaunchArgument("playback_rate", default_value="0.1"),
        DeclareLaunchArgument(
            "bag_output",
            default_value="/root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz",
        ),

        Node(
            package="perception_pkg",
            executable="semantic_segmentation_node",
            name="semantic_segmentation_node",
            output="screen",
            parameters=[{
                "checkpoint_path":         "/root/finetuned_models/gscnn_finetuned_mixed_best_final.pth",
                "resize_width":            640,   # model trained on 640×400 (landscape)
                "resize_height":           400,
                "max_inference_hz":        0.0,
                "use_fp16":                False,
                "load_model_fp16_on_cuda": False,
            }],
        ),

        ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "-o", bag_output,
                "/image_raw",
                "/camera_info",
                "/livox/lidar",
                "/livox/imu",
                "/segmentation/probabilities",
                "/segmentation/probability_classes",
            ],
            output="screen",
        ),

        # Delay playback so the segmentation node and recorder are ready first.
        TimerAction(
            period=10.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2", "bag", "play",
                        bag_path,
                        "-r", playback_rate,
                        "--qos-profile-overrides-path", qos_overrides_path,
                    ],
                    output="screen",
                ),
            ],
        ),
    ])
