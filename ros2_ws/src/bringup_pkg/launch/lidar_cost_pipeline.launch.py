"""
Bag playback + lidar_cost_node only → /lidar_costmap

Usage:
  # Rellis / Ouster OS1-64 (default):
  ros2 launch bringup_pkg lidar_cost_pipeline.launch.py \
      bag_path:=/home/emilknarreborg/bachelor_thesis/data/bags/segmented_test_20260326_125630

  # Livox Mid-360 live or bag:
  ros2 launch bringup_pkg lidar_cost_pipeline.launch.py \
      sensor_config:=livox_mid360 play_bag:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = get_package_share_directory("bringup_pkg")
    qos_overrides_path = os.path.join(bringup_share, "config", "bag_qos_overrides.yaml")
    rviz_config = os.path.join(bringup_share, "rviz", "bachelor_thesis_rviz.rviz")

    urdf_path = os.path.join(
        get_package_share_directory("platform_description"), "urdf", "warthog.urdf"
    )
    with open(urdf_path) as f:
        robot_description = f.read()

    bag_path      = LaunchConfiguration("bag_path")
    playback_rate = LaunchConfiguration("playback_rate")
    play_bag      = LaunchConfiguration("play_bag")
    use_rviz      = LaunchConfiguration("use_rviz")
    sensor_config = LaunchConfiguration("sensor_config")
    fill_nn = LaunchConfiguration("fill_nn")
    neighbor_radius = LaunchConfiguration("neighbor_radius")
    slope_weight = LaunchConfiguration("slope_weight")
    roughness_weight = LaunchConfiguration("roughness_weight")

    sensor_params = PathJoinSubstitution([
        FindPackageShare("bringup_pkg"), "config", "sensors",
        [sensor_config, ".yaml"],
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value="/root/data/bags/dtu_lidar_image_20260415_120314",
        ),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("play_bag", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "sensor_config",
            default_value="livox_mid360",
            description="Sensor profile to load: ouster_os1 | livox_mid360",
        ),
        DeclareLaunchArgument(
            "fill_nn",
            default_value="false",
            description="Override fill_nn from sensor config (true|false)",
        ),
        DeclareLaunchArgument(
            "neighbor_radius",
            default_value="0.6",
            description="Radius for neighbor search in cost calculation",
        ),
        DeclareLaunchArgument(
            "slope_weight",
            default_value="0.5",
            description="Weight for slope in cost calculation",
        ),
        DeclareLaunchArgument(
            "roughness_weight",
            default_value="0.5",
            description="Weight for roughness in cost calculation",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),

        Node(
            package="traversability_pkg",
            executable="lidar_cost_node",
            name="lidar_cost_node",
            output="screen",
            parameters=[sensor_params, {"fill_nn": fill_nn, "neighbor_radius": neighbor_radius, "slope_weight": slope_weight, "roughness_weight": roughness_weight}],
        ),

        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            name="imu_filter",
            parameters=[{
                "use_mag": False,
                "world_frame": "enu",
                "publish_tf": False,
            }],
            remappings=[("imu/data_raw", "/livox/imu"),
                        ("imu/data",     "/imu/data")],
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="livox_tf",
            arguments=[
                "--x", "0", "--y", "0", "--z", "1",
                "--roll", "3.14159", "--pitch", "0", "--yaw", "0",
                "--frame-id", "base_link",
                "--child-frame-id", "livox_frame",
            ],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
            condition=IfCondition(use_rviz),
        ),

        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play",
                bag_path,
                "-r", playback_rate,
                "--read-ahead-queue-size", "1000",
                "--qos-profile-overrides-path", qos_overrides_path,
            ],
            output="screen",
            condition=IfCondition(play_bag),
        ),
    ])
