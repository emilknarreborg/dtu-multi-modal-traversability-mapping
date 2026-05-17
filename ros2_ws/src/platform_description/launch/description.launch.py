"""
Publish robot_description and start robot_state_publisher for the Warthog.

Usage (standalone):
  ros2 launch platform_description description.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory("platform_description"), "urdf", "warthog.urdf"
    )
    with open(urdf_path) as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
    ])
