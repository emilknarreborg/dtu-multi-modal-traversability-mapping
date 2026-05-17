import os
from datetime import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_output = f'/root/data/bags/dtu_lidar_image_{timestamp}'

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('livox_ros_driver2'),
                'launch_ROS2',
                'msg_MID360_launch.py',
            )
        )
    )

    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{
            "video_device": "/dev/video0",
            "image_width": 1280,
            "image_height": 720,
            "pixel_format": "mjpeg2rgb",
            "frame_rate": 30,
            "camera_info_url": "file:///root/ros2_ws/src/bringup_pkg/config/camera_info.yaml"
        }]
    )

    rosbag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', bag_output,
            '/image_raw',
            '/camera_info',
            '/livox/lidar',
            '/livox/imu'
        ],
        output='screen'
    )

    return LaunchDescription([
        lidar_launch,
        camera_node,
        rosbag_record,
    ])
