from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bag_path = LaunchConfiguration("bag_path")
    playback_rate = LaunchConfiguration("playback_rate")

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value="/home/emilknarreborg/bachelor_thesis/data/bags/segmented_test_20260326_125630",
        ),
        DeclareLaunchArgument("playback_rate", default_value="10.0"),
        ExecuteProcess(
            cmd=[
                "ros2",
                "bag",
                "play",
                bag_path,
                "-r",
                playback_rate,
            ],
            output="screen",
        ),
    ])
