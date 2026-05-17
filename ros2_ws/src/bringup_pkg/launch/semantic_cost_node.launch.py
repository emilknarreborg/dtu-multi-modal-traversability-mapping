from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    publish_cost_image      = LaunchConfiguration("publish_cost_image")
    publish_class_overlay   = LaunchConfiguration("publish_class_overlay")
    publish_lidar_overlay   = LaunchConfiguration("publish_lidar_overlay")
    lidar_overlay_dot_radius = LaunchConfiguration("lidar_overlay_dot_radius")
    point_footprint_cells   = LaunchConfiguration("point_footprint_cells")
    cost_mode               = LaunchConfiguration("cost_mode")
    uncertainty_penalty     = LaunchConfiguration("uncertainty_penalty")
    lidar_frame             = LaunchConfiguration("lidar_frame")
    camera_frame            = LaunchConfiguration("camera_frame")
    costmap_frame_id        = LaunchConfiguration("costmap_frame_id")
    costmap_resolution      = LaunchConfiguration("costmap_resolution")
    costmap_width_m         = LaunchConfiguration("costmap_width_m")
    costmap_height_m        = LaunchConfiguration("costmap_height_m")
    costmap_origin_x        = LaunchConfiguration("costmap_origin_x")
    costmap_origin_y        = LaunchConfiguration("costmap_origin_y")
    costmap_origin_z        = LaunchConfiguration("costmap_origin_z")
    min_point_z             = LaunchConfiguration("min_point_z")
    max_point_z             = LaunchConfiguration("max_point_z")
    cam_topic_prefix        = LaunchConfiguration("cam_topic_prefix")
    sensor_config           = LaunchConfiguration("sensor_config")
    lidar_input_topic       = LaunchConfiguration("lidar_input_topic")
    temporal_smooth         = LaunchConfiguration("temporal_smooth")
    temporal_alpha          = LaunchConfiguration("temporal_alpha")

    return LaunchDescription([
        DeclareLaunchArgument("publish_cost_image",       default_value="false"),
        DeclareLaunchArgument("publish_class_overlay",    default_value="false"),
        DeclareLaunchArgument("publish_lidar_overlay",    default_value="false"),
        DeclareLaunchArgument("lidar_overlay_dot_radius", default_value="3"),
        DeclareLaunchArgument("point_footprint_cells",    default_value="1",
                              description="Radius in cells to dilate each lidar point in the semantic costmap"),
        DeclareLaunchArgument("cost_mode",                default_value="penalized_expected_cost"),
        DeclareLaunchArgument("uncertainty_penalty",   default_value="0.25"),
        DeclareLaunchArgument("lidar_frame",           default_value="livox_frame"),
        DeclareLaunchArgument("camera_frame",          default_value="default_cam"),
        DeclareLaunchArgument("costmap_frame_id",      default_value="base_link"),
        DeclareLaunchArgument("costmap_resolution",    default_value="0.2"),
        DeclareLaunchArgument("costmap_width_m",       default_value="40.0"),
        DeclareLaunchArgument("costmap_height_m",      default_value="20.0"),
        DeclareLaunchArgument("costmap_origin_x",      default_value="-20.0"),
        DeclareLaunchArgument("costmap_origin_y",      default_value="-10.0"),
        DeclareLaunchArgument("costmap_origin_z",      default_value="-0.3"),
        DeclareLaunchArgument("min_point_z",           default_value="-2.0"),
        DeclareLaunchArgument("max_point_z",           default_value="2.0"),
        DeclareLaunchArgument(
            "cam_topic_prefix", default_value="",
            description="Prefix prepended to /segmentation/probabilities, "
                        "/segmentation/probability_classes, and /camera_info. "
                        "Set to e.g. /pylon_camera_node for RELLIS bags."),
        DeclareLaunchArgument(
            "sensor_config", default_value="livox_mid360",
            description="Sensor profile — controls whether the livox_frame→camera static TF is published. "
                        "Set to ouster_os1 to skip it (Ouster TF comes from the URDF/bag)."),
        DeclareLaunchArgument(
            "lidar_input_topic", default_value="/livox/lidar",
            description="LiDAR topic to subscribe to for point projection into the semantic image "
                        "(e.g. /os1_cloud_node/points for Ouster/RELLIS)."),
        DeclareLaunchArgument("temporal_smooth", default_value="false",
                              description="Enable EMA temporal smoothing on the semantic costmap"),
        DeclareLaunchArgument("temporal_alpha",  default_value="0.4",
                              description="EMA weight for the incoming frame (0–1); lower = smoother but laggier"),

        # Livox-only: publish the calibrated livox_frame → default_cam static TF.
        # For Ouster/RELLIS the TF is provided by the URDF or recorded in the bag,
        # so publishing it here would clobber the correct extrinsic.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="lidar_camera_tf",
            output="screen",
            arguments=[
                "0.04", "0", "-0.22",
                "1.361357", "0", "1.500983",
                "livox_frame", "default_cam",
            ],
            condition=IfCondition(PythonExpression(["'", sensor_config, "' == 'livox_mid360'"])),
        ),

        Node(
            package="traversability_pkg",
            executable="semantic_cost_node",
            name="semantic_cost_node",
            output="screen",
            parameters=[{
                "publish_cost_image":       publish_cost_image,
                "publish_class_overlay":    publish_class_overlay,
                "publish_lidar_overlay":    publish_lidar_overlay,
                "lidar_overlay_dot_radius": lidar_overlay_dot_radius,
                "point_footprint_cells":    point_footprint_cells,
                "cost_mode":               cost_mode,
                "uncertainty_penalty":   uncertainty_penalty,
                "lidar_frame":           lidar_frame,
                "camera_frame":          camera_frame,
                "lidar_input_topic":     lidar_input_topic,
                "costmap_frame_id":      costmap_frame_id,
                "costmap_resolution":    costmap_resolution,
                "costmap_width_m":       costmap_width_m,
                "costmap_height_m":      costmap_height_m,
                "costmap_origin_x":      costmap_origin_x,
                "costmap_origin_y":      costmap_origin_y,
                "costmap_origin_z":      costmap_origin_z,
                "min_point_z":           min_point_z,
                "max_point_z":           max_point_z,
                "temporal_smooth":       temporal_smooth,
                "temporal_alpha":        temporal_alpha,
                "semantic_probabilities_topic":
                    PythonExpression(["'", cam_topic_prefix,
                                      "' + '/segmentation/probabilities'"]),
                "semantic_probability_classes_topic":
                    PythonExpression(["'", cam_topic_prefix,
                                      "' + '/segmentation/probability_classes'"]),
                "camera_intrinsics_topic":
                    PythonExpression(["'", cam_topic_prefix, "' + '/camera_info'"]),
            }],
        ),
    ])
