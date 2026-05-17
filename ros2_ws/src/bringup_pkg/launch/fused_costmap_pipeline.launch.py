"""
Fused costmap pipeline from a pre-segmented rosbag.

  bag: /segmentation/probabilities + /segmentation/probability_classes
  bag: <lidar_topic> ──► lidar_cost_node  ──► /lidar_costmap ──────────────────┐
  bag: <lidar_topic> ──► semantic_cost_node ──► /semantic_costmap ──────────────┤  (semantic_mode=lidar_projection)
  OR  /imu/data ──────► ground_projection_node ──► /semantic_costmap ───────────┤  (semantic_mode=ground_projection)
                                                                                 ▼
                                                           fused_costmap_node ──► /fused_costmap

The bag must already contain segmentation topics (produced by
semseg_record_bag.launch.py or scripts/offline_semseg.py).

Usage (DTU / Livox Mid-360 bag, default):
  ros2 launch bringup_pkg fused_costmap_pipeline.launch.py \\
      bag_path:=/home/emilknarreborg/bachelor_thesis/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz

Usage with ground-plane backprojection (IMU-based, no LiDAR density needed):
  ros2 launch bringup_pkg fused_costmap_pipeline.launch.py \\
      semantic_mode:=ground_projection \\
      camera_height_m:=1.0 \\
      bag_path:=/home/emilknarreborg/bachelor_thesis/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz

Usage (RELLIS / Ouster OS1 bag):
  ros2 launch bringup_pkg fused_costmap_pipeline.launch.py \\
      sensor_config:=ouster_os1 \\
      bag_path:=/home/emilknarreborg/bachelor_thesis/data/bags/segmented/segmented_test_20260326_125630 \\
      lidar_frame:=ouster1/os1_lidar \\
      camera_frame:=pylon_camera_node \\
      cam_topic_prefix:=/pylon_camera_node \\
      lidar_input_topic:=/os1_cloud_node/points

Live hardware (no bag):
  ros2 launch bringup_pkg fused_costmap_pipeline.launch.py play_bag:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
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

    # ── launch argument handles ───────────────────────────────────────────────
    bag_path            = LaunchConfiguration("bag_path")
    playback_rate       = LaunchConfiguration("playback_rate")
    play_bag            = LaunchConfiguration("play_bag")
    use_rviz               = LaunchConfiguration("use_rviz")
    use_gravity_correction = LaunchConfiguration("use_gravity_correction")
    sensor_config          = LaunchConfiguration("sensor_config")

    # shared costmap grid (must match across lidar + semantic nodes)
    costmap_resolution  = LaunchConfiguration("costmap_resolution")
    costmap_width_m     = LaunchConfiguration("costmap_width_m")
    costmap_height_m    = LaunchConfiguration("costmap_height_m")
    costmap_origin_x    = LaunchConfiguration("costmap_origin_x")
    costmap_origin_y    = LaunchConfiguration("costmap_origin_y")
    costmap_origin_z    = LaunchConfiguration("costmap_origin_z")
    min_point_z         = LaunchConfiguration("min_point_z")
    max_point_z         = LaunchConfiguration("max_point_z")

    # lidar_cost_node specific
    neighbor_radius     = LaunchConfiguration("neighbor_radius")
    slope_weight        = LaunchConfiguration("slope_weight")
    roughness_weight    = LaunchConfiguration("roughness_weight")
    voxel_size          = LaunchConfiguration("voxel_size")
    fill_nn             = LaunchConfiguration("fill_nn")
    forward_fov_deg     = LaunchConfiguration("forward_fov_deg")
    accumulate_frames   = LaunchConfiguration("accumulate_frames")
    r_crit              = LaunchConfiguration("r_crit")
    s_crit              = LaunchConfiguration("s_crit")

    # Sensor YAML sets lidar_input_topic and lidar_tf_frame for the chosen sensor.
    sensor_params = PathJoinSubstitution([
        FindPackageShare("bringup_pkg"), "config", "sensors",
        [sensor_config, ".yaml"],
    ])

    # semantic node selection
    semantic_mode          = LaunchConfiguration("semantic_mode")

    # semantic_cost_node specific
    cost_mode              = LaunchConfiguration("cost_mode")
    lidar_frame            = LaunchConfiguration("lidar_frame")
    camera_frame           = LaunchConfiguration("camera_frame")
    lidar_input_topic      = LaunchConfiguration("lidar_input_topic")
    publish_cost_image     = LaunchConfiguration("publish_cost_image")
    publish_class_overlay  = LaunchConfiguration("publish_class_overlay")
    publish_lidar_overlay  = LaunchConfiguration("publish_lidar_overlay")
    lidar_overlay_dot_radius = LaunchConfiguration("lidar_overlay_dot_radius")
    point_footprint_cells  = LaunchConfiguration("point_footprint_cells")
    # RELLIS bags publish camera topics under a node-namespaced prefix
    # (e.g. /pylon_camera_node/image_raw).  Set this to remap them onto the
    # canonical unprefixed names that semantic_cost_node expects.
    cam_topic_prefix       = LaunchConfiguration("cam_topic_prefix")

    # fused_costmap_node specific
    appearance_alpha    = LaunchConfiguration("appearance_alpha")
    geometric_alpha     = LaunchConfiguration("geometric_alpha")
    max_stamp_diff_sec  = LaunchConfiguration("max_stamp_diff_sec")
    lidar_buffer_size   = LaunchConfiguration("lidar_buffer_size")
    accumulate_scans    = LaunchConfiguration("accumulate_scans")
    fused_fill_nn       = LaunchConfiguration("fused_fill_nn")
    publish_costmap     = LaunchConfiguration("publish_costmap")
    publish_voxels      = LaunchConfiguration("publish_voxels")
    lidar_temporal_smooth    = LaunchConfiguration("lidar_temporal_smooth")
    lidar_temporal_alpha     = LaunchConfiguration("lidar_temporal_alpha")
    semantic_temporal_smooth = LaunchConfiguration("semantic_temporal_smooth")
    semantic_temporal_alpha  = LaunchConfiguration("semantic_temporal_alpha")
    fused_temporal_smooth    = LaunchConfiguration("fused_temporal_smooth")
    fused_temporal_alpha     = LaunchConfiguration("fused_temporal_alpha")

    launch_dir = PathJoinSubstitution([FindPackageShare("bringup_pkg"), "launch"])

    return LaunchDescription([
        # ── bag / runtime ─────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "bag_path",
            default_value="/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz",
            description="Path to pre-segmented rosbag (only used when play_bag:=true)",
        ),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("play_bag",      default_value="true",
                              description="Set false when running on live hardware"),
        DeclareLaunchArgument("use_rviz",      default_value="true"),

        # ── shared costmap grid ───────────────────────────────────────────────
        DeclareLaunchArgument("costmap_resolution", default_value="0.2",
                              description="[m/cell] grid resolution for both costmaps"),
        DeclareLaunchArgument("costmap_width_m",    default_value="15.0",
                              description="[m] costmap width (X axis)"),
        DeclareLaunchArgument("costmap_height_m",   default_value="10.0",
                              description="[m] costmap height (Y axis)"),
        DeclareLaunchArgument("costmap_origin_x",   default_value="0.0",
                              description="[m] costmap origin X (lower-left corner)"),
        DeclareLaunchArgument("costmap_origin_y",   default_value="-5.0",
                              description="[m] costmap origin Y (lower-left corner)"),
        DeclareLaunchArgument("costmap_origin_z",   default_value="-0.3",
                              description="[m] costmap Z offset from base_link to place map at ground level"),
        DeclareLaunchArgument("min_point_z",        default_value="-2.0",
                              description="[m] minimum point Z to include"),
        DeclareLaunchArgument("max_point_z",        default_value="4.0",
                              description="[m] maximum point Z to include"),

        # ── lidar_cost_node ───────────────────────────────────────────────────
        DeclareLaunchArgument(
            "sensor_config", default_value="livox_mid360",
            description="Sensor profile: ouster_os1 | livox_mid360 "
                        "(sets lidar_input_topic and lidar_tf_frame via config/sensors/<name>.yaml)"),
        DeclareLaunchArgument("neighbor_radius",   default_value="1.0",
                              description="[m] KDTree radius for slope/roughness"),
        DeclareLaunchArgument("slope_weight",      default_value="0.25",
                              description="Weight for slope in traversability cost (0–1)"),
        DeclareLaunchArgument("roughness_weight",  default_value="0.75",
                              description="Weight for roughness in traversability cost (0–1)"),
        DeclareLaunchArgument("voxel_size",        default_value="0.0",
                              description="[m] voxel downsample size; 0 = off"),
        DeclareLaunchArgument("fill_nn",           default_value="false",
                              description="Fill costmap gaps with nearest-neighbor interpolation"),
        DeclareLaunchArgument("forward_fov_deg",   default_value="180.0",
                              description="[deg] azimuth FOV mask (±half around forward X axis)"),
        DeclareLaunchArgument("accumulate_frames", default_value="1",
                              description="Number of lidar frames to accumulate per costmap publish"),
        DeclareLaunchArgument("r_crit",            default_value="0.2",
                              description="[m] RMS off-plane deviation above which roughness cost saturates"),
        DeclareLaunchArgument("s_crit",            default_value="20.0",
                              description="[degrees] slope above which cost is maxed out"),
        DeclareLaunchArgument("use_gravity_correction", default_value="true",
                              description="Apply IMU-based gravity alignment before slope/roughness computation"),


        # ── semantic node selection ───────────────────────────────────────────
        DeclareLaunchArgument(
            "semantic_mode", default_value="lidar_projection",
            description="Which semantic costmap producer to run: "
                        "lidar_projection (LiDAR-guided, default) or "
                        "ground_projection (IMU-tilted homography, no LiDAR density needed)"),

        # ── semantic_cost_node (lidar_projection mode) ────────────────────────
        DeclareLaunchArgument("cost_mode",           default_value="expected_cost",
                              description="Method to compute semantic cost from class probabilities: "
                                          "argmax_cost (use cost of most likely class) or "
                                          "expected_cost (compute expected cost over all classes)"),
        DeclareLaunchArgument("lidar_frame",         default_value="livox_frame"),
        DeclareLaunchArgument("camera_frame",        default_value="default_cam"),
        DeclareLaunchArgument("lidar_input_topic",   default_value="/livox/lidar",
                              description="LiDAR topic consumed by semantic_cost_node for point projection "
                                          "(e.g. /os1_cloud_node/points for Ouster/RELLIS)"),
        DeclareLaunchArgument("publish_cost_image",       default_value="false"),
        DeclareLaunchArgument("publish_class_overlay",    default_value="false"),
        DeclareLaunchArgument("publish_lidar_overlay",    default_value="false"),
        DeclareLaunchArgument("lidar_overlay_dot_radius", default_value="3"),
        DeclareLaunchArgument("point_footprint_cells",    default_value="1",
                              description="Radius in cells to dilate each lidar point in the semantic costmap"),
        DeclareLaunchArgument(
            "cam_topic_prefix", default_value="",
            description="Prefix of camera topics in the bag "
                        "(e.g. /pylon_camera_node for RELLIS). "
                        "Topics are remapped from <prefix>/image_raw → /image_raw, etc."),


        # ── fused_costmap_node ────────────────────────────────────────────────
        DeclareLaunchArgument("appearance_alpha",   default_value="0.5",
                              description="Semantic costmap weight in fusion (0–1)"),
        DeclareLaunchArgument("geometric_alpha",    default_value="0.5",
                              description="LiDAR costmap weight in fusion (0–1)"),
        DeclareLaunchArgument("max_stamp_diff_sec", default_value="0.15",
                              description="[s] max timestamp difference to consider costmaps a valid pair"),
        DeclareLaunchArgument("lidar_buffer_size",  default_value="100",
                              description="Number of lidar costmap frames to buffer for stamp matching"),
        DeclareLaunchArgument("accumulate_scans",   default_value="1",
                              description="Rolling-max window: fuse this many recent lidar frames to fill sparse grids; 1 = single frame only"),
        DeclareLaunchArgument("fused_fill_nn",      default_value="false",
                              description="Fill remaining empty cells in the fused costmap using nearest-neighbor propagation"),
        DeclareLaunchArgument("publish_costmap",    default_value="true",
                              description="Whether fused_costmap_node should publish the /fused_costmap OccupancyGrid"),
        DeclareLaunchArgument("publish_voxels",     default_value="false",
                              description="Whether fused_costmap_node should publish a VoxelGrid of the fused costmap for visualization in RViz"),
        # Per-node EMA temporal smoothing toggles. Smoothing each input upstream
        # makes the LiDAR-only / Semantic-only / Fused comparison fair (otherwise
        # only the fused map benefits from temporal stability).
        DeclareLaunchArgument("lidar_temporal_smooth",    default_value="false",
                              description="Enable EMA temporal smoothing on the lidar costmap"),
        DeclareLaunchArgument("lidar_temporal_alpha",     default_value="0.6",
                              description="Lidar EMA weight for the incoming frame (0–1); lower = smoother but laggier"),
        DeclareLaunchArgument("semantic_temporal_smooth", default_value="false",
                              description="Enable EMA temporal smoothing on the semantic costmap"),
        DeclareLaunchArgument("semantic_temporal_alpha",  default_value="0.6",
                              description="Semantic EMA weight for the incoming frame (0–1); lower = smoother but laggier"),
        DeclareLaunchArgument("fused_temporal_smooth",    default_value="false",
                              description="Enable EMA temporal smoothing on the fused costmap "
                                          "(usually disabled when both inputs are already smoothed)"),
        DeclareLaunchArgument("fused_temporal_alpha",     default_value="0.6",
                              description="Fused EMA weight for the incoming frame (0–1); lower = smoother but laggier"),

        # ── nodes ─────────────────────────────────────────────────────────────
        # Livox Mid-360 native frame has Y pointing right (opposite ROS convention).
        # roll=π flips Y and Z so livox_frame aligns with base_link (X=fwd, Y=left, Z=up).
        # Not needed for Ouster/RELLIS — the URDF handles that sensor's pose.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="livox_tf",
            output="screen",
            arguments=[
                "--x", "0", "--y", "0", "--z", "1",
                "--roll", "3.14159", "--pitch", "0", "--yaw", "0",
                "--frame-id", "base_link",
                "--child-frame-id", "livox_frame",
            ],
            condition=IfCondition(PythonExpression(["'", sensor_config, "' == 'livox_mid360'"])),
        ),

        # Madgwick AHRS filter — fuses Livox accel+gyro into a stable orientation.
        # Required because the raw /livox/imu has an empty orientation field, and
        # accelerometer-only tilt is too noisy during walking/driving.
        Node(
            package="imu_filter_madgwick",
            executable="imu_filter_madgwick_node",
            name="livox_imu_filter",
            output="screen",
            parameters=[{
                "use_mag": False,
                "world_frame": "enu",
                "publish_tf": False,
                "gain": 0.1,
            }],
            remappings=[
                ("imu/data_raw", "/livox/imu"),
                ("imu/data",     "/livox/imu_filtered"),
            ],
            condition=IfCondition(PythonExpression(["'", sensor_config, "' == 'livox_mid360'"])),
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),

        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            output="screen",
        ),

        # LiDAR traversability costmap
        # sensor_params YAML (loaded first) sets lidar_input_topic and lidar_tf_frame;
        # the dict below overrides everything else.
        Node(
            package="traversability_pkg",
            executable="lidar_cost_node",
            name="lidar_cost_node",
            output="screen",
            parameters=[sensor_params, {
                "costmap_resolution": costmap_resolution,
                "costmap_width_m":    costmap_width_m,
                "costmap_height_m":   costmap_height_m,
                "costmap_origin_x":   costmap_origin_x,
                "costmap_origin_y":   costmap_origin_y,
                "costmap_origin_z":   costmap_origin_z,
                "min_point_z":        min_point_z,
                "max_point_z":        max_point_z,
                "neighbor_radius":    neighbor_radius,
                "slope_weight":       slope_weight,
                "roughness_weight":   roughness_weight,
                "voxel_size":         voxel_size,
                "fill_nn":            fill_nn,
                "forward_fov_deg":    forward_fov_deg,
                "accumulate_frames":  accumulate_frames,
                "r_crit":                  r_crit,
                "s_crit":                  s_crit,
                "use_gravity_correction":  use_gravity_correction,
                "temporal_smooth":         lidar_temporal_smooth,
                "temporal_alpha":          lidar_temporal_alpha,
            }],
        ),

        # Semantic costmap — lidar_projection mode (default)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([launch_dir, "semantic_cost_node.launch.py"])
            ),
            launch_arguments={
                "publish_cost_image":       publish_cost_image,
                "publish_class_overlay":    publish_class_overlay,
                "publish_lidar_overlay":    publish_lidar_overlay,
                "lidar_overlay_dot_radius": lidar_overlay_dot_radius,
                "point_footprint_cells":    point_footprint_cells,
                "cost_mode":               cost_mode,
                "lidar_frame":             lidar_frame,
                "camera_frame":            camera_frame,
                "costmap_frame_id":        "base_link",
                "costmap_resolution":      costmap_resolution,
                "costmap_width_m":         costmap_width_m,
                "costmap_height_m":        costmap_height_m,
                "costmap_origin_x":        costmap_origin_x,
                "costmap_origin_y":        costmap_origin_y,
                "costmap_origin_z":        costmap_origin_z,
                "min_point_z":             min_point_z,
                "max_point_z":             max_point_z,
                "cam_topic_prefix":        cam_topic_prefix,
                "sensor_config":           sensor_config,
                "lidar_input_topic":       lidar_input_topic,
                "temporal_smooth":         semantic_temporal_smooth,
                "temporal_alpha":          semantic_temporal_alpha,
            }.items(),
            condition=IfCondition(
                PythonExpression(["'", semantic_mode, "' == 'lidar_projection'"])
            ),
        ),

        # Fused costmap
        Node(
            package="traversability_pkg",
            executable="fused_costmap_node",
            name="fused_costmap_node",
            output="screen",
            parameters=[{
                "appearance_alpha":   appearance_alpha,
                "geometric_alpha":    geometric_alpha,
                "max_stamp_diff_sec": max_stamp_diff_sec,
                "lidar_buffer_size":  lidar_buffer_size,
                "accumulate_scans":   accumulate_scans,
                "fill_nn":            fused_fill_nn,
                "publish_costmap":    publish_costmap,
                "publish_voxels":     publish_voxels,
                "temporal_smooth":    fused_temporal_smooth,
                "temporal_alpha":     fused_temporal_alpha,
            }],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
            condition=IfCondition(use_rviz),
        ),

        # Exclude /tf from playback — robot_state_publisher re-publishes joint TFs
        # from the current clock, so replaying old /tf messages causes TF_OLD_DATA.
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play",
                bag_path,
                "-r", playback_rate,
                "--qos-profile-overrides-path", qos_overrides_path,
                "--exclude-topics", "/tf",
                "--loop"
            ],
            output="screen",
            condition=IfCondition(play_bag),
        ),
    ])
