"""ROS2 node — thin wrapper around lidar_cost_processor."""

import numpy as np
import rclpy
from collections import deque
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

from .semantic_fusion_utils import OccupancyGridEma, pointcloud2_to_xyz_array
from .lidar_cost_processor import (
    compute_traversability_costs,
    compute_slope_and_roughness,
    rasterize,
    fill_gaps,
    voxel_downsample,
    COSTMAP_RESOLUTION,
    COSTMAP_WIDTH_M,
    COSTMAP_HEIGHT_M,
    COSTMAP_ORIGIN_X,
    COSTMAP_ORIGIN_Y,
    MIN_POINT_Z,
    MAX_POINT_Z,
    NEIGHBOR_RADIUS,
    SLOPE_WEIGHT,
    ROUGHNESS_WEIGHT,
)


class LidarCostNode(Node):
    def __init__(self):
        super().__init__("lidar_cost_node")

        self.declare_parameter("lidar_input_topic",   "/livox/lidar")
        self.declare_parameter("imu_topic",           "/livox/imu")
        self.declare_parameter("output_costmap_topic", "/lidar_costmap")
        self.declare_parameter("costmap_resolution",   COSTMAP_RESOLUTION)
        self.declare_parameter("costmap_width_m",      COSTMAP_WIDTH_M)
        self.declare_parameter("costmap_height_m",     COSTMAP_HEIGHT_M)
        self.declare_parameter("costmap_origin_x",     COSTMAP_ORIGIN_X)
        self.declare_parameter("costmap_origin_y",     COSTMAP_ORIGIN_Y)
        self.declare_parameter("costmap_origin_z",     -0.3)
        self.declare_parameter("min_point_z",          MIN_POINT_Z)
        self.declare_parameter("max_point_z",          MAX_POINT_Z)
        self.declare_parameter("neighbor_radius",      NEIGHBOR_RADIUS)
        self.declare_parameter("slope_weight",         SLOPE_WEIGHT)
        self.declare_parameter("roughness_weight",     ROUGHNESS_WEIGHT)
        self.declare_parameter("r_crit",              0.20)  # [m] RMS off-plane deviation above which cost is maxed out
        self.declare_parameter("s_crit",              20.0) # [degrees] slope above which cost is maxed out
        self.declare_parameter("voxel_size",           0.0)  # [m] pre-KDTree downsample; 0 = off
        # Publish the costmap in this TF frame.  base_link (ground plane) keeps
        # the 2-D grid at Z=0 on the ground rather than floating at sensor height.
        self.declare_parameter("costmap_frame_id",     "base_link")
        self.declare_parameter("fill_nn",               False)  # whether to fill gaps with nearest-neighbor interpolation
        # If set, incoming cloud is transformed from this TF frame into costmap_frame_id
        # before processing.  Leave empty to skip (e.g. Ouster with URDF already correct).
        self.declare_parameter("lidar_tf_frame",        "")
        # Keep only points within ±(forward_fov_deg/2) of the forward axis (X in base_link).
        # 360 = full scan (off), 180 = front hemisphere, etc.
        self.declare_parameter("forward_fov_deg",       180.0)
        # Number of consecutive frames to accumulate before computing the costmap.
        # Mid-360's non-repetitive scan fills in coverage over multiple frames;
        # accumulating 3-5 frames gives a much more stable, dense costmap.
        # 1 = off (single frame, default for Ouster which is already dense).
        self.declare_parameter("accumulate_frames",     1)
        self.declare_parameter("use_gravity_correction", True)
        self.declare_parameter("temporal_smooth",         False)
        self.declare_parameter("temporal_alpha",          0.4)

        p = self.get_parameter
        self.lidar_input_topic    = p("lidar_input_topic").value
        self.output_costmap_topic = p("output_costmap_topic").value
        self.resolution   = float(p("costmap_resolution").value)
        self.width_m      = float(p("costmap_width_m").value)
        self.height_m     = float(p("costmap_height_m").value)
        self.origin_x     = float(p("costmap_origin_x").value)
        self.origin_y     = float(p("costmap_origin_y").value)
        self.origin_z     = float(p("costmap_origin_z").value)
        self.min_z        = float(p("min_point_z").value)
        self.max_z        = float(p("max_point_z").value)
        self.radius       = float(p("neighbor_radius").value)
        self.slope_weight           = float(p("slope_weight").value)
        self.roughness_weight              = float(p("roughness_weight").value)
        self.r_crit              = float(p("r_crit").value)
        self.s_crit              = float(p("s_crit").value)
        self.voxel_size      = float(p("voxel_size").value)
        self.costmap_frame   = p("costmap_frame_id").value
        self.fill_nn          = bool(p("fill_nn").value)
        self.lidar_tf_frame   = p("lidar_tf_frame").value.strip()
        self.forward_fov_rad  = float(p("forward_fov_deg").value) * (np.pi / 180.0)
        n_acc = int(p("accumulate_frames").value)
        self._point_buffer: deque[np.ndarray] = deque(maxlen=max(1, n_acc))
        self.use_gravity_correction = bool(p("use_gravity_correction").value)
        self._temporal_smooth = bool(p("temporal_smooth").value)
        self._ema = OccupancyGridEma(float(p("temporal_alpha").value))

        # IMU orientation gives gravity-referenced roll/pitch (from AHRS filter),
        # independent of odometry drift.
        self._R_imu_to_world: np.ndarray | None = None

        # Static lidar→imu rotation — looked up once from TF on first use.
        # The Livox IMU is inside livox_frame, so this is typically identity.
        self._R_lidar_to_imu: np.ndarray | None = None
        # Static lidar→base_link rotation — looked up once, used to compose
        # gravity rotation when points are already in base_link frame.
        self._R_lidar_to_base: np.ndarray | None = None
        self.lidar_frame = "livox_frame"
        self.imu_frame   = "livox_frame"

        self.tf_buffer    = Buffer()
        self.tf_listener  = TransformListener(self.tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        sensor_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointCloud2, self.lidar_input_topic, self.lidar_callback, sensor_qos)
        self.create_subscription(Imu, p("imu_topic").value, self.imu_callback, sensor_qos)
        self.costmap_pub = self.create_publisher(OccupancyGrid, self.output_costmap_topic, 10)

    # ── IMU callback ──────────────────────────────────────────────────────────

    def imu_callback(self, msg: Imu):
        """Cache the latest IMU orientation and broadcast gravity-aligned TF.

        Expects the IMU topic to carry a fused orientation (e.g. from
        imu_filter_madgwick).  Raw Livox IMU messages have an empty orientation
        field and would yield identity here — run a Madgwick/complementary
        filter upstream to get a usable AHRS quaternion.
        """
        q = msg.orientation
        self._R_imu_to_world = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        R_full = self._gravity_rotation()
        if R_full is None:
            return

        # Extract only pitch+roll (strip yaw) so base_link_gravity stays
        # forward-facing regardless of heading. Finds the minimal rotation
        # that aligns base_link Z with true vertical.
        z_world_in_base = R_full.T @ np.array([0.0, 0.0, 1.0])
        z_base = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z_base, z_world_in_base)
        sin_angle = np.linalg.norm(axis)
        cos_angle = float(np.dot(z_base, z_world_in_base))
        if sin_angle < 1e-6:
            R_tilt = np.eye(3)
        else:
            axis /= sin_angle
            R_tilt = Rotation.from_rotvec(axis * np.arctan2(sin_angle, cos_angle)).as_matrix()

        quat = Rotation.from_matrix(R_tilt).as_quat()
        t = TransformStamped()
        t.header.stamp    = msg.header.stamp
        t.header.frame_id = self.costmap_frame
        t.child_frame_id  = "base_link_gravity"
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.0
        self._tf_broadcaster.sendTransform(t)

    # ── gravity rotation ──────────────────────────────────────────────────────

    def _gravity_rotation(self) -> np.ndarray | None:
        """
        Return R such that  pts_gravity_aligned = pts_lidar @ R.T

        Pipeline:
          1. lidar → imu  : static extrinsic from TF (looked up once)
          2. imu  → world : absolute orientation from IMU AHRS filter

        Returns None if either is unavailable yet.
        """
        if self._R_imu_to_world is None:
            return None

        # Look up the static lidar→imu rotation once
        if self._R_lidar_to_imu is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.imu_frame, self.lidar_frame, rclpy.time.Time()
                )
                q = tf.transform.rotation
                self._R_lidar_to_imu = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            except Exception:
                return None  # TF not ready yet

        # R_lidar_to_world = R_imu_to_world @ R_lidar_to_imu
        return self._R_imu_to_world @ self._R_lidar_to_imu

    def _gravity_rotation_from_base_link(self) -> np.ndarray | None:
        """
        Return R such that  pts_gravity_aligned = pts_base_link @ R.T

        The IMU reports orientation of livox_frame in its gravity-aligned world
        frame.  Points are already in base_link (static TF already applied).
        We need R_base_to_world:

            R_base_to_world = R_lidar_to_base.T  @  R_imu_to_world.T  @  R_imu_to_world  @  R_lidar_to_base
            ... simplifies to just composing in the right order:

            R_base_to_world = R_lidar_to_base @ R_imu_to_world @ R_lidar_to_base.T

        This expresses the IMU tilt in the base_link frame by sandwiching
        R_imu_to_world between the static mount rotations, so the static roll=π
        cancels and only the dynamic tilt remains.
        """
        if self._R_imu_to_world is None:
            return None

        if self._R_lidar_to_base is None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.costmap_frame, self.lidar_frame, rclpy.time.Time()
                )
                q = tf.transform.rotation
                self._R_lidar_to_base = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            except Exception:
                return None

        # sandwich: bring IMU tilt into base_link coordinates
        return self._R_lidar_to_base @ self._R_imu_to_world @ self._R_lidar_to_base.T

    # ── TF point transform ────────────────────────────────────────────────────

    def _transform_points(self, points_xyz: np.ndarray, source_frame: str) -> np.ndarray | None:
        """Rigid-body transform points from source_frame into costmap_frame using TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.costmap_frame, source_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF {source_frame}→{self.costmap_frame} unavailable: {e}",
                throttle_duration_sec=5.0,
            )
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.array([t.x, t.y, t.z], dtype=np.float32)
        return (points_xyz @ R.T) + T

    # ── LiDAR callback ────────────────────────────────────────────────────────

    def lidar_callback(self, msg: PointCloud2):
        points_xyz = pointcloud2_to_xyz_array(msg)

        # 1. Transform points into base_link (costmap frame) and determine the
        #    gravity rotation for slope/roughness computation.
        #
        #    a) lidar_tf_frame set (Livox): transform via static TF into base_link.
        #       The static TF encodes the fixed mount rotation but NOT dynamic tilt
        #       from handheld shake or robot pitch/roll.  R_gravity_from_base_link
        #       applies the remaining IMU-measured tilt on top, so slope/roughness
        #       are computed in a truly gravity-aligned frame.
        #
        #    b) lidar_tf_frame empty (Ouster via URDF / no static TF configured):
        #       points stay in sensor frame; R_gravity (lidar→world) is used directly.
        if self.lidar_tf_frame:
            points_xyz = self._transform_points(points_xyz, self.lidar_tf_frame)
            if points_xyz is None:
                return
            R_gravity = self._gravity_rotation_from_base_link() if self.use_gravity_correction else None
        else:
            R_gravity = self._gravity_rotation() if self.use_gravity_correction else None

        # 2. Forward FOV azimuth mask — drop points outside ±fov/2 of the X axis.
        #    Applied after TF so angles are in base_link (X=forward) frame.
        if self.forward_fov_rad < 2.0 * np.pi:
            angles = np.abs(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))
            points_xyz = points_xyz[angles <= self.forward_fov_rad / 2.0]

        # 3. Z filter
        mask = (points_xyz[:, 2] >= self.min_z) & (points_xyz[:, 2] <= self.max_z)
        points_xyz = points_xyz[mask]

        # 4. XY bounds filter — drop points outside the costmap footprint
        mask_xy = (
            (points_xyz[:, 0] >= self.origin_x) & (points_xyz[:, 0] <= self.origin_x + self.width_m) &
            (points_xyz[:, 1] >= self.origin_y) & (points_xyz[:, 1] <= self.origin_y + self.height_m)
        )
        points_xyz = points_xyz[mask_xy]

        if points_xyz.shape[0] == 0:
            return

        # 5. Accumulate frames — concatenate with previous N-1 frames.
        #    Compensates for the Mid-360's non-repetitive scan pattern where
        #    each single frame only covers a fraction of the FOV.
        self._point_buffer.append(points_xyz)
        if len(self._point_buffer) < self._point_buffer.maxlen:
            return  # wait until buffer is full before first publish
        points_xyz = np.concatenate(list(self._point_buffer), axis=0)

        # 6. Voxel downsample — applied after accumulation to merge duplicate
        #    voxels from overlapping frames and cap KDTree cost.
        points_xyz = voxel_downsample(points_xyz, self.voxel_size)

        if R_gravity is None and self.use_gravity_correction:
            self.get_logger().warn(
                "R_gravity unavailable (IMU/TF not ready) — slope computed in sensor frame",
                throttle_duration_sec=5.0,
            )

        costs = compute_traversability_costs(
            points_xyz, radius=self.radius, s_crit=self.s_crit, r_crit=self.r_crit,
            w1=self.slope_weight, w2=self.roughness_weight,
            R_gravity=R_gravity,
        )

        costmap = rasterize(
            points_xyz, costs,
            resolution=self.resolution,
            width_m=self.width_m,
            height_m=self.height_m,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            z_min=self.min_z,
            z_max=self.max_z,
        )
        
        if self.fill_nn:
            costmap_filled = fill_gaps(costmap)
        else:
            costmap_filled = costmap

        if self._temporal_smooth:
            costmap_filled = self._ema.apply(costmap_filled.astype(np.int8))

        width_cells  = int(np.ceil(self.width_m  / self.resolution))
        height_cells = int(np.ceil(self.height_m / self.resolution))

        out = OccupancyGrid()
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = self.costmap_frame
        out.info.resolution = self.resolution
        out.info.width  = width_cells
        out.info.height = height_cells
        out.info.origin.position.x = self.origin_x
        out.info.origin.position.y = self.origin_y
        out.info.origin.position.z = self.origin_z
        out.info.origin.orientation.w = 1.0
        out.data = costmap_filled.astype(np.int8).flatten().tolist()
        self.costmap_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCostNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
