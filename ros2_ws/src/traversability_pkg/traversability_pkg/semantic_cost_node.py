import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformListener

from .semantic_fusion_utils import (
    CLASS_NAMES,
    DEFAULT_SEMANTIC_COSTS,
    SUPPORTED_COST_MODES,
    COSTMAP_RESOLUTION,
    COSTMAP_WIDTH_M,
    COSTMAP_HEIGHT_M,
    COSTMAP_ORIGIN_X,
    COSTMAP_ORIGIN_Y,
    MIN_POINT_Z,
    MAX_POINT_Z,
    OccupancyGridEma,
    compute_semantic_cost_image,
    depth_filter_projected_points,
    pointcloud2_to_xyz_array,
    project_points_to_image,
    rasterize_point_costs_to_costmap,
    transform_stamped_to_matrix,
)

class SemanticCostNode(Node):
    def __init__(self):
        super().__init__("semantic_cost_node")

        self.declare_parameter("semantic_probabilities_topic", "/segmentation/probabilities")
        self.declare_parameter(
            "semantic_probability_classes_topic",
            "/segmentation/probability_classes",
        )
        self.declare_parameter("lidar_input_topic", "/livox/lidar")
        self.declare_parameter("camera_intrinsics_topic", "/camera_info")
        self.declare_parameter("output_costmap_topic", "/semantic_costmap")
        self.declare_parameter("publish_cost_image", False)
        self.declare_parameter("output_cost_image_topic", "/semantic_cost_image")
        self.declare_parameter("publish_class_overlay", False)
        self.declare_parameter("output_class_overlay_topic", "/semantic_class_overlay")
        self.declare_parameter("publish_lidar_overlay", False)
        self.declare_parameter("output_lidar_overlay_topic", "/semantic_lidar_overlay")
        self.declare_parameter("lidar_overlay_dot_radius", 3)
        self.declare_parameter("point_footprint_cells", 1)
        self.declare_parameter("cost_mode", "expected_cost")
        self.declare_parameter("costmap_resolution", COSTMAP_RESOLUTION)
        self.declare_parameter("costmap_width_m", COSTMAP_WIDTH_M)
        self.declare_parameter("costmap_height_m", COSTMAP_HEIGHT_M)
        self.declare_parameter("costmap_origin_x", COSTMAP_ORIGIN_X)
        self.declare_parameter("costmap_origin_y", COSTMAP_ORIGIN_Y)
        self.declare_parameter("costmap_origin_z", -0.3)
        self.declare_parameter("costmap_layer", "semantic_traversability")
        self.declare_parameter("min_point_z", MIN_POINT_Z)
        self.declare_parameter("max_point_z", MAX_POINT_Z)
        self.declare_parameter("costmap_frame_id", "base_link")
        self.declare_parameter("lidar_frame", "livox_frame")
        self.declare_parameter("camera_frame", "default_cam")
        self.declare_parameter("temporal_smooth", False)
        self.declare_parameter("temporal_alpha", 0.4)

        self.semantic_probabilities_topic = self.get_parameter("semantic_probabilities_topic").value
        self.semantic_probability_classes_topic = self.get_parameter("semantic_probability_classes_topic").value
        self.lidar_topic = self.get_parameter("lidar_input_topic").value
        self.camera_intrinsics_topic = self.get_parameter("camera_intrinsics_topic").value
        self.output_costmap_topic = self.get_parameter("output_costmap_topic").value
        self.publish_cost_image = self.get_parameter("publish_cost_image").value
        self.output_cost_image_topic = self.get_parameter("output_cost_image_topic").value
        self.publish_class_overlay = self.get_parameter("publish_class_overlay").value
        self.output_class_overlay_topic = self.get_parameter("output_class_overlay_topic").value
        self.publish_lidar_overlay = self.get_parameter("publish_lidar_overlay").value
        self.output_lidar_overlay_topic = self.get_parameter("output_lidar_overlay_topic").value
        self._lidar_overlay_dot_radius = int(self.get_parameter("lidar_overlay_dot_radius").value)
        self._point_footprint_cells = int(self.get_parameter("point_footprint_cells").value)
        self.cost_mode = self.get_parameter("cost_mode").value
        self.costmap_resolution = float(self.get_parameter("costmap_resolution").value)
        self.costmap_width_m = float(self.get_parameter("costmap_width_m").value)
        self.costmap_height_m = float(self.get_parameter("costmap_height_m").value)
        self.costmap_origin_x = float(self.get_parameter("costmap_origin_x").value)
        self.costmap_origin_y = float(self.get_parameter("costmap_origin_y").value)
        self.costmap_origin_z = float(self.get_parameter("costmap_origin_z").value)
        self.costmap_layer    = self.get_parameter("costmap_layer").value
        self.costmap_frame    = self.get_parameter("costmap_frame_id").value
        self.min_point_z = float(self.get_parameter("min_point_z").value)
        self.max_point_z = float(self.get_parameter("max_point_z").value)
        self._temporal_smooth = bool(self.get_parameter("temporal_smooth").value)
        self._ema = OccupancyGridEma(float(self.get_parameter("temporal_alpha").value))

        if self.cost_mode not in SUPPORTED_COST_MODES:
            raise ValueError(
                f"Unsupported cost_mode '{self.cost_mode}'. Expected one of {SUPPORTED_COST_MODES}."
            )

        self.class_names = CLASS_NAMES
        self.semantic_costs = DEFAULT_SEMANTIC_COSTS.copy()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.lidar_frame = self.get_parameter("lidar_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.costmap_size_x = int(np.ceil(self.costmap_width_m / self.costmap_resolution))
        self.costmap_size_y = int(np.ceil(self.costmap_height_m / self.costmap_resolution))

        self.latest_probabilities_msg = None
        self.latest_probabilities = None
        self.latest_probability_classes = None
        self.camera_info = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self._last_warning_time_ns: dict[str, int] = {}

        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.semantic_probabilities_subscriber = self.create_subscription(
            Image,
            self.semantic_probabilities_topic,
            self.semantic_probabilities_callback,
            sensor_qos,
        )
        self.semantic_probability_classes_subscriber = self.create_subscription(
            Image,
            self.semantic_probability_classes_topic,
            self.semantic_probability_classes_callback,
            sensor_qos,
        )
        self.lidar_subscriber = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            sensor_qos,
        )
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            self.camera_intrinsics_topic,
            self.camera_info_callback,
            sensor_qos,
        )

        self.costmap_publisher = self.create_publisher(OccupancyGrid, self.output_costmap_topic, 10)
        self.cost_image_publisher = None
        if self.publish_cost_image:
            self.cost_image_publisher = self.create_publisher(Image, self.output_cost_image_topic, 10)
        self.class_overlay_publisher = None
        if self.publish_class_overlay:
            self.class_overlay_publisher = self.create_publisher(Image, self.output_class_overlay_topic, 10)
        self.lidar_overlay_publisher = None
        self._latest_camera_image: np.ndarray | None = None
        self._latest_camera_image_msg = None
        if self.publish_lidar_overlay:
            self.lidar_overlay_publisher = self.create_publisher(Image, self.output_lidar_overlay_topic, 10)
            self.create_subscription(Image, "/image_raw", self._camera_image_callback, sensor_qos)

    def _warn_throttled(self, key: str, message: str, throttle_seconds: float = 5.0):
        """
        Log a warning message, but throttle it so that it only appears at most once every `throttle_seconds` for each unique `key`.
        """
        now_ns = self.get_clock().now().nanoseconds
        previous_ns = self._last_warning_time_ns.get(key)
        throttle_ns = int(throttle_seconds * 1e9)
        if previous_ns is None or now_ns - previous_ns >= throttle_ns:
            self.get_logger().warn(message)
            self._last_warning_time_ns[key] = now_ns

    def _get_active_lidar_frame(self, lidar_msg: PointCloud2 | None = None) -> str:
        """
    Determine the active lidar frame to use for transformations. If a lidar message is provided and contains a valid frame_id, use that. 
    Otherwise, fall back to the default lidar_frame parameter.
        """
        if lidar_msg is not None and lidar_msg.header.frame_id:
            return lidar_msg.header.frame_id
        return self.lidar_frame

    def _get_active_camera_frame(self) -> str:
        """ 
        Determine the active camera frame to use for transformations. If camera info has been received and contains a valid frame_id, use that.
        """

        if self.camera_info is not None and self.camera_info.header.frame_id:
            return self.camera_info.header.frame_id
        return self.camera_frame

    def _quaternion_to_rotation_matrix(self, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """
        Convert a quaternion in (w, x, y, z) format to a 3x3 rotation matrix."""
        q = np.asarray(quaternion_wxyz, dtype=np.float64)
        norm = np.linalg.norm(q)
        w, x, y, z = q / norm
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def _transform_points_to_costmap_frame(self, points: np.ndarray, lidar_msg: PointCloud2) -> np.ndarray:
        """Transform points from the lidar message frame into the costmap frame (base_link).
        Falls back to the original points if the TF is unavailable."""
        lidar_frame = self._get_active_lidar_frame(lidar_msg)
        if lidar_frame == self.costmap_frame:
            return points
        if not self.tf_buffer.can_transform(self.costmap_frame, lidar_frame, Time()):
            return points
        try:
            tf_msg = self.tf_buffer.lookup_transform(self.costmap_frame, lidar_frame, Time())
        except Exception:
            return points
        T = transform_stamped_to_matrix(tf_msg)
        ones = np.ones((len(points), 1), dtype=np.float64)
        pts_h = np.hstack([points.astype(np.float64), ones])
        return (T @ pts_h.T).T[:, :3].astype(np.float32)

    def _get_lidar_to_camera_transform_matrix(self, lidar_msg: PointCloud2) -> np.ndarray | None:
        """
        Get the transformation matrix from the lidar frame to the camera frame. This is needed to project lidar points into the camera image.
        The method first determines the active lidar and camera frames, then checks if the transform is available
        and retrieves it. If successful, it converts the transform to a 4x4 matrix. If any step fails, it returns None.
        """
        lidar_frame = self._get_active_lidar_frame(lidar_msg)
        camera_frame = self._get_active_camera_frame()

        if not self.tf_buffer.can_transform(camera_frame, lidar_frame, Time()):
            self.get_logger().warn(
                f"Transform not available yet: {lidar_frame} -> {camera_frame}"
            )
            return None

        try:
            tf_msg = self.tf_buffer.lookup_transform(camera_frame, lidar_frame, Time())
        except Exception as error:
            self.get_logger().warn(
                f"Failed to get transform from {lidar_frame} to {camera_frame}: {error}"
            )
            return None
        return transform_stamped_to_matrix(tf_msg)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)

    def semantic_probabilities_callback(self, msg: Image):
        self.latest_probabilities_msg = msg
        self.latest_probabilities = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._publish_current_cost_image()

    def semantic_probability_classes_callback(self, msg: Image):
        self.latest_probability_classes = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._publish_current_cost_image()
        self._publish_class_overlay(msg)

    def lidar_callback(self, msg: PointCloud2):
        """
        Main callback for processing incoming lidar data. This method performs the following steps:
        1. Checks if the latest semantic probabilities and camera info are available. If not,
           it returns early since the costmap cannot be computed without them.
        2. Projects the lidar points into the semantic image using the current transform and camera intrinsics. If the projection fails (e.g., due to missing transform), it returns early.
        3. Computes the current cost image from the semantic probabilities and classes. If this fails (e.g., due to missing probabilities), it returns early.
        4. If there are projected points, it retrieves their corresponding costs from the cost image and rasterizes them into a costmap
           grid. If no points are projected, it creates an empty costmap grid and logs a warning.
        5. Finally, it publishes the costmap as an OccupancyGrid message.
        """
        if self.latest_probabilities is None or self.camera_info is None:
            return

        projection = self._project_lidar_into_semantics(msg)
        if projection is None:
            return

        pixel_points, visible_lidar_points, visible_camera_points, distances = projection
        pixel_points, visible_lidar_points, visible_camera_points, distances = \
            depth_filter_projected_points(pixel_points, visible_lidar_points, visible_camera_points, distances)
        cost_image = self._compute_current_cost_image()
        if cost_image is None:
            return

        if pixel_points.shape[0] == 0:
            self._warn_throttled(
                "no_projected_points",
                "No lidar points projected into the semantic image. "
                "The semantic costmap will stay empty until TF, camera intrinsics, and frame selection line up.",
            )
            costmap_grid = np.zeros((self.costmap_size_y, self.costmap_size_x), dtype=np.uint8)
            point_costs = np.empty(0, dtype=np.float32)
        else:
            point_costs = cost_image[pixel_points[:, 1], pixel_points[:, 0]]
            costmap_grid = rasterize_point_costs_to_costmap(
                self._transform_points_to_costmap_frame(visible_lidar_points, msg),
                point_costs,
                self.costmap_resolution,
                self.costmap_size_x,
                self.costmap_size_y,
                self.costmap_origin_x,
                self.costmap_origin_y,
                self.min_point_z,
                self.max_point_z,
                self._point_footprint_cells,
            )
            if not np.any(costmap_grid):
                self._warn_throttled(
                    "no_costmap_cells",
                    "Projected lidar points were found in the semantic image, but none survived the costmap "
                    "bounds or z filtering. Check costmap origin/size and min_point_z/max_point_z.",
                )

        self._publish_costmap(costmap_grid, msg)

        if self.publish_lidar_overlay and pixel_points.shape[0] > 0:
            self._publish_lidar_overlay(pixel_points, point_costs, projection, msg)

    def _project_lidar_into_semantics(self, msg: PointCloud2):
        transform_matrix = self._get_lidar_to_camera_transform_matrix(msg)
        if transform_matrix is None:
            return None

        semantic_image_shape = self._get_semantic_image_shape()
        projection_image_shape = self._get_projection_image_shape()
        if semantic_image_shape is None or projection_image_shape is None:
            return None

        points = pointcloud2_to_xyz_array(msg)
        pixel_points, visible_lidar_points, visible_camera_points, distances = project_points_to_image(
            points,
            transform_matrix,
            self.camera_matrix,
            self.dist_coeffs,
            projection_image_shape[1],
            projection_image_shape[0],
        )
        if pixel_points.shape[0] == 0:
            return pixel_points, visible_lidar_points, visible_camera_points, distances

        if projection_image_shape == semantic_image_shape:
            return pixel_points, visible_lidar_points, visible_camera_points, distances

        scale_x = semantic_image_shape[1] / projection_image_shape[1]
        scale_y = semantic_image_shape[0] / projection_image_shape[0]
        semantic_pixels = pixel_points.astype(np.float64)
        semantic_pixels[:, 0] = np.floor(semantic_pixels[:, 0] * scale_x)
        semantic_pixels[:, 1] = np.floor(semantic_pixels[:, 1] * scale_y)

        in_bounds = (
            (semantic_pixels[:, 0] >= 0) &
            (semantic_pixels[:, 0] < semantic_image_shape[1]) &
            (semantic_pixels[:, 1] >= 0) &
            (semantic_pixels[:, 1] < semantic_image_shape[0])
        )
        return (
            semantic_pixels[in_bounds].astype(np.int32),
            visible_lidar_points[in_bounds],
            visible_camera_points[in_bounds],
            distances[in_bounds],
        )

    def _get_semantic_image_shape(self) -> tuple[int, int] | None:
        if self.latest_probabilities is not None:
            return self.latest_probabilities.shape[:2]
        if self.latest_probability_classes is not None:
            return self.latest_probability_classes.shape[:2]
        return None

    def _get_projection_image_shape(self) -> tuple[int, int] | None:
        if self.camera_info is not None and self.camera_info.height > 0 and self.camera_info.width > 0:
            return (int(self.camera_info.height), int(self.camera_info.width))
        return self._get_semantic_image_shape()

    def _compute_current_cost_image(self) -> np.ndarray | None:
        probabilities, probability_classes = self._get_probability_inputs_if_usable()
        if probabilities is None:
            return None

        return compute_semantic_cost_image(
            self.semantic_costs,
            self.cost_mode,
            probabilities=probabilities,
            probability_classes=probability_classes,
        )

    def _get_probability_inputs_if_usable(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.latest_probabilities is None:
            return None, None

        probabilities = np.asarray(self.latest_probabilities)
        if probabilities.ndim != 3:
            self.get_logger().warn(
                "Semantic probabilities image must be HxWxC; ignoring probabilities for now."
            )
            return None, None

        if self.latest_probability_classes is None:
            self.get_logger().warn(
                "Waiting for semantic probability classes before computing cost image."
            )
            return None, None

        probability_classes = np.asarray(self.latest_probability_classes)
        if probability_classes.shape != probabilities.shape:
            self._warn_throttled(
                "probability_shape_mismatch",
                "Semantic probability classes shape does not match the probabilities image; "
                f"got probabilities {probabilities.shape} vs classes {probability_classes.shape}.",
            )
            return None, None

        return probabilities, probability_classes

    def _publish_current_cost_image(self):
        if not self.publish_cost_image or self.cost_image_publisher is None:
            return

        cost_image = self._compute_current_cost_image()
        if cost_image is None or self.latest_probabilities_msg is None:
            return

        cost_u8 = np.clip(cost_image * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(cost_u8, cv2.COLORMAP_PLASMA)  # yellow=safe, blue/magenta=lethal
        cost_msg = self.bridge.cv2_to_imgmsg(colored, encoding="bgr8")
        cost_msg.header = self.latest_probabilities_msg.header
        self.cost_image_publisher.publish(cost_msg)

    # BGR colours per class (matches CLASS_NAMES order, train_id 0-18).
    # Source RGB values from RELLIS-3D label definition — flipped to BGR for OpenCV.
    _CLASS_COLORS_BGR = np.array([
        (  0, 128, 255),  # dirt     train_id=0   RGB [255,128,0]
        (  0, 102,   0),  # grass    train_id=1   RGB [0,102,0]
        (  0, 255,   0),  # tree     train_id=2   RGB [0,255,0]
        (153, 153,   0),  # pole     train_id=3   RGB [0,153,153]
        (255, 128,   0),  # water    train_id=4   RGB [0,128,255]
        (255,   0,   0),  # sky      train_id=5   RGB [0,0,255]
        (  0, 255, 255),  # vehicle  train_id=6   RGB [255,255,0]
        (127,   0, 255),  # object   train_id=7   RGB [255,0,127]
        ( 64,  64,  64),  # asphalt  train_id=8   RGB [64,64,64]
        (  0,   0, 255),  # building train_id=9   RGB [255,0,0]
        (  0,   0, 102),  # log      train_id=10  RGB [102,0,0]
        (255, 153, 204),  # person   train_id=11  RGB [204,153,255]
        (204,   0, 102),  # fence    train_id=12  RGB [102,0,204]
        (204, 153, 255),  # bush     train_id=13  RGB [255,153,204]
        (170, 170, 170),  # concrete train_id=14  RGB [170,170,170]
        (255, 121,  41),  # barrier  train_id=15  RGB [41,121,255]
        (239, 255, 134),  # puddle   train_id=16  RGB [134,255,239]
        ( 34,  66,  99),  # mud      train_id=17  RGB [99,66,34]
        (138,  22, 110),  # rubble   train_id=18  RGB [110,22,138]
    ], dtype=np.uint8)

    def _publish_class_overlay(self, msg: Image) -> None:
        if not self.publish_class_overlay or self.class_overlay_publisher is None:
            return
        if self.latest_probability_classes is None:
            return

        top1 = self.latest_probability_classes[:, :, 0].astype(np.int32)  # (H, W)
        H, W = top1.shape
        overlay = np.zeros((H, W, 3), dtype=np.uint8)

        n_colors = len(self._CLASS_COLORS_BGR)
        valid = (top1 >= 0) & (top1 < n_colors)
        overlay[valid] = self._CLASS_COLORS_BGR[top1[valid]]
        overlay[~valid] = (50, 50, 50)  # unknown / masked → dark grey

        out = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        out.header = msg.header
        self.class_overlay_publisher.publish(out)

    def _camera_image_callback(self, msg: Image) -> None:
        self._latest_camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._latest_camera_image_msg = msg

    def _publish_lidar_overlay(self, pixel_points: np.ndarray, point_costs: np.ndarray,
                                projection, lidar_msg: PointCloud2) -> None:
        """Draw cost-colored dots (green=safe, red=lethal) on the camera image and publish."""
        if self._latest_camera_image is None or self.lidar_overlay_publisher is None:
            return

        img = self._latest_camera_image.copy()
        img_h, img_w = img.shape[:2]

        # pixel_points are in semantic image coordinates — scale to camera image size
        sem_shape = self._get_semantic_image_shape()
        if sem_shape is None:
            return
        sem_h, sem_w = sem_shape
        scale_x = img_w / sem_w
        scale_y = img_h / sem_h
        r = self._lidar_overlay_dot_radius

        # Pre-build HSV→BGR lookup for 61 hue values
        hues = np.arange(61, dtype=np.uint8)
        hsv_lut = np.stack([hues, np.full(61, 255, dtype=np.uint8),
                             np.full(61, 255, dtype=np.uint8)], axis=1)[:, np.newaxis, :]
        bgr_lut = cv2.cvtColor(hsv_lut, cv2.COLOR_HSV2BGR)[:, 0, :]  # (61, 3)

        costs_arr = np.asarray(point_costs, dtype=np.float32)
        hue_idxs = np.clip(np.round((1.0 - costs_arr) * 60).astype(int), 0, 60)

        for i, (px, py) in enumerate(pixel_points):
            u = int(round(float(px) * scale_x))
            v = int(round(float(py) * scale_y))
            if 0 <= u < img_w and 0 <= v < img_h:
                color = bgr_lut[hue_idxs[i]].tolist()
                cv2.circle(img, (u, v), r, color, -1)

        out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out.header = lidar_msg.header
        self.lidar_overlay_publisher.publish(out)

    def _publish_costmap(self, costmap_grid: np.ndarray, lidar_msg: PointCloud2):
        metadata = MapMetaData()
        metadata.map_load_time = lidar_msg.header.stamp
        metadata.resolution = float(self.costmap_resolution)
        metadata.width = self.costmap_size_x
        metadata.height = self.costmap_size_y
        metadata.origin.position.x = self.costmap_origin_x
        metadata.origin.position.y = self.costmap_origin_y
        metadata.origin.position.z = self.costmap_origin_z
        metadata.origin.orientation.w = 1.0

        occupancy_values = costmap_grid.astype(np.int8)
        if self._temporal_smooth:
            occupancy_values = self._ema.apply(occupancy_values)

        costmap_msg = OccupancyGrid()
        costmap_msg.header = lidar_msg.header
        costmap_msg.header.frame_id = self.costmap_frame
        costmap_msg.info = metadata
        costmap_msg.data = occupancy_values.reshape(-1).tolist()

        self.costmap_publisher.publish(costmap_msg)


def main():
    rclpy.init()
    node = SemanticCostNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
