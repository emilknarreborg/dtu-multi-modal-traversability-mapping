import cv2
import matplotlib.cm as cm
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from traversability_pkg.lidar_cost_processor import load_camera_info, load_extrinsics
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformListener

from traversability_pkg.semantic_fusion_utils import (
    CLASS_NAMES,
    DEFAULT_SEMANTIC_COSTS,
    SUPPORTED_COST_MODES,
    compute_semantic_cost_image,
    pointcloud2_to_xyz_array,
    project_points_to_image,
    transform_stamped_to_matrix,
)


class SemanticLidarDebugNode(Node):
    OVERLAY_MODE_NONE = "none"
    OVERLAY_MODE_SINGLE_COSTMAP = "single_costmap"
    OVERLAY_MODE_ALL_COST_VISUALIZATIONS = "all_cost_visualizations"
    OVERLAY_MODE_LIDAR = "lidar_overlay"

    def __init__(self):
        super().__init__("semantic_lidar_debug_node")

        self.declare_parameter("semantic_probabilities_topic", "/pylon_camera_node/segmentation/probabilities")
        self.declare_parameter(
            "semantic_probability_classes_topic",
            "/pylon_camera_node/segmentation/probability_classes",
        )
        self.declare_parameter("image_input_topic", "/pylon_camera_node/image_raw")
        self.declare_parameter("lidar_overlay_output_topic", "/lidar_image_overlay")
        self.declare_parameter("lidar_input_topic", "/os1_cloud_node/points")
        self.declare_parameter("image_costmap_topic", "/image_costmap_visualization")
        self.declare_parameter("argmax_cost_visualization_topic", "/semantic_debug/argmax_cost")
        self.declare_parameter("expected_cost_visualization_topic", "/semantic_debug/expected_cost")
        self.declare_parameter(
            "penalized_cost_visualization_topic",
            "/semantic_debug/expected_cost_with_uncertainty_penalty",
        )
        self.declare_parameter("publish_image_costmap", False)
        self.declare_parameter("publish_image_costmap_blended", False)
        self.declare_parameter("publish_all_cost_visualizations", False)
        self.declare_parameter("publish_lidar_overlay", True)
        self.declare_parameter("camera_intrinsics_topic", "/pylon_camera_node/camera_info")
        self.declare_parameter("cost_mode", "expected_cost_with_uncertainty_penalty")
        self.declare_parameter("uncertainty_penalty", 0.25)

        self.semantic_probabilities_topic = self.get_parameter("semantic_probabilities_topic").value
        self.semantic_probability_classes_topic = self.get_parameter("semantic_probability_classes_topic").value
        self.image_input_topic = self.get_parameter("image_input_topic").value
        self.lidar_overlay_output_topic = self.get_parameter("lidar_overlay_output_topic").value
        self.lidar_topic = self.get_parameter("lidar_input_topic").value
        self.image_costmap_topic = self.get_parameter("image_costmap_topic").value
        self.argmax_cost_visualization_topic = self.get_parameter("argmax_cost_visualization_topic").value
        self.expected_cost_visualization_topic = self.get_parameter("expected_cost_visualization_topic").value
        self.penalized_cost_visualization_topic = self.get_parameter("penalized_cost_visualization_topic").value
        self.publish_image_costmap = self.get_parameter("publish_image_costmap").value
        self.publish_image_costmap_blended = self.get_parameter("publish_image_costmap_blended").value
        self.publish_all_cost_visualizations = self.get_parameter("publish_all_cost_visualizations").value
        self.publish_lidar_overlay = self.get_parameter("publish_lidar_overlay").value
        self.camera_intrinsics_topic = self.get_parameter("camera_intrinsics_topic").value
        self.cost_mode = self.get_parameter("cost_mode").value
        self.uncertainty_penalty = float(self.get_parameter("uncertainty_penalty").value)
        self.overlay_mode = self._select_overlay_mode()

        if self.cost_mode not in SUPPORTED_COST_MODES:
            raise ValueError(
                f"Unsupported cost_mode '{self.cost_mode}'. Expected one of {SUPPORTED_COST_MODES}."
            )

        self.class_names = CLASS_NAMES
        self.semantic_costs = DEFAULT_SEMANTIC_COSTS.copy()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.lidar_frame = "ouster1/os1_lidar"
        self.camera_frame = "default_cam"

        self.latest_probabilities_msg = None
        self.latest_probabilities = None
        self.latest_probability_classes_msg = None
        self.latest_probability_classes = None
        self.latest_camera_msg = None
        self.latest_lidar_msg = None
        self.last_published_semantic_pair = None

        self.camera_info = None
        self.camera_matrix = None
        self.dist_coeffs = None

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
        self.camera_image_subscriber = self.create_subscription(
            Image,
            self.image_input_topic,
            self.camera_image_callback,
            sensor_qos,
        )

        self.lidar_overlay_publisher = None
        self.image_costmap_publisher = None
        self.argmax_cost_visualization_publisher = None
        self.expected_cost_visualization_publisher = None
        self.penalized_cost_visualization_publisher = None
        if self.overlay_mode == self.OVERLAY_MODE_LIDAR:
            self.lidar_overlay_publisher = self.create_publisher(Image, self.lidar_overlay_output_topic, 10)
        elif self.overlay_mode == self.OVERLAY_MODE_SINGLE_COSTMAP:
            self.image_costmap_publisher = self.create_publisher(Image, self.image_costmap_topic, 10)
        elif self.overlay_mode == self.OVERLAY_MODE_ALL_COST_VISUALIZATIONS:
            self.argmax_cost_visualization_publisher = self.create_publisher(Image, self.argmax_cost_visualization_topic, 10)
            self.expected_cost_visualization_publisher = self.create_publisher(Image, self.expected_cost_visualization_topic, 10)
            self.penalized_cost_visualization_publisher = self.create_publisher(Image, self.penalized_cost_visualization_topic, 10)

    def _select_overlay_mode(self) -> str:
        enabled_modes = []
        if self.publish_all_cost_visualizations:
            enabled_modes.append(self.OVERLAY_MODE_ALL_COST_VISUALIZATIONS)
        if self.publish_image_costmap or self.publish_image_costmap_blended:
            enabled_modes.append(self.OVERLAY_MODE_SINGLE_COSTMAP)
        if self.publish_lidar_overlay:
            enabled_modes.append(self.OVERLAY_MODE_LIDAR)

        if not enabled_modes:
            return self.OVERLAY_MODE_NONE

        selected_mode = enabled_modes[0]
        if len(enabled_modes) > 1:
            self.get_logger().warn(
                "Multiple debug overlay modes were enabled. "
                f"Using only '{selected_mode}' and disabling {enabled_modes[1:]}."
            )

        self.publish_all_cost_visualizations = selected_mode == self.OVERLAY_MODE_ALL_COST_VISUALIZATIONS
        single_costmap_enabled = selected_mode == self.OVERLAY_MODE_SINGLE_COSTMAP
        self.publish_image_costmap = self.publish_image_costmap and single_costmap_enabled
        self.publish_image_costmap_blended = self.publish_image_costmap_blended and single_costmap_enabled
        self.publish_lidar_overlay = selected_mode == self.OVERLAY_MODE_LIDAR
        return selected_mode

    def _get_lidar_to_camera_transform(self) -> TransformStamped | None:
        if not self.tf_buffer.can_transform(self.camera_frame, self.lidar_frame, Time()):
            self.get_logger().warn(
                f"Transform not available yet: {self.lidar_frame} -> {self.camera_frame}"
            )
            return None

        try:
            return self.tf_buffer.lookup_transform(self.camera_frame, self.lidar_frame, Time())
        except Exception as error:
            self.get_logger().warn(
                f"Failed to get transform from {self.lidar_frame} to {self.camera_frame}: {error}"
            )
            return None

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)

    def camera_image_callback(self, msg: Image):
        self.latest_camera_msg = msg

    def semantic_probabilities_callback(self, msg: Image):
        self.latest_probabilities_msg = msg
        self.latest_probabilities = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._publish_cost_overlays_if_ready()

    def semantic_probability_classes_callback(self, msg: Image):
        self.latest_probability_classes_msg = msg
        self.latest_probability_classes = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._publish_cost_overlays_if_ready()

    def lidar_callback(self, msg: PointCloud2):
        self.latest_lidar_msg = msg
        if self.overlay_mode != self.OVERLAY_MODE_LIDAR:
            return

        if self.latest_camera_msg is None or self.camera_info is None:
            return

        projected = self._process_lidar_projection(msg)
        if projected is None:
            return

        pixel_points, _, _, distances = projected
        self._publish_lidar_overlay(pixel_points, distances)

    def _publish_cost_overlays_if_ready(self):
        if self.overlay_mode == self.OVERLAY_MODE_SINGLE_COSTMAP:
            self._publish_current_image_costmap()
        elif self.overlay_mode == self.OVERLAY_MODE_ALL_COST_VISUALIZATIONS:
            self._publish_all_cost_visualizations()

    def _semantic_pair_ready(self) -> bool:
        if self.latest_probabilities_msg is None or self.latest_probability_classes_msg is None:
            return False

        semantic_pair = (
            self._header_stamp_ns(self.latest_probabilities_msg.header),
            self._header_stamp_ns(self.latest_probability_classes_msg.header),
        )
        if semantic_pair == self.last_published_semantic_pair:
            return False

        self.last_published_semantic_pair = semantic_pair
        return True

    def _header_stamp_ns(self, header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def _publish_current_image_costmap(self):
        if self.image_costmap_publisher is None or not self._semantic_pair_ready():
            return

        costmap = self._compute_cost_image(self.cost_mode)
        if costmap is None or self.latest_probabilities_msg is None:
            return

        image_msg = self._cost_image_to_msg(costmap)
        image_msg.header = self.latest_probabilities_msg.header
        self.image_costmap_publisher.publish(image_msg)

    def _publish_all_cost_visualizations(self):
        if self.latest_probabilities_msg is None or not self._semantic_pair_ready():
            return

        publisher_configs = (
            (self.argmax_cost_visualization_publisher, "argmax_cost"),
            (self.expected_cost_visualization_publisher, "expected_cost"),
            (self.penalized_cost_visualization_publisher, "expected_cost_with_uncertainty_penalty"),
        )
        for publisher, cost_mode in publisher_configs:
            if publisher is None:
                continue
            cost_image = self._compute_cost_image(cost_mode)
            if cost_image is None:
                continue
            image_msg = self._cost_image_to_msg(cost_image)
            image_msg.header = self.latest_probabilities_msg.header
            publisher.publish(image_msg)

    def _compute_cost_image(self, cost_mode: str) -> np.ndarray | None:
        probabilities, probability_classes = self._get_probability_inputs_if_usable()
        if probabilities is None:
            return None

        return compute_semantic_cost_image(
            self.semantic_costs,
            cost_mode,
            probabilities=probabilities,
            probability_classes=probability_classes,
            uncertainty_penalty=self.uncertainty_penalty,
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
            return None, None

        probability_classes = np.asarray(self.latest_probability_classes)
        if probability_classes.shape != probabilities.shape:
            self.get_logger().warn(
                "Semantic probability classes shape does not match the probabilities image; ignoring inputs for now."
            )
            return None, None

        return probabilities, probability_classes

    def _cost_image_to_msg(self, cost_image: np.ndarray) -> Image:
        grayscale_costmap = (np.clip(cost_image, 0.0, 1.0) * 255.0).astype(np.uint8)
        if self.publish_image_costmap_blended and self.latest_camera_msg is not None:
            rgba = cm.get_cmap("RdYlGn_r")(np.clip(cost_image, 0.0, 1.0))
            color_costmap = (rgba[..., :3] * 255).astype(np.uint8)
            color_costmap = cv2.cvtColor(color_costmap, cv2.COLOR_RGB2BGR)
            camera_image = self.bridge.imgmsg_to_cv2(self.latest_camera_msg, desired_encoding="bgr8")
            if camera_image.shape[:2] != color_costmap.shape[:2]:
                camera_image = cv2.resize(
                    camera_image,
                    (color_costmap.shape[1], color_costmap.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            blended = cv2.addWeighted(color_costmap, 0.5, camera_image, 0.5, 0.0)
            return self.bridge.cv2_to_imgmsg(blended, encoding="bgr8")
        return self.bridge.cv2_to_imgmsg(grayscale_costmap, encoding="mono8")

    def _process_lidar_projection(self, msg: PointCloud2):
        tf_msg = self._get_lidar_to_camera_transform()
        if tf_msg is None:
            return None

        image_shape = self._get_projection_image_shape()
        if image_shape is None:
            return None

        rt_matrix = transform_stamped_to_matrix(tf_msg)
        points = pointcloud2_to_xyz_array(msg)
        return project_points_to_image(
            points,
            rt_matrix,
            self.camera_matrix,
            self.dist_coeffs,
            image_shape[1],
            image_shape[0],
        )

    def _get_projection_image_shape(self) -> tuple[int, int] | None:
        if self.overlay_mode == self.OVERLAY_MODE_LIDAR and self.latest_camera_msg is not None:
            return (self.latest_camera_msg.height, self.latest_camera_msg.width)
        if self.latest_probabilities is not None:
            return self.latest_probabilities.shape[:2]
        if self.latest_probability_classes is not None:
            return self.latest_probability_classes.shape[:2]
        if self.latest_camera_msg is not None:
            return (self.latest_camera_msg.height, self.latest_camera_msg.width)
        return None

    def _publish_lidar_overlay(self, pixel_points: np.ndarray, distances: np.ndarray):
        if self.lidar_overlay_publisher is None:
            return

        background_image = self.bridge.imgmsg_to_cv2(self.latest_camera_msg, desired_encoding="bgr8")
        overlay_msg = self._lidar_overlay(pixel_points.T, self._depth_color(distances), background_image)
        self.lidar_overlay_publisher.publish(overlay_msg)

    def _depth_color(self, val, min_d=0, max_d=70):
        np.clip(val, min_d, max_d, out=val)
        return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

    def _lidar_overlay(self, points, color, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for i in range(points.shape[1]):
            cv2.circle(
                hsv,
                (np.int32(points[0, i]), np.int32(points[1, i])),
                2,
                (int(color[i]), 255, 255),
                -1,
            )

        overlay_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")

def main():
    rclpy.init()
    node = SemanticLidarDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()