"""
Interactive LiDAR-camera extrinsic calibration node.

Subscribes to:
  /livox/lidar        (sensor_msgs/PointCloud2)
  /usb_cam/image_raw  (sensor_msgs/Image)
  /usb_cam/camera_info (sensor_msgs/CameraInfo)

Publishes:
  /calibration/overlay (sensor_msgs/Image)  — lidar points projected onto image

Tune the 6-DOF transform (x y z roll pitch yaw) via ROS2 parameters at runtime:
  ros2 param set /lidar_camera_calibration_node x 0.05
  ros2 param set /lidar_camera_calibration_node yaw 1.57

When alignment looks good, the node logs the equivalent static_transform_publisher
command to copy into your launch file.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation
from sensor_msgs_py import point_cloud2


class LidarCameraCalibrationNode(Node):
    def __init__(self):
        super().__init__("lidar_camera_calibration_node")

        # 6-DOF extrinsic: lidar → camera (translate then rotate)
        self.declare_parameter("x",     0.0)
        self.declare_parameter("y",     0.0)
        self.declare_parameter("z",     0.0)
        self.declare_parameter("roll",  0.0)
        self.declare_parameter("pitch", 0.0)
        self.declare_parameter("yaw",   0.0)

        self.declare_parameter("lidar_topic",       "/livox/lidar")
        self.declare_parameter("image_topic",       "/image_raw")
        self.declare_parameter("camera_info_topic", "/camera_info")
        # Only project points within this depth range [m]
        self.declare_parameter("min_range", 0.3)
        self.declare_parameter("max_range", 15.0)
        # Point size in the overlay image
        self.declare_parameter("point_radius", 3)

        self._bridge = CvBridge()
        self._K: np.ndarray | None = None       # 3×3 intrinsic matrix
        self._dist: np.ndarray | None = None    # distortion coefficients
        self._latest_image: np.ndarray | None = None

        sensor_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        best_effort_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_cb,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self._image_cb,
            best_effort_qos,
        )
        self.create_subscription(
            PointCloud2,
            self.get_parameter("lidar_topic").value,
            self._lidar_cb,
            sensor_qos,
        )
        self._overlay_pub = self.create_publisher(Image, "/calibration/overlay", 1)

        # Log the static TF command every 5 s so the user can copy it
        self.create_timer(5.0, self._log_tf_command)
        self.get_logger().info(
            "Calibration node ready. Tune parameters with:\n"
            "  ros2 param set /lidar_camera_calibration_node x 0.05\n"
            "View overlay in RViz2 or:  ros2 run rqt_image_view rqt_image_view"
        )

    # ── callbacks ────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self._K is None:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._dist = np.array(msg.d, dtype=np.float64)
            self.get_logger().info(f"Camera intrinsics received: fx={self._K[0,0]:.1f} fy={self._K[1,1]:.1f}")

    def _image_cb(self, msg: Image):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}", throttle_duration_sec=5.0)

    def _lidar_cb(self, msg: PointCloud2):
        if self._K is None or self._latest_image is None:
            return

        cloud_arr = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.column_stack((cloud_arr["x"], cloud_arr["y"], cloud_arr["z"])).astype(np.float32)

        # Apply extrinsic: transform lidar points into camera frame
        T_mat, R_mat = self._get_extrinsic()
        pts_cam = (pts @ R_mat.T) + T_mat  # (N, 3)

        # Keep only points in front of the camera and within range
        depth = pts_cam[:, 2]
        min_r = float(self.get_parameter("min_range").value)
        max_r = float(self.get_parameter("max_range").value)
        mask = (depth >= min_r) & (depth <= max_r)
        pts_cam = pts_cam[mask]

        if pts_cam.shape[0] == 0:
            return

        # Project onto image plane using OpenCV (handles distortion too)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        pts_2d, _ = cv2.projectPoints(
            pts_cam.astype(np.float64),
            rvec, tvec,
            self._K, self._dist,
        )
        pts_2d = pts_2d.reshape(-1, 2)

        h, w = self._latest_image.shape[:2]
        overlay = self._latest_image.copy()
        radius = int(self.get_parameter("point_radius").value)

        # Colour by depth: near=red, far=blue
        d_vals = pts_cam[:, 2]
        d_norm = np.clip((d_vals - min_r) / (max_r - min_r + 1e-6), 0.0, 1.0)

        for (px, py), dn in zip(pts_2d, d_norm):
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < w and 0 <= iy < h:
                # BGR: near=red (0,0,255), far=blue (255,0,0)
                r = int((1.0 - dn) * 255)
                b = int(dn * 255)
                cv2.circle(overlay, (ix, iy), radius, (b, 0, r), -1)

        try:
            self._overlay_pub.publish(
                self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            )
        except Exception as e:
            self.get_logger().warn(f"Overlay publish failed: {e}", throttle_duration_sec=5.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_extrinsic(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (T, R) for the current parameter values."""
        x     = float(self.get_parameter("x").value)
        y     = float(self.get_parameter("y").value)
        z     = float(self.get_parameter("z").value)
        roll  = float(self.get_parameter("roll").value)
        pitch = float(self.get_parameter("pitch").value)
        yaw   = float(self.get_parameter("yaw").value)
        T = np.array([x, y, z], dtype=np.float64)
        R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        return T, R

    def _log_tf_command(self):
        x     = float(self.get_parameter("x").value)
        y     = float(self.get_parameter("y").value)
        z     = float(self.get_parameter("z").value)
        roll  = float(self.get_parameter("roll").value)
        pitch = float(self.get_parameter("pitch").value)
        yaw   = float(self.get_parameter("yaw").value)
        self.get_logger().info(
            f"\n--- Current extrinsic (lidar → camera) ---\n"
            f"  static_transform_publisher args:\n"
            f"  --x {x:.4f} --y {y:.4f} --z {z:.4f}\n"
            f"  --roll {roll:.4f} --pitch {pitch:.4f} --yaw {yaw:.4f}\n"
            f"  --frame-id livox_frame --child-frame-id camera\n"
            f"------------------------------------------"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
