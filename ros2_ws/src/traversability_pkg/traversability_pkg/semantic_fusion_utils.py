import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2

COSTMAP_RESOLUTION = 0.2   # [m/cell]
COSTMAP_WIDTH_M    = 40.0
COSTMAP_HEIGHT_M   = 20.0
COSTMAP_ORIGIN_X   = -20.0
COSTMAP_ORIGIN_Y   = -10.0
MIN_POINT_Z        = -2.0
MAX_POINT_Z        =  2.0

COST_FREE = 0
COST_LETHAL = 100


class OccupancyGridEma:
    """
    Per-cell exponential moving average for OccupancyGrid-style int8 arrays.

    EMA: state ← α·new + (1−α)·state.  Lower α → smoother, laggier.
    The first frame initialises the state and is returned unchanged.
    Output is rounded and clipped to the OccupancyGrid range [-1, 100].

    Reused by lidar_cost_node, semantic_cost_node and fused_costmap_node so
    smoothing is applied identically wherever it is enabled.
    """

    def __init__(self, alpha: float):
        self._alpha: float = float(alpha)
        self._state: np.ndarray | None = None

    def apply(self, grid: np.ndarray) -> np.ndarray:
        if self._state is None or self._state.shape != grid.shape:
            self._state = grid.astype(np.float32)
            return grid
        self._state = self._alpha * grid.astype(np.float32) \
                      + (1.0 - self._alpha) * self._state
        return np.clip(np.rint(self._state), -1, 100).astype(np.int8)


CLASS_NAMES = [
    "dirt", "grass", "tree", "pole", "water", "sky", "vehicle",
    "object", "asphalt", "building", "log", "person", "fence",
    "bush", "concrete", "barrier", "puddle", "mud", "rubble",
]

DEFAULT_SEMANTIC_COSTS = np.array([
    0.10, 0.25, 1.00, 1.00, 1.00, 1.00, 1.00, 
    0.80, 0.05, 1.00, 0.60, 1.00, 1.00, 
    0.80, 0.05, 1.00, 0.40, 0.50, 0.80,
], dtype=np.float32)


SUPPORTED_COST_MODES = (
    "argmax_cost",
    "expected_cost",
)

INVALID_CLASS_ID = 255


def transform_stamped_to_matrix(tf_msg: TransformStamped) -> np.ndarray:
    """Convert a ROS TransformStamped into a 4x4 homogeneous transform matrix."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation

    x, y, z, w = q.x, q.y, q.z, q.w

    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [t.x, t.y, t.z]
    return transform


def pointcloud2_to_xyz_array(cloud_msg) -> np.ndarray:
    """Convert PointCloud2 to an (N, 3) float32 XYZ array."""
    cloud_arr = point_cloud2.read_points(
        cloud_msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    return np.column_stack((
        cloud_arr["x"],
        cloud_arr["y"],
        cloud_arr["z"],
    )).astype(np.float32)


def project_points_to_image(
    points_lidar: np.ndarray,
    transform_matrix: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    img_width: int,
    img_height: int,
) -> (np.int32):
    """
    Project lidar points into image pixels.

    Returns:
        pixels: (N, 2) integer pixel coordinates
        visible_lidar_points: (N, 3) lidar-frame points that project into the image
        visible_camera_points: (N, 3) camera-frame points that project into the image
        distances: (N,) Euclidean distances in the camera frame
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)

    ones = np.ones((points_lidar.shape[0], 1), dtype=np.float64)
    points_lidar_h = np.hstack((points_lidar, ones))

    points_camera_h = (transform_matrix @ points_lidar_h.T).T
    points_camera = points_camera_h[:, :3]

    in_front = points_camera[:, 2] > 0.0
    points_camera = points_camera[in_front]
    visible_lidar_points = points_lidar[in_front]

    if points_camera.shape[0] == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_pixels = np.empty((0, 2), dtype=np.int32)
        empty_distances = np.empty((0,), dtype=np.float32)
        return empty_pixels, empty_points, empty_points, empty_distances

    fov_x = 2 * np.arctan2(img_width, 2 * camera_matrix[0, 0]) * 180 / np.pi + 10
    fov_y = 2 * np.arctan2(img_height, 2 * camera_matrix[1, 1]) * 180 / np.pi + 10

    xangle = np.arctan2(points_camera[:, 0], points_camera[:, 2]) * 180 / np.pi
    yangle = np.arctan2(points_camera[:, 1], points_camera[:, 2]) * 180 / np.pi
    in_fov = (
        (xangle > -fov_x / 2) & (xangle < fov_x / 2) &
        (yangle > -fov_y / 2) & (yangle < fov_y / 2)
    )

    points_camera = points_camera[in_fov]
    visible_lidar_points = visible_lidar_points[in_fov]

    if points_camera.shape[0] == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_pixels = np.empty((0, 2), dtype=np.int32)
        empty_distances = np.empty((0,), dtype=np.float32)
        return empty_pixels, empty_points, empty_points, empty_distances

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        points_camera,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    image_points = image_points.reshape(-1, 2)

    in_image = (
        (image_points[:, 0] >= 0) &
        (image_points[:, 0] < img_width) &
        (image_points[:, 1] >= 0) &
        (image_points[:, 1] < img_height)
    )

    pixels = image_points[in_image]
    visible_lidar_points = visible_lidar_points[in_image]
    visible_camera_points = points_camera[in_image]
    distances = np.linalg.norm(visible_camera_points, axis=1)

    return (
        pixels.astype(np.int32),
        visible_lidar_points.astype(np.float32),
        visible_camera_points.astype(np.float32),
        distances.astype(np.float32),
    )


def compute_semantic_cost_image(
    semantic_costs: np.ndarray,
    cost_mode: str,
    probabilities: np.ndarray,
    probability_classes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a semantic cost image from class probabilities and a cost mapping.
        The cost can be computed in different modes:
        - "argmax_cost": Use the cost of the most probable class.
        - "expected_cost": Compute the expected cost based on class probabilities.

        Args:
            semantic_costs: Array of shape (num_classes,) mapping class IDs to costs.
            cost_mode: One of "argmax_cost" or "expected_cost"
            probabilities: Array of shape (H, W, num_classes) with class probabilities for each pixel, or shape (H, W) if probability_classes is provided.
            probability_classes: Optional array of shape (H, W) with class IDs corresponding to the probabilities. Required if probabilities has shape (H, W).

        Returns:
            A 2D array of shape (H, W) with semantic costs in the range [0.0, 1.0].
    """
    if cost_mode not in SUPPORTED_COST_MODES:
        raise ValueError(f"Unsupported cost mode: {cost_mode}")

    if probabilities is None:
        raise ValueError("Probabilities are required for semantic cost computation.")

    if probability_classes is not None:
        return _compute_sparse_semantic_cost_image(
            semantic_costs,
            cost_mode,
            probabilities,
            probability_classes,
        )

    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 0.0, None)

    probs_sum = probabilities.sum(axis=-1, keepdims=True)
    probs_sum[probs_sum == 0.0] = 1.0
    probabilities = probabilities / probs_sum

    if cost_mode == "argmax_cost":
        argmax_labels = np.argmax(probabilities, axis=-1)
        return semantic_costs[argmax_labels]

    expected_cost = np.tensordot(probabilities, semantic_costs, axes=([-1], [0]))
    if cost_mode == "expected_cost":
        return expected_cost.astype(np.float32)


def _compute_sparse_semantic_cost_image(
    semantic_costs: np.ndarray,
    cost_mode: str,
    probabilities: np.ndarray,
    probability_classes: np.ndarray,
) -> np.ndarray:
    """
    Compute semantic cost when only a single class ID with probability is available per pixel.

        This is used when the semantic segmentation model outputs a single class ID per pixel.
        The class probabilities are expected to be unnormalized and may contain invalid class IDs, which are handled gracefully.

        The cost is computed as follows:
        - If cost_mode is "argmax_cost", the cost is the cost of the most probable class.
        - If cost_mode is "expected_cost", the cost is the expected cost based on the class probabilities.

    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probability_classes = np.asarray(probability_classes, dtype=np.int32)

    valid = (
        (probability_classes != INVALID_CLASS_ID) &
        (probability_classes >= 0) &
        (probability_classes < semantic_costs.shape[0])
    )
    probabilities = np.where(valid, np.clip(probabilities, 0.0, 1.0), 0.0)

    class_costs = np.zeros_like(probabilities, dtype=np.float32)
    if np.any(valid):
        class_costs[valid] = semantic_costs[probability_classes[valid]]

    probability_sum = probabilities.sum(axis=-1, keepdims=True)
    probability_sum[probability_sum <= 0.0] = 1.0
    normalized_probabilities = probabilities / probability_sum

    if cost_mode == "argmax_cost":
        best_index = np.argmax(normalized_probabilities, axis=-1)
        best_classes = np.take_along_axis(
            probability_classes,
            best_index[..., None],
            axis=-1,
        )[..., 0]
        return semantic_costs[np.clip(best_classes, 0, semantic_costs.shape[0] - 1)]

    expected_cost = np.sum(normalized_probabilities * class_costs, axis=-1)
    if cost_mode == "expected_cost":
        return expected_cost.astype(np.float32)


def depth_filter_projected_points(
    pixel_points: np.ndarray,
    lidar_points: np.ndarray,
    camera_points: np.ndarray,
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pixel that multiple LiDAR points project onto, keep only the
    closest one (Z-buffer pass).  This prevents points behind an occluder
    (e.g. a tree trunk) from sampling the occluder's semantic label and
    writing a false lethal cost at their actual ground position.
    """
    if pixel_points.shape[0] == 0:
        return pixel_points, lidar_points, camera_points, distances

    # Build a scalar key per pixel so we can group by (u, v) cheaply.
    max_u = int(pixel_points[:, 0].max()) + 1
    keys = pixel_points[:, 1] * max_u + pixel_points[:, 0]

    # For each unique pixel keep the index of the minimum distance.
    order = np.argsort(distances)
    sorted_keys = keys[order]
    _, first = np.unique(sorted_keys, return_index=True)
    keep = order[first]

    return (
        pixel_points[keep],
        lidar_points[keep],
        camera_points[keep],
        distances[keep],
    )


def rasterize_point_costs_to_costmap(
    points_lidar: np.ndarray,
    point_costs: np.ndarray,
    resolution: float,
    width_cells: int,
    height_cells: int,
    origin_x: float,
    origin_y: float,
    z_min: float,
    z_max: float,
    point_footprint_cells: int = 1,
) -> np.ndarray:
    """
    Rasterize per-point semantic costs into a 2D costmap grid.

    The lidar frame is assumed to use x forward and y left/right. Costs are
    conservatively aggregated with a per-cell max.

    point_footprint_cells: radius in cells to dilate each point (1 = single cell,
        2 = 3×3 disk, 3 = 5×5 disk, etc.). Useful to fill gaps in sparse costmaps.
    """
    costmap = np.zeros((height_cells, width_cells), dtype=np.uint8)

    if points_lidar.shape[0] == 0:
        return costmap

    points_lidar = np.asarray(points_lidar, dtype=np.float32)
    point_costs = np.asarray(point_costs, dtype=np.float32)

    in_height = (points_lidar[:, 2] >= z_min) & (points_lidar[:, 2] <= z_max)
    points_lidar = points_lidar[in_height]
    point_costs = point_costs[in_height]

    if points_lidar.shape[0] == 0:
        return costmap

    x = points_lidar[:, 0]
    y = points_lidar[:, 1]

    grid_x = np.floor((x - origin_x) / resolution).astype(np.int32)
    grid_y = np.floor((y - origin_y) / resolution).astype(np.int32)

    in_bounds = (
        (grid_x >= 0) & (grid_x < width_cells) &
        (grid_y >= 0) & (grid_y < height_cells)
    )

    grid_x = grid_x[in_bounds]
    grid_y = grid_y[in_bounds]
    point_costs = point_costs[in_bounds]

    if grid_x.size == 0:
        return costmap

    occupancy_costs = np.clip(np.round(point_costs * COST_LETHAL), COST_FREE, COST_LETHAL).astype(np.uint8)

    r = max(0, int(point_footprint_cells) - 1)
    if r == 0:
        np.maximum.at(costmap, (grid_y, grid_x), occupancy_costs)
    else:
        # Dilate each point into a disk of radius r cells
        offsets = [(dy, dx)
                   for dy in range(-r, r + 1)
                   for dx in range(-r, r + 1)
                   if dy * dy + dx * dx <= r * r]
        for dy, dx in offsets:
            ny = grid_y + dy
            nx = grid_x + dx
            valid = (nx >= 0) & (nx < width_cells) & (ny >= 0) & (ny < height_cells)
            np.maximum.at(costmap, (ny[valid], nx[valid]), occupancy_costs[valid])

    return costmap
