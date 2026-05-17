"""
Pure-Python LiDAR traversability cost computation — no ROS2 dependencies.

Entry point for standalone testing:
    python lidar_cost_processor.py [scan.bin] [--image frame.jpg] [--show] [--save out.npy]
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree

# ── tuneable constants ────────────────────────────────────────────────────────
S_CRIT = 20       # critical slope [degrees]
R_CRIT = 0.2     # critical roughness [m] — RMS off-plane deviation
SLOPE_WEIGHT     = 0.25
ROUGHNESS_WEIGHT = 1.0 - SLOPE_WEIGHT

NEIGHBOR_RADIUS    = 1.0   # [m] radius for slope / roughness neighbourhood
NEIGHBOR_K         = 32    # k nearest neighbours to consider

# Costmap grid defaults (mirrors semantic_fusion_utils — kept local so this
# module stays standalone / no ROS dependency)
COSTMAP_RESOLUTION = 0.2   # [m/cell]
COSTMAP_WIDTH_M    = 40.0  # [m]
COSTMAP_HEIGHT_M   = 20.0  # [m]
COSTMAP_ORIGIN_X   = -20.0 # [m]
COSTMAP_ORIGIN_Y   = -10.0 # [m]
MIN_POINT_Z        = -2.0  # [m]
MAX_POINT_Z        =  2.0  # [m]
# ─────────────────────────────────────────────────────────────────────────────


def load_bin(path: str) -> np.ndarray:
    """Load a Ouster .bin file → (N, 3) float32 xyz."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # ignore reflectance


def voxel_downsample(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Reduce point count by keeping one representative point per voxel cell.
    Uses the centroid of each occupied voxel — preserves local geometry for
    slope/roughness while cutting KDTree cost by ~10-20x on dense LiDAR.
    """
    if voxel_size <= 0 or len(points_xyz) == 0:
        return points_xyz
    keys = np.floor(points_xyz / voxel_size).astype(np.int32)
    # encode (ix, iy, iz) → single int for fast unique
    lo = keys.min(axis=0)
    dims = (keys - lo).max(axis=0) + 1
    flat = (keys[:, 0] - lo[0]) * (dims[1] * dims[2]) + \
           (keys[:, 1] - lo[1]) * dims[2] + \
           (keys[:, 2] - lo[2])
    order = np.argsort(flat)
    flat_sorted = flat[order]
    pts_sorted  = points_xyz[order]
    split = np.where(np.diff(flat_sorted))[0] + 1
    groups = np.split(pts_sorted, split)
    return np.array([g.mean(axis=0) for g in groups], dtype=np.float32)


def compute_slope_and_roughness(
    points_xyz: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute slope and roughness for every point using all neighbours within radius.

    Slope: sin(tilt of best-fit plane normal away from vertical) via PCA.
    Roughness: RMS deviation of points from the best-fit plane, in metres.

    Vectorised: neighbourhoods are padded to a fixed width (capped at MAX_NB)
    so all PCA runs as a single batched eigh call — no Python loop.
    """
    MAX_NB = 32  # cap neighbours per point; enough for a stable plane fit

    tree = cKDTree(points_xyz[:, :2])
    # query up to MAX_NB neighbours (includes self); returns fixed-shape arrays
    dists, idxs = tree.query(points_xyz[:, :2], k=min(MAX_NB, len(points_xyz)), workers=-1)

    N  = points_xyz.shape[0]
    k  = idxs.shape[1]

    in_radius = dists <= radius                        # (N, k)
    counts    = in_radius.sum(axis=1)                  # (N,)
    enough    = counts >= 3

    nb_pts = points_xyz[idxs]                          # (N, k, 3)
    mask3  = in_radius[:, :, None]                     # (N, k, 1)

    safe_counts = np.maximum(counts, 1)[:, None]       # (N, 1)
    centroid    = (nb_pts * mask3).sum(axis=1) / safe_counts   # (N, 3)
    centred     = (nb_pts - centroid[:, None, :]) * mask3      # (N, k, 3)

    cov = np.einsum("nki,nkj->nij", centred, centred) / safe_counts[:, :, None]  # (N,3,3)

    eigvals, eigvecs = np.linalg.eigh(cov)             # (N,3), (N,3,3) ascending

    normals  = eigvecs[:, :, 0]                         # (N, 3) plane normals (smallest eigenvector)
    # flip normals to point upward (z > 0) so arrows are consistent
    normals  = np.where(normals[:, 2:3] < 0, -normals, normals)

    nz       = np.abs(normals[:, 2])                    # z-component of (upward) normal
    slopes   = np.where(enough, np.sqrt(np.clip(1.0 - nz**2, 0.0, 1.0)), 0.0)

    # Roughness = std deviation of points from the best-fit plane, in metres.
    # The smallest eigenvalue of the covariance is the variance perpendicular
    # to the fitted plane (because the smallest eigenvector is the plane normal),
    # so sqrt(eigvals[0]) is the RMS off-plane deviation.
    #   ~0.00 m → perfectly flat surface
    #   ~0.05 m → light bumpiness (gravel, short grass)
    #   ~0.20 m+ → vegetation, rubble, vertical structure
    roughness = np.where(enough, np.sqrt(np.clip(eigvals[:, 0], 0.0, None)), 0.0)

    return slopes.astype(np.float32), roughness.astype(np.float32), normals.astype(np.float32)


def compute_traversability_costs(
    points_xyz: np.ndarray,
    radius: float = NEIGHBOR_RADIUS,
    s_crit: float = S_CRIT,
    r_crit: float = R_CRIT,
    w1: float = SLOPE_WEIGHT,
    w2: float = ROUGHNESS_WEIGHT,
    R_gravity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Combine slope and roughness into a traversability cost in [0, 1].

    R_gravity:
        Optional (3, 3) rotation from sensor frame → gravity-aligned frame
        (z = up). Without this, slope/roughness are in sensor frame and will
        be wrong when the robot is on a slope.
    """
    pts = points_xyz @ R_gravity.T if R_gravity is not None else points_xyz

    slopes, roughness, _ = compute_slope_and_roughness(pts, radius)

    slope_sin_crit = np.sin(np.radians(s_crit))
    slope_cost     = w1 * (slopes    / slope_sin_crit)
    roughness_cost = w2 * (roughness / r_crit)
    return np.clip(slope_cost + roughness_cost, 0.0, 1.0).astype(np.float32)


def rasterize(
    points_xyz: np.ndarray,
    costs: np.ndarray,
    resolution: float = COSTMAP_RESOLUTION,
    width_m: float = COSTMAP_WIDTH_M,
    height_m: float = COSTMAP_HEIGHT_M,
    origin_x: float = COSTMAP_ORIGIN_X,
    origin_y: float = COSTMAP_ORIGIN_Y,
    z_min: float = MIN_POINT_Z,
    z_max: float = MAX_POINT_Z,
) -> np.ndarray:
    """Project (points_xyz, costs) onto a 2-D uint8 costmap [0-100] (OccupancyGrid scale)."""
    width_cells  = int(np.ceil(width_m  / resolution))
    height_cells = int(np.ceil(height_m / resolution))
    costmap = np.zeros((height_cells, width_cells), dtype=np.uint8)
    # filter to points within z bounds (ignore points too high/low to be relevant for navigation)
    mask = (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)
    pts, c = points_xyz[mask], costs[mask]
    if pts.shape[0] == 0:
        return costmap
    # convert (x, y) → (gx, gy) grid coordinates; filter to points that fall within the grid bounds
    gx = np.floor((pts[:, 0] - origin_x) / resolution).astype(np.int32)
    gy = np.floor((pts[:, 1] - origin_y) / resolution).astype(np.int32)
    in_bounds = (gx >= 0) & (gx < width_cells) & (gy >= 0) & (gy < height_cells)
    gx, gy, c = gx[in_bounds], gy[in_bounds], c[in_bounds]

    # If multiple points fall in the same cell, take the max cost (worst-case) — achieved by sorting by cost and painting low→high.
    order = np.argsort(c)   # paint low cost first so high cost wins
    costmap[gy[order], gx[order]] = np.clip(np.round(c[order] * 100), 0, 100).astype(np.uint8)
    return costmap


def fill_gaps(costmap: np.ndarray) -> np.ndarray:
    """
    Fill empty cells (value 0) in the costmap using nearest-neighbour
    propagation from known cells.

    Uses scipy distance transform: each unknown cell inherits the value of
    its nearest known cell — O(N) and handles arbitrarily shaped gaps.
    Returns a new uint8 array of the same shape.
    """
    from scipy.ndimage import distance_transform_edt

    known = costmap > 0                          # mask of cells with a LiDAR hit
    if known.all():
        return costmap.copy()

    # For each unknown cell, find the index of the nearest known cell
    _, nearest_idx = distance_transform_edt(~known, return_distances=True, return_indices=True)
    filled = costmap[nearest_idx[0], nearest_idx[1]]
    filled[~known] = costmap[nearest_idx[0][~known], nearest_idx[1][~known]]
    return filled.astype(np.uint8)

# ── calibration helpers ───────────────────────────────────────────────────────

def load_camera_info(path: str):
    """Load [fx, fy, cx, cy] from a plain-text camera_info file → 3x3 K matrix."""
    fx, fy, cx, cy = np.loadtxt(path)
    K = np.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
    return K


def load_extrinsics(path: str, key: str = "os1_cloud_node-pylon_camera_node") -> np.ndarray:
    """Load lidar→camera extrinsic from a ROS-style YAML → 4x4 matrix."""
    import yaml
    from scipy.spatial.transform import Rotation
    with open(path) as f:
        data = yaml.safe_load(f)
    d = data[key]
    q = np.array([d["q"]["x"], d["q"]["y"], d["q"]["z"], d["q"]["w"]])
    t = np.array([d["t"]["x"], d["t"]["y"], d["t"]["z"]])
    RT = np.eye(4)
    RT[:3, :3] = Rotation.from_quat(q).as_matrix()
    RT[:3,  3] = t
    return np.linalg.inv(RT)   # invert: world→cam becomes lidar→cam


def overlay_costs_on_image(
    image_path: str,
    points_xyz: np.ndarray,
    costs: np.ndarray,
    camera_info_path: str,
    transforms_path: str,
    dist_coeffs: np.ndarray = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0.0]),  # logitech camera calibration
    dot_radius: int = 3,
) -> np.ndarray:
    """
    Project LiDAR points coloured by traversability cost onto the camera image.

    Returns an RGB image (H, W, 3) uint8.
    """
    import cv2
    import matplotlib.cm as cm

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    K  = load_camera_info(camera_info_path)
    RT = load_extrinsics(transforms_path)

    # ── filter to camera FOV ──────────────────────────────────────────────────
    pts_h = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])  # (N, 4)
    pts_cam = (RT @ pts_h.T).T                                        # (N, 4) in cam frame

    # keep only points in front of the camera
    in_front = pts_cam[:, 2] > 0
    pts_cam  = pts_cam[in_front]
    pts_lidar_fov = points_xyz[in_front]
    costs_fov     = costs[in_front]

    fov_x = 2 * np.arctan2(img_w, 2 * K[0, 0]) * 180 / np.pi + 10
    fov_y = 2 * np.arctan2(img_h, 2 * K[1, 1]) * 180 / np.pi + 10
    xa = np.arctan2(pts_cam[:, 0], pts_cam[:, 2]) * 180 / np.pi
    ya = np.arctan2(pts_cam[:, 1], pts_cam[:, 2]) * 180 / np.pi
    in_fov = (np.abs(xa) < fov_x / 2) & (np.abs(ya) < fov_y / 2)

    pts_lidar_fov = pts_lidar_fov[in_fov]
    costs_fov     = costs_fov[in_fov]

    if pts_lidar_fov.shape[0] == 0:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── project ───────────────────────────────────────────────────────────────
    R_vec, _ = cv2.Rodrigues(RT[:3, :3])
    T_vec    = RT[:3, 3].reshape(3, 1)
    imgpts, _ = cv2.projectPoints(
        pts_lidar_fov.astype(np.float64), R_vec, T_vec,
        K.astype(np.float64), dist_coeffs.astype(np.float64),
    )
    uv = np.squeeze(imgpts, 1)   # (M, 2)

    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )
    uv        = uv[in_bounds]
    costs_fov = costs_fov[in_bounds]

    # ── draw coloured dots (RdYlGn_r: green=safe, red=obstacle) ─────────────
    cmap   = cm.get_cmap("RdYlGn_r")
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    for (u, v), cost in zip(uv, costs_fov):
        r, g, b, _ = cmap(float(cost))
        color = (int(r * 255), int(g * 255), int(b * 255))
        cv2.circle(result, (int(u), int(v)), dot_radius, color, -1)

    return result


# ── full pipeline ─────────────────────────────────────────────────────────────

def process_bin(
    path: str,
    radius: float = NEIGHBOR_RADIUS,
    z_min: float = MIN_POINT_Z,
    z_max: float = MAX_POINT_Z,
):
    """Full pipeline for a single .bin file.  Returns (points_xyz, costs, costmap)."""
    import time
    t0 = time.perf_counter()

    points_xyz = load_bin(path)
    points_xyz = points_xyz[(points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)]
    print(f"  load+filter : {time.perf_counter()-t0:.2f}s  ({len(points_xyz)} pts)")

    t1 = time.perf_counter()
    costs = compute_traversability_costs(points_xyz, radius=radius)
    print(f"  costs       : {time.perf_counter()-t1:.2f}s")

    t2 = time.perf_counter()
    costmap = rasterize(points_xyz, costs)
    print(f"  rasterize   : {time.perf_counter()-t2:.2f}s")

    return points_xyz, costs, costmap