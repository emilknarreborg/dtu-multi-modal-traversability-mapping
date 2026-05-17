"""ROS2 node — Fused costmap from lidar and semantic segmentation."""

import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from scipy.ndimage import distance_transform_edt
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3

from .semantic_fusion_utils import COST_FREE, COST_LETHAL, OccupancyGridEma

DEFAULT_APPEARANCE_ALPHA = 0.5
DEFAULT_GEOMETRIC_ALPHA  = 0.5


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class FusedCostMap(Node):
    def __init__(self):
        super().__init__("fused_costmap_node")

        self.declare_parameter("lidar_costmap_topic",    "/lidar_costmap")
        self.declare_parameter("semantic_costmap_topic", "/semantic_costmap")
        self.declare_parameter("output_costmap_topic",   "/fused_costmap")
        self.declare_parameter("appearance_alpha",       DEFAULT_APPEARANCE_ALPHA)
        self.declare_parameter("geometric_alpha",        DEFAULT_GEOMETRIC_ALPHA)
        self.declare_parameter("lidar_buffer_size",      10)
        self.declare_parameter("max_stamp_diff_sec",     0.15)
        # Rolling accumulation: keep a running max over this many recent lidar
        # costmaps before fusing. Fills sparse single-scan grids without the
        # edge-bleeding artefacts of nearest-neighbour gap-fill.
        # 1 = no accumulation (use only the matched frame).
        self.declare_parameter("accumulate_scans",       10)
        self.declare_parameter("fill_nn",                False)
        # Voxel marker: cells with cost >= this threshold (0–100) are visualised.
        # Lethal (100) voxels are 1 m tall; others scale linearly down to voxel_min_height.
        self.declare_parameter("voxel_cost_threshold",   1)
        self.declare_parameter("voxel_max_height",       1.0)
        self.declare_parameter("voxel_min_height",       0.05)
        self.declare_parameter("publish_costmap",        True)
        self.declare_parameter("publish_voxels",         False)
        self.declare_parameter("temporal_smooth",        False)
        self.declare_parameter("temporal_alpha",         0.4)

        p = self.get_parameter
        self.lidar_costmap_topic    = p("lidar_costmap_topic").value
        self.semantic_costmap_topic = p("semantic_costmap_topic").value
        self.output_costmap_topic   = p("output_costmap_topic").value
        self.appearance_alpha       = float(p("appearance_alpha").value)
        self.geometric_alpha        = float(p("geometric_alpha").value)
        self.max_stamp_diff         = float(p("max_stamp_diff_sec").value)
        self._accumulate_scans      = max(1, int(p("accumulate_scans").value))
        self._fill_nn               = bool(p("fill_nn").value)
        self._voxel_threshold       = int(p("voxel_cost_threshold").value)
        self._voxel_max_height      = float(p("voxel_max_height").value)
        self._voxel_min_height      = float(p("voxel_min_height").value)
        self.publish_costmap       = bool(p("publish_costmap").value)
        self.publish_voxels        = bool(p("publish_voxels").value)
        self._temporal_smooth      = bool(p("temporal_smooth").value)
        self._ema = OccupancyGridEma(float(p("temporal_alpha").value))

        buf_size = int(p("lidar_buffer_size").value)
        self._lidar_buf: collections.deque = collections.deque(maxlen=buf_size)

        sub_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        pub_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        self.create_subscription(OccupancyGrid, self.lidar_costmap_topic,
                                 self._lidar_cb, sub_qos)
        self.create_subscription(OccupancyGrid, self.semantic_costmap_topic,
                                 self._semantic_cb, sub_qos)
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, self.output_costmap_topic, pub_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, self.output_costmap_topic + "/voxels", pub_qos)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _lidar_cb(self, msg: OccupancyGrid) -> None:
        self._lidar_buf.append(msg)

    def _semantic_cb(self, msg: OccupancyGrid) -> None:
        lidar_msg = self._find_matching_lidar(msg)
        if lidar_msg is None:
            self.get_logger().warn(
                "No lidar costmap within %.3f s of semantic stamp %.3f — dropping frame."
                % (self.max_stamp_diff, _stamp_to_sec(msg.header.stamp)),
                throttle_duration_sec=2.0,
            )
            return
        self._fuse_and_publish(lidar_msg, msg)

    # ── synchronisation ───────────────────────────────────────────────────────

    def _find_matching_lidar(self, semantic_msg: OccupancyGrid):
        """Return the buffered lidar costmap whose stamp is closest to the
        semantic costmap's stamp, within max_stamp_diff_sec."""
        if not self._lidar_buf:
            return None

        sem_t = _stamp_to_sec(semantic_msg.header.stamp)
        best  = min(self._lidar_buf,
                    key=lambda m: abs(_stamp_to_sec(m.header.stamp) - sem_t))

        if abs(_stamp_to_sec(best.header.stamp) - sem_t) > self.max_stamp_diff:
            return None
        return best

    # ── fusion ────────────────────────────────────────────────────────────────

    def _resize_nearest(self, src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        src_h, src_w = src.shape
        tgt_h, tgt_w = target_shape
        y_idx = np.linspace(0, src_h - 1, tgt_h).round().astype(int)
        x_idx = np.linspace(0, src_w - 1, tgt_w).round().astype(int)
        return src[np.ix_(y_idx, x_idx)]

    def _accumulated_lidar(self, anchor_msg: OccupancyGrid) -> np.ndarray:
        """Return a lidar grid built from the rolling max of the N most recent
        frames up to and including anchor_msg.  Cells that have never been hit
        stay at 0; cells hit in any recent frame keep the highest observed cost.
        This fills sparse single-scan gaps without bleeding edges."""
        h = anchor_msg.info.height
        w = anchor_msg.info.width
        acc = np.zeros((h, w), dtype=np.float32)

        anchor_t = _stamp_to_sec(anchor_msg.header.stamp)

        # Walk the buffer newest-first, collect up to accumulate_scans frames
        # that are older than or equal to the anchor stamp.
        count = 0
        for msg in reversed(self._lidar_buf):
            if _stamp_to_sec(msg.header.stamp) > anchor_t:
                continue
            if msg.info.height != h or msg.info.width != w:
                continue
            frame = np.array(msg.data, dtype=np.float32).reshape(h, w)
            # rolling max — unknown cells (0) are ignored; a real zero-cost cell
            # is indistinguishable but that only makes the map slightly optimistic.
            acc = np.maximum(acc, frame)
            count += 1
            if count >= self._accumulate_scans:
                break

        return acc

    def _fill_nearest(self, grid: np.ndarray) -> np.ndarray:
        """Fill unknown cells (value 0) by propagating the nearest known cell's value."""
        known = grid != 0
        if known.all():
            return grid
        _, idx = distance_transform_edt(~known, return_indices=True)
        filled = grid.copy()
        filled[~known] = grid[idx[0][~known], idx[1][~known]]
        return filled

    def _fuse_and_publish(self, lidar_msg: OccupancyGrid, semantic_msg: OccupancyGrid):
        lidar_data = self._accumulated_lidar(lidar_msg)

        semantic_data = np.array(semantic_msg.data, dtype=np.float32).reshape(
            semantic_msg.info.height, semantic_msg.info.width
        )
        if lidar_data.shape != semantic_data.shape:
            semantic_data = self._resize_nearest(semantic_data, lidar_data.shape)

        lidar_hit    = lidar_data > 0
        semantic_hit = semantic_data > 0

        blended = self.geometric_alpha * lidar_data + self.appearance_alpha * semantic_data
        fused = np.select(
            [lidar_hit & semantic_hit,   # both available → blend
             lidar_hit,                  # only lidar → lidar as-is
             semantic_hit],              # only semantic → semantic as-is
            [blended, lidar_data, semantic_data],
            default=0,
        )
        fused = np.clip(np.rint(fused), -1, 100).astype(np.int8)

        if self._temporal_smooth:
            fused = self._ema.apply(fused)

        if self._fill_nn:
            fused = self._fill_nearest(fused)

        out = OccupancyGrid()
        out.header = lidar_msg.header
        out.info   = lidar_msg.info
        out.data   = fused.flatten().tolist()
        
        # Publish the fused costmap and voxel markers (if enabled).
        if self.publish_costmap:
            self.costmap_pub.publish(out)
        if self.publish_voxels:
            self._publish_voxels(fused, out)


    def _publish_voxels(self, grid: np.ndarray, costmap_msg: OccupancyGrid) -> None:
        """Publish a MarkerArray of cubes representing the fused costmap in 3D.

        Each occupied cell becomes one cube sitting on the ground plane.
        Height scales linearly with cost (voxel_min_height at threshold,
        voxel_max_height at cost=100 / lethal).  Color goes green → red.
        A single DELETE_ALL marker is prepended so stale voxels are cleared.
        """
        info = costmap_msg.info
        res  = info.resolution
        ox   = info.origin.position.x
        oy   = info.origin.position.y
        oz   = info.origin.position.z

        rows, cols = np.where(grid >= self._voxel_threshold)
        costs = grid[rows, cols].astype(np.float32)  # 0–100

        array = MarkerArray()

        # Clear all previous markers in a single call.
        clear = Marker()
        clear.header    = costmap_msg.header
        clear.ns        = "costmap_voxels"
        clear.action    = Marker.DELETEALL
        array.markers.append(clear)

        if rows.size == 0:
            self.marker_pub.publish(array)
            return

        t = self._voxel_threshold
        height_range = self._voxel_max_height - self._voxel_min_height
        # Normalise cost into [0, 1] over [threshold, 100].
        norm = np.clip((costs - t) / max(COST_LETHAL - t, 1), 0.0, 1.0)
        heights = self._voxel_min_height + norm * height_range

        # Cell centre X/Y in the costmap frame.
        cx = ox + (cols + 0.5) * res
        cy = oy + (rows + 0.5) * res
        # Cube sits on the ground (oz); centre it at half its height.
        cz = oz + heights / 2.0

        # Green (low cost) → red (lethal).
        r = norm
        g = 1.0 - norm

        for i in range(len(rows)):
            m = Marker()
            m.header    = costmap_msg.header
            m.ns        = "costmap_voxels"
            m.id        = int(i)
            m.type      = Marker.CUBE
            m.action    = Marker.ADD
            m.pose.position    = Point(x=float(cx[i]), y=float(cy[i]), z=float(cz[i]))
            m.pose.orientation.w = 1.0
            m.scale     = Vector3(x=res, y=res, z=float(heights[i]))
            m.color     = ColorRGBA(r=float(r[i]), g=float(g[i]), b=0.0, a=0.8)
            array.markers.append(m)

        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = FusedCostMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
