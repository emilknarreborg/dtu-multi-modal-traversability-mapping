# Command Reference

## Setup

### Enter the container (Jetson)
```bash
cd ~/bachelor_thesis/docker
docker compose -f docker-compose.jetson.humble.yml exec ros2_perception_humble bash
```

### Build and source (inside container)
```bash
cd /root/ros2_ws
colcon build --packages-select traversability_pkg perception_pkg bringup_pkg platform_description
source install/setup.bash
```

---

## Topic map

```
/image_raw                             ← camera (bag or live usb_cam)
/camera_info                           ← camera intrinsics
/livox/lidar                           ← Livox Mid-360 point cloud
/livox/imu                             ← Livox IMU

/segmentation/probabilities            ← semantic_segmentation_node output (32FCk)
/segmentation/probability_classes      ← semantic_segmentation_node output (8UCk)

/lidar_costmap                         ← lidar_cost_node output
/semantic_costmap                      ← semantic_cost_node output
/fused_costmap                         ← fused_costmap_node output
/semantic_cost_image                   ← per-pixel cost image (optional, publish_cost_image:=true)
```

---

## 1 — Record a raw bag (live hardware)

```bash
ros2 launch bringup_pkg rosbag_record_lidar_image.launch.py
```

Records `/image_raw`, `/camera_info`, `/livox/lidar`, `/livox/imu` to
`/root/data/bags/dtu_lidar_image_<timestamp>`.

---

## 2 — Produce a segmented bag from a raw bag

**Option A — Launch file** (plays at 0.1× speed so GPU keeps up):
```bash
ros2 launch bringup_pkg semseg_record_bag.launch.py \
    bag_path:=/root/data/bags/dtu_lidar_image_20260416_093308_10hz \
    bag_output:=/root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz
```

**Option B — Offline script** (faster, no ROS middleware, runs inside container):
```bash
python3 /root/scripts/offline_semseg.py \
    /root/data/bags/dtu_lidar_image_20260416_093308_10hz \
    /root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz
```

---

## 3 — Semantic costmap only (pre-segmented bag)

For DTU bags (`*_10hz` segmented bags):
```bash
ros2 launch bringup_pkg semantic_cost_pipeline.launch.py \
    bag_path:=/root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz
```

Key overrides:

| Argument | Default | Description |
|---|---|---|
| `bag_path` | (see above) | Pre-segmented bag directory |
| `playback_rate` | `1.0` | Bag speed multiplier |
| `cost_mode` | `penalized_expected_cost` | `argmax_cost` / `expected_cost` / `penalized_expected_cost` |
| `uncertainty_penalty` | `0.25` | Entropy penalty weight |
| `publish_cost_image` | `true` | Publish `/semantic_cost_image` for RViz |

---

## 4 — Fused costmap (LiDAR + semantic, pre-segmented bag)

**DTU / Livox Mid-360 bag (default):**
```bash
ros2 launch bringup_pkg fused_costmap_pipeline.launch.py \
    bag_path:=/root/data/bags/segmented/segmented_dtu_lidar_image_20260416_093308_10hz
```

**RELLIS / Ouster OS1 bag:**
```bash
ros2 launch bringup_pkg fused_costmap_pipeline.launch.py \
    sensor_config:=ouster_os1 \
    cam_topic_prefix:=/pylon_camera_node \
    bag_path:=/root/data/bags/segmented/segmented_test_20260326_125630
```

Key overrides:

| Argument | Default | Description |
|---|---|---|
| `sensor_config` | `livox_mid360` | `livox_mid360` or `ouster_os1` — sets lidar topic and TF frame |
| `cam_topic_prefix` | `""` | Topic prefix for camera topics in the bag (e.g. `/pylon_camera_node` for RELLIS) |
| `bag_path` | (see above) | Pre-segmented bag directory |
| `playback_rate` | `1.0` | Bag speed multiplier |
| `play_bag` | `true` | Set `false` for live hardware |
| `use_rviz` | `true` | Open RViz |
| `appearance_alpha` | `0.5` | Semantic weight in fusion |
| `geometric_alpha` | `0.5` | LiDAR weight in fusion |
| `cost_mode` | `penalized_expected_cost` | Cost mode |
| `publish_cost_image` | `false` | Enable `/semantic_cost_image` topic |

> **Note:** `/tf` is excluded from bag playback in all configs — `robot_state_publisher` re-publishes joint TFs fresh to avoid `TF_OLD_DATA` warnings from old bag timestamps.

---

## 5 — Full pipeline with live inference (raw bag, no pre-segmented topics)

```bash
ros2 launch bringup_pkg semantic_full_pipeline.launch.py \
    bag_path:=/root/data/bags/dtu_lidar_image_20260416_093308_10hz
```

Runs `semantic_segmentation_node` + `semantic_cost_node` in one launch.
Use `playback_rate:=0.1` if inference can't keep up.

---

## 6 — LiDAR costmap only

```bash
ros2 launch bringup_pkg lidar_cost_pipeline.launch.py \
    bag_path:=/root/data/bags/dtu_lidar_image_20260416_093308_10hz
```

Key overrides:

| Argument | Default | Description |
|---|---|---|
| `sensor_config` | `livox_mid360` | `livox_mid360` or `ouster_os1` |
| `play_bag` | `true` | Set `false` for live |
| `fill_nn` | `false` | NN-fill for sparse clouds |

---

## 7 — Live hardware (no bag)

```bash
ros2 launch bringup_pkg fused_costmap_pipeline.launch.py play_bag:=false
```

Expects live topics from the Livox driver and camera node.

---

## Camera–LiDAR TF calibration

The extrinsic is published as a static TF by `semantic_cost_node.launch.py`.
To update after re-calibration, edit the `arguments` of the
`lidar_camera_tf` node in [semantic_cost_node.launch.py](launch/semantic_cost_node.launch.py):

```
# format: x y z yaw pitch roll parent_frame child_frame
"0.04", "0", "-0.22", "1.361357", "0", "1.500983", "livox_frame", "pylon_camera"
```

Quaternion equivalent (for reference):
```
w=0.56836858  x=0.46025580  y=0.42919547  z=0.53001227
```

---

## Tuning parameters

### LiDAR costmap (`lidar_cost_node`)

Edit [config/sensors/livox_mid360.yaml](config/sensors/livox_mid360.yaml) and restart the node. All values are in SI units.

#### Slope & roughness

| Parameter | Default (Livox) | What it does |
|---|---|---|
| `neighbor_radius` | `0.6` m | Radius of the local neighbourhood used to compute slope and roughness. Larger → smoother, less sensitive to noise; smaller → sharper detection of small obstacles. |
| `w1` | `0.2` | Weight of slope in the combined cost (`w1 * slope_cost + w2 * roughness_cost`). |
| `w2` | `0.8` | Weight of roughness. `w1 + w2` should equal 1. |

Slope and roughness are normalised by their respective critical values before weighting:

| Constant | Value | Meaning |
|---|---|---|
| `S_CRIT` | `20°` | Slope angle that maps to cost = 1. Steeper → unsafe. Defined in `lidar_cost_processor.py`. |
| `R_CRIT` | `2.0` | Roughness ratio that maps to cost = 1. Defined in `lidar_cost_processor.py`. |

#### Point cloud filtering

| Parameter | Default (Livox) | What it does |
|---|---|---|
| `min_point_z` / `max_point_z` | `-2.0` / `2.0` m | Z-filter in the costmap frame. Excludes ground reflections below and irrelevant overhead points above. Tighten to reduce noise on flat ground. |
| `forward_fov_deg` | `200°` | Azimuth FOV kept around the forward X axis. `360` = full scan. Reduce to eliminate backward-facing beams that hit the ground noisily. |
| `voxel_size` | `0.1` m | Pre-KDTree voxel downsampling. `0` = off. Larger → faster but loses fine structure. |
| `accumulate_frames` | `1` | Number of consecutive LiDAR scans merged before computing costs. Increase (e.g. 3–5) for the Mid-360's non-repetitive pattern to get denser coverage. |

#### Distance uncertainty

| Parameter | Default | What it does |
|---|---|---|
| `DISTANCE_PENALTY_W` | `0.1` | Blends cost upward for distant points (`cost += w * range / MAX_RELIABLE_DIST`). `0` = off. Defined in `lidar_cost_processor.py`. |
| `MAX_RELIABLE_DIST` | `10.0` m | Range beyond which points are considered uncertain. Defined in `lidar_cost_processor.py`. |
| `SPARSITY_PENALTY` | `1.0` | Cost assigned to points with fewer than 2 neighbours within `neighbor_radius`. `1.0` = treat isolated points as lethal. Defined in `lidar_cost_processor.py`. |

---

### Semantic costmap (`semantic_cost_node`)

#### Cost mode

Controlled by the `cost_mode` launch argument:

| Mode | Behaviour |
|---|---|
| `argmax_cost` | Cost = cost of the single most-probable class per pixel. Fast, ignores uncertainty. |
| `expected_cost` | Cost = probability-weighted average of all top-K class costs. Smoother than argmax. |
| `penalized_expected_cost` | `expected_cost + uncertainty_penalty * normalised_entropy`. Raises cost where the model is uncertain. **Recommended default.** |

#### Uncertainty penalty

| Parameter | Default | What it does |
|---|---|---|
| `uncertainty_penalty` | `0.25` | Weight on the normalised Shannon entropy term in `penalized_expected_cost`. `0` = pure expected cost; `1` = uncertain pixels always reach max cost. |

#### Per-class costs

Defined in `DEFAULT_SEMANTIC_COSTS` in [semantic_fusion_utils.py](../../traversability_pkg/traversability_pkg/semantic_fusion_utils.py). Values are in `[0, 1]` — `0` = fully traversable, `1` = lethal.

| Class | Cost | Class | Cost |
|---|---|---|---|
| dirt | 0.10 | asphalt | 0.05 |
| grass | 0.10 | concrete | 0.20 |
| mud | 0.40 | puddle | 0.70 |
| bush | 0.60 | water | 0.80 |
| rubble | 0.80 | log | 0.70 |
| tree | 1.00 | pole | 1.00 |
| building | 1.00 | fence | 1.00 |
| barrier | 1.00 | vehicle | 1.00 |
| person | 1.00 | sky | 1.00 |
| object | 0.50 | | |

To change a class cost, edit `DEFAULT_SEMANTIC_COSTS` in `semantic_fusion_utils.py` and rebuild `traversability_pkg`.

#### Segmentation input filtering

Controlled via the `semantic_segmentation_node` parameters:

| Parameter | Default | What it does |
|---|---|---|
| `probability_threshold` | `0.05` | Class probabilities below this are zeroed out before cost computation. Raise to suppress very uncertain predictions. |
| `max_probability_classes` | `4` | Top-K classes retained per pixel. Reducing to `1` is equivalent to argmax and halves memory. |

---

### Fusion (`fused_costmap_node`)

| Parameter | Default | What it does |
|---|---|---|
| `appearance_alpha` | `0.5` | Weight of the semantic costmap in `fused = geometric_alpha * lidar + appearance_alpha * semantic`. |
| `geometric_alpha` | `0.5` | Weight of the LiDAR costmap. The two weights are independent — they do not need to sum to 1, but values that sum to >1 will saturate the output. |

---

## Monitor and debug

```bash
# Check topic rates
ros2 topic hz /fused_costmap
ros2 topic hz /semantic_costmap
ros2 topic hz /lidar_costmap

# Inspect costmap metadata (no large array dump)
ros2 topic echo /fused_costmap --no-arr

# Check TF tree
ros2 run tf2_tools view_frames

# GPU check
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## RViz standalone

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix bringup_pkg)/share/bringup_pkg/rviz/bachelor_thesis_rviz.rviz
```
