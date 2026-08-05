# Cartographer

Google's real-time 2D and 3D LiDAR SLAM running on ROS Noetic.

- **Repo**: bundled snapshot in `cartographer.tar.xz`
- **Sensors**: 2D laser, 3D LiDAR, optional IMU
- **GPU**: not required for SLAM; used only to render rviz

## Build

```bash
podman build -t slam_zero_to_hero:cartographer .
```

The image extracts `cartographer.tar.xz`, builds Abseil from source, then runs `catkin build` over `cartographer`, `cartographer_ros`, and `cartographer_rviz`. `rviz` is installed; `map_server` is not.

## Verified run — Hilti 2022 `exp14_basement_2.bag` (3D + IMU)

```bash
mkdir -p results
./run_carto.sh                     # offline: preprocess bag -> SLAM -> 3D map + occupancy grid
./run_carto_live.sh                # online: same SLAM with rviz, submaps appearing as it goes
```

Both scripts bind-mount `config/`, `urdf/` and `scripts/` into `slam_zero_to_hero:cartographer`, so nothing needs rebuilding to change a parameter. Neither needs `--net=host` — each starts its own roscore in the container's network namespace.

Last verified: Ryzen 9 7950X, 2026-08-05. Independently re-measured from the artifacts by a second pass, with scripts calibrated against this repo's published FAST-LIO2 reference before being trusted.

| | Cartographer 3D | FAST-LIO2 reference |
|---|---|---|
| Poses | 730 | 737 |
| Path length | **38.378 m** | 37.934 m |
| Start → end | **21.496 m** | 21.350 m |
| z extent | **4.189 m** | 4.154 m |
| Median / max inter-frame step | 0.0551 / 0.1716 m | 0.0540 / 0.1696 m |
| Rigid SE(3) ATE vs FAST-LIO2 | **0.0844 m RMSE** over 728 matched pairs | — |

Result quality: **8 submaps**, 221 loop-closure computations with **26 constraints accepted**, final pose-graph residuals translational mean 0.0274 m / max 0.134 m, rotational mean 0.0032 rad. `min_score` is left at **0.62** — *tighter* than Cartographer's own 0.55 default — so nothing was loosened to make constraints appear. The run is bit-for-bit deterministic: identical `md5` for the trajectory and pbstream across three independent output trees.

Outputs:

| File | Size | Description |
|---|---|---|
| `map.pbstream` | 10,576,369 B | Pose graph + 3D submaps |
| `assets_map3d.ply` / `.pcd` | 341 MB / 256 MB | Dense 3D map, **21,335,730 points** |
| `final_slab.pgm` + `.yaml` | 232,272 B | Occupancy grid, floor-level slab |
| `grid_allz.pgm` + `.yaml` | 283,512 B | Occupancy grid, all heights |
| `carto_tum.txt` | 730 lines | Trajectory, TUM format |

### The old 2D config was the bug — and why

The previous `hilti_3d.lua` (misleadingly named; it ran the **2D** builder) produced a smeared map: occupied cells outnumbered free ones, 71 % unknown, and occupied runs reached 8.30 m. Three causes, all now addressed:

| Before | After |
|---|---|
| 2D trajectory builder on a traverse with **4.19 m of vertical motion** | `MAP_BUILDER.use_trajectory_builder_3d = true` |
| `use_imu_data = false`, so nothing compensated the handheld tilt; the cloud was cropped to a fixed z-slab **in the tilted sensor frame** | IMU integrated directly, `tracking_frame = "imu_sensor_frame"` |
| Every point had de-skew time 0 | a real per-point `time` field (see below) |

Measured improvement in the occupancy grid: **free/occupied ratio 0.859 → 5.153**, longest occupied run **8.30 m → 2.80 m**. Rooms read as rooms — the central hall measures about 10.5 × 12 m off the grid.

Importantly the trajectory is not gaming the sharpness metric by standing still: it travels **1.2 % further** than FAST-LIO2. On the voxel-sharpness arbiter (`scripts/arbiter.py`, lower is sharper, 1.000 = pretending the sensor never moved):

| Window | identity | old 2D config | **Cartographer 3D** | FAST-LIO2 |
|---|---|---|---|---|
| scans 100–129 | 1.000 | ~0.98 | **0.599** | 0.587 |
| scans 300–329 | 1.000 | ~0.77 | **0.513** | 0.485 |
| scans 500–529 | 1.000 | ~0.96 | **0.435** | 0.434 |

### Cartographer could not de-skew this bag at all

This was not in the original diagnosis and it mattered more than any tuning. `cartographer_ros` reads per-point times from a field named literally **`time`**, as `float32` (`PointXYZIT` in `msg_conversion.cc`). Hilti publishes an absolute **`float64 timestamp`** instead, so every point arrived with `time = 0` and each 100 ms sweep was treated as instantaneous — while the operator walked through it.

`scripts/hesai_add_time_field.py` rewrites the bag once, adding a real relative `time` field. The effect:

| | rigid ATE vs FAST-LIO2 | arbiter (100/300/500) |
|---|---|---|
| without `time` | 0.217 m RMSE | 0.520 / 0.613 / 0.603 |
| with `time` | **0.084 m RMSE** | **0.599 / 0.513 / 0.435** |

### 2D + IMU: better, but still wrong

`config/hilti_2d_imu.lua` is the cheap comparison point — same TF and tracking frame, `use_imu_data = true`, still the 2D builder. Tilt compensation removes almost all of the *smearing*, but the *geometry* stays wrong, because no single 2D grid can represent a 4.19 m level change. That contrast is the lesson worth teaching: the original map was not merely mistuned, it was the wrong model for the data.

### Things that will bite you

- **The offline node needs a URDF, not a `static_transform_publisher`.** The bag has no `/tf` or `/tf_static`, and `cartographer_offline_node` never subscribes to a live `/tf` — it reads TF only from the bag and from `-urdf_filenames`. Hence `urdf/hilti_alphasense_pandar.urdf`. The *online* node (`run_carto_live.sh`) can use a `static_transform_publisher`, and does; its trajectory lands within 0.2 m of the offline one.
- **Extrinsic direction.** `TfBridge::LookupToTracking()` asks for `lookupTransform(tracking_frame, frame_id)` = `T_imu_lidar`, so the URDF joint is parent `imu_sensor_frame` → child `PandarXT-32` with FAST-LIVO2's values used **as given**, no inversion. As it happens this particular rotation is a symmetric involution (`R = Rᵀ = R⁻¹`), so getting the direction backwards would leave the rotation bit-identical and only flip the 5.5 cm translation — a mistake too small to notice here, but not one to rely on.
- **`tracking_frame` must be the IMU frame** for 3D. Cartographer 3D integrates the IMU in the tracking frame with no IMU-to-tracking extrinsic; using the LiDAR frame is the most common way to break a 3D run.
- **Cartographer's map frame is z-UP, FAST-LIO2's is z-DOWN** (its world is the raw first IMU body frame, and this IMU's +z points down). The two z profiles are sign-flipped; only the *extent* is comparable. Gravity needs no config help — `ImuTracker` seeds its gravity vector from the first measurement.
- **`POSE_GRAPH.optimization_problem.huber_scale = 5e2`** is inherited verbatim from the old broken lua and is 50× Cartographer's default. At that scale the Huber loss is effectively quadratic for every residual in this run (max 0.134 m), i.e. outlier down-weighting is off. Harmless here — `min_score` 0.62 admitted no outliers — but it is untuned, not chosen.
- `roslaunch cartographer_ros hilti_3d.launch` still fails: the Dockerfile copies launch files into a directory `rospack` does not resolve. Use the scripts, which invoke the nodes with explicit paths.

## Watching it run (GUI on your desktop)

```bash
./run_carto_live.sh
```

That starts `cartographer_node` plus rviz with cartographer_rviz's `Submaps` display, so you watch submaps appear as the bag plays. It passes the X11 and GPU flags:

```
--runtime=/usr/bin/nvidia-container-runtime
-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
```

**No `xhost +local:root`** — podman here is rootless, so container root maps to your uid, which X already authorizes; earlier revisions of this file told you to loosen `xhost`, which is an unnecessary security downgrade. **No `--net=host`** either. With the NVIDIA flags rviz renders on the RTX 5090 (OpenGL 4.6); without them it falls back to software GL and is noticeably slower on a 21 M-point map.

Expect one cosmetic red herring: rviz's `RobotModel` display shows a red error, since this configuration has no full URDF for display purposes. The map still renders.

To inspect a **finished** map, use `visualize_pbstream.launch` (already in the image) rather than `map_server`, which is not installed:

```bash
rosrun cartographer_ros cartographer_pbstream_to_ros_map -pbstream_filename results/map.pbstream
```

Verifying a window mapped: `xwininfo -root -tree | grep -i rviz`, **not** `-root -children` — the window manager reparents it, so `-children` finds only a stray `Tool Properties` dock and looks like a failed launch.

## Caveats on the legacy KITTI demo

`velodyne_kitti_2D.lua` + `velodyne_kitti_uamc.launch` remain for backwards compatibility and are **not** verified. The launch chain resolves, but it declares `<node name="rviz" ... required="true">` so it dies without a display, its `rosbag play` node is commented out, and it needs a KITTI **ROS bag** publishing `/velodyne_points` (e.g. via `kitti2bag`) — this host has raw KITTI odometry files, not a bag.
