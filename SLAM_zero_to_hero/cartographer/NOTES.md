# Cartographer — implementation notes

Reference material behind the short [README.md](README.md): exact verified numbers, the
reasoning behind each config value, and the upstream bugs and gotchas found while getting
this running. You do not need any of it to build and run — start with the README.

---

Google's real-time 2D and 3D LiDAR SLAM running on ROS Noetic.

- **Repo**: bundled snapshot in `cartographer.tar.xz`
- **Sensors**: 2D laser, 3D LiDAR, optional IMU
- **GPU**: not required for SLAM; used only to render rviz

## Build

```bash
podman build -t slam_zero_to_hero:cartographer .
```

The image extracts `cartographer.tar.xz`, builds Abseil from source, then runs `catkin build` over `cartographer`, `cartographer_ros`, and `cartographer_rviz`. `rviz` is installed; `map_server` is not. The last layers add Xvfb, x11-utils, xdotool and ImageMagick for headless rviz screenshots, then build the course's rviz view controllers from `localhost/slam_zero_to_hero:rviz_unified_controls` (build that image first, see `../rviz_unified_controls/README.md`). A `sed` switches the upstream `demo_2d.rviz` / `demo_3d.rviz` from `rviz/TopDownOrtho` to `slam_zero_to_hero/UnifiedTopDownOrtho`; `config/hilti_*.rviz` use `slam_zero_to_hero/UnifiedOrbit`. Left drag rotates, the wheel zooms, right or middle drag pans.

## Verified run — Hilti 2022 `exp21_outside_building.bag` (3D + IMU, default)

A 152 s handheld walk around the outside of a building: 1,528 sweeps, 130 m of path, a map about 87 × 84 m. It uses the same rig, topics and LiDAR↔IMU extrinsic as exp14, so `urdf/`, the static transform in `run_carto_live.sh` and `scripts/hesai_add_time_field.py` are unchanged. The per-point times after conversion run 0.000 to 0.101 s. Outdoors the returns reach 20–60 m (p95), up to 151 m.

**Ground truth.** Hilti publishes the IMU position at 5 surveyed timestamps (`ground_truth/exp21_outside_building.txt` on the Hugging Face mirror). The points are up to 44 m apart; the quaternions in the file are dummies. `scripts/eval_survey.py` interpolates the trajectory at those 5 times, fits one rigid SE(3) transform (no scale), and prints the residuals. Five points cannot say much about local accuracy, but they are enough to expose height drift.

**Result** (`config/hilti_outdoor_3d.lua`, measured 2026-09-28 at load average 27–42): **0.093 m RMSE, max 0.112 m**. The per-point errors are 0.095 / 0.052 / 0.112 / 0.085 / 0.109 m. There are 1,518 poses, 1,536 loop-closure match attempts and 192 accepted constraints. SLAM takes 31 s of wall clock. The run is deterministic: pbstream md5 `0eb1bbb6…` in two runs at different host loads. The online node (`run_carto_live.sh`, 1x, rviz on the GPU) gives 1,527 poses and 0.099 m RMSE, and wrote the same 24,608,369 B pbstream in two runs.

**Tuning, one change at a time** (survey RMSE; each row changes only that value from the row above, unless it says otherwise):

| Config | RMSE | What it shows |
|---|---|---|
| `hilti_3d_lio.lua` (basement config) unchanged | 0.517 m | xy within 0.12 m at every point; **height drifts 1.43 m** by the end |
| first outdoor draft: 80 m range, 10 cm, 160 sweeps/submap, 30 m loop search | 0.503 m | range alone does not fix the height |
| + loop closure z window ±3 m (stock ±1 m) | 0.503 m | no change by itself |
| + sampling ratio 0.3, `min_score` 0.55 (Cartographer's default; basement uses 0.62) | 0.295 m | loops can now close; 35 → 150 constraints |
| + `max_range` 120 m (and the coarse voxel filter's range) | 0.272 m | |
| + high-resolution submap 15 cm (voxel filter stays 10 cm) | 0.120 m | the big step. A 5–10 cm grid is mostly empty at 30 m from a 32-beam sensor |
| + `occupied_space_weight_0` 3 (stock 1) | **0.093 m** | the fine grid counts for more against the coarse one. **Shipped.** |

The following changes were rejected, each tested against a nearby config. 5 cm high resolution with a 5 cm voxel filter: 0.550 m, no loop closures at all. 20 cm: 0.155 m. 100 sweeps per submap: 0.241 m. `min_score` 0.62: 0.352 m. `occupied_space_weight_0` 6: 0.208 m. Low-resolution submap 60 cm: 0.227 m. Scan matcher `rotation_weight` 2e3: 0.820 m, and 1e2: 0.404 m. Pose-graph IMU `rotation_weight` 1e5: 0.506 m. `optimize_every_n_nodes` 50: 0.449 m. `max_constraint_distance` 50 m (0.101 m) and sampling ratio 0.5 (0.098 m) were neutral.

Take the last centimetres with a pinch of salt: 5 survey points, and the configs from 0.093 to 0.12 m are not different in any way a reader could see in the map. What the numbers are good for is the direction. Coarser matching and working loop closure remove the height drift; the xy error stays around 0.1 m throughout.

**Map quality.** `map.pgm` (the 2D projection of the submaps, 0.05 m/px) shows the building faces, including the curved facade, as single thin lines, and no wall is doubled. The occupancy slab from `assets_writer_hilti_outdoor_grid.lua` (0.10 m/px, 1133 × 1333 px) has 1.41 % occupied cells, a free/occupied ratio of 20.6, and a median occupied run of 0.30 m. `pgm_stats.py` assumes 0.05 m/px, so double the lengths it prints for this grid. The dense cloud `assets_map3d.ply` (`assets_writer_hilti_outdoor.lua`, 80 m range, 10 cm voxels) has 49,928,673 points (800 MB PLY, 600 MB PCD).


## Verified run — Hilti 2022 `exp14_basement_2.bag` (3D + IMU)

This was the default demo until 2026-09-28, and it is still the smaller indoor
demo. The exact commands are in the [README](README.md): pass
`CFG=hilti_3d_lio.lua BAG=/data/exp14_basement_2_carto.bag` (plus
`ASSETS_CFG=assets_writer_hilti.lua` offline, `RVIZ_CFG=hilti_3d.rviz` live).
`run_carto.sh` and `run_carto_live.sh` run **inside** the container. They read
the output of `scripts/hesai_add_time_field.py` and bind-mount `config/`, `urdf/`
and `scripts/`, so nothing needs rebuilding to change a parameter. Do not edit
`run_carto.sh` while a run is using it: bash reads the script as it goes, and a
run that is already going will pick up the half-edited file. Neither needs `--net=host` — each starts its own roscore in the container's
network namespace.

Last verified: Ryzen 9 7950X, 2026-08-05; re-run 2026-09-27 with byte-identical pbstream and the same numbers. Independently re-measured from the artifacts by a second pass, with scripts calibrated against this repo's published FAST-LIO2 reference before being trusted.

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

`run_carto_live.sh` starts `cartographer_node`, a `static_transform_publisher` for the
extrinsic, `rosbag play` (default 1x, `RATE=3` to speed up) and rviz on
`config/$RVIZ_CFG`: `hilti_outdoor_3d.rviz` by default (a 95 m orbit over the building,
5 m grid) or `hilti_3d.rviz` for the basement. Both show cartographer_rviz `Submaps`,
`/trajectory_node_list` and `/scan_matched_points2`. With `XVFB=1` rviz renders on a
private Xvfb display (llvmpipe, ~30 fps). `SHOT=<png>` captures the rviz window (found
with `xdotool`, so a shared desktop's other windows stay out of it) when the bag ends. The script exports
`DISABLE_ROS1_EOL_WARNINGS=1`; without it rviz opens a modal "Noetic end-of-life" dialog
over the map.

**No `xhost +local:root`** — podman here is rootless, so container root maps to your uid, which X already authorizes. **No `--net=host`** either. With the NVIDIA flags rviz renders on the GPU; without them it falls back to software GL.

The `Trajectories` (MarkerArray) display shows a red status; the trajectory line still draws.

To inspect a **finished** map, use `visualize_pbstream.launch` (already in the image) rather than `map_server`, which is not installed:

```bash
rosrun cartographer_ros cartographer_pbstream_to_ros_map -pbstream_filename results/map.pbstream
```

Verifying a window mapped: `xwininfo -root -tree | grep -i rviz`, **not** `-root -children` — the window manager reparents it, so `-children` finds only a stray `Tool Properties` dock and looks like a failed launch.

## Caveats on the legacy KITTI demo

`velodyne_kitti_2D.lua` + `velodyne_kitti_uamc.launch` remain for backwards compatibility and are **not** verified. The launch chain resolves, but it declares `<node name="rviz" ... required="true">` so it dies without a display, its `rosbag play` node is commented out, and it needs a KITTI **ROS bag** publishing `/velodyne_points` (e.g. via `kitti2bag`) — this host has raw KITTI odometry files, not a bag.
