# Cartographer

Google's real-time 2D and 3D LiDAR SLAM running on ROS Noetic.

- **Repo**: bundled snapshot in `cartographer.tar.xz`
- **Sensors**: 2D laser, 3D LiDAR, optional IMU
- **GPU**: not required

## Build

```bash
podman build -t slam_zero_to_hero:cartographer .
```

The image extracts `cartographer.tar.xz`, builds Abseil from source, then runs `catkin build` over `cartographer`, `cartographer_ros`, and `cartographer_rviz`. Two configurations are baked in:

| Config | Mode | Topic | Use |
|---|---|---|---|
| `hilti_3d.lua` + `run_hilti_offline.sh` | 2D (LiDAR-only) | `/hesai/pandar` | Verified below on Hilti 2022 `exp14_basement_2.bag`. |
| `velodyne_kitti_2D.lua` + `velodyne_kitti_uamc.launch` | 2D | `/velodyne_points` | Legacy KITTI demo. Launch file resolves, but see the caveats at the end — it is **not** verified end-to-end. |

`hilti_3d.lua` is named for backwards compatibility with earlier docs but runs the trajectory builder in **2D mode**: the Hesai PandarXT-32 cloud is projected onto the horizontal plane to drive the 2D scan matcher. This avoids the IMU/TF prerequisites the 3D builder pulls in.

## Verified run — Hilti 2022 `exp14_basement_2.bag`

```bash
mkdir -p results
podman run --rm \
  -v ~/data/hilti_2022:/data:ro \
  -v "$(pwd)/results":/out:rw \
  slam_zero_to_hero:cartographer \
  /usr/local/bin/run_hilti_offline.sh /data/exp14_basement_2.bag
```

The bundled `run_hilti_offline.sh` starts a local roscore, runs `cartographer_offline_node` against the bag (with `points2:=/hesai/pandar` remapped), then converts the resulting pbstream into an occupancy grid via `cartographer_pbstream_to_ros_map`. **No `--net=host`**: the roscore lives in the container's own network namespace, so this can run alongside the other ROS containers in this repo without fighting over port 11311.

Outputs:

| File | Size | Description |
|---|---|---|
| `results/hilti_basement.pbstream` | 1,044,861 B | Cartographer pose graph + submaps |
| `results/hilti_basement.pgm` | 676,081 B | Occupancy grid, 859 × 787 px at 0.05 m/px (42.95 × 39.35 m), origin `[-13.3883, -19.0855, 0.0]` |
| `results/hilti_basement.yaml` | 133 B | ROS map metadata |

Last verified: Ryzen 9 7950X, 2026-08-05. Full 74.0128 s of bag consumed (740 Hesai clouds). `cartographer_offline_node` reports **elapsed wall clock 12.0–12.7 s, peak memory 78–82 MB** across two runs — roughly **6× real time**. Result: **5 submaps, 390 trajectory nodes** (52.7 % of the 740 clouds survive the motion filter), and **14 loop-closure constraints** with match scores min 0.649 / max 0.843 / mean 0.721. Whole container, including roscore startup and pgm conversion, takes ~17 s.

The pipeline is deterministic on this bag: two independent runs produced **byte-identical** outputs (1,044,861 / 676,081 / 133 B).

### The map is not a clean floorplan — and that is worth teaching

Do not present this run's occupancy grid as a good map. Measured on the emitted `.pgm`: **occupied cells 37,715 (5.58 %) outnumber free cells 32,415 (4.79 %)**, 71.1 % of the grid is still unknown, and occupied horizontal run lengths reach 8.30 m (median 0.25 m, p90 1.80 m). Walls are smeared rather than crisp.

The causes are all visible in the configuration, which makes this a useful exercise rather than a failure:

- The sensor is a **handheld** Hesai PandarXT-32 that tilts as the operator walks, but the cloud is flattened into a fixed `min_z = -0.5 … max_z = 2.0` slab. Tilt drags ceiling and floor returns into the 2D scan, and they land at the wrong range.
- `TRAJECTORY_BUILDER_2D.use_imu_data = false`, so nothing compensates that tilt.
- `POSE_GRAPH.constraint_builder.min_score = 0.62` is permissive; the 14 accepted constraints score as low as 0.649.

Fixing it properly means the 3D trajectory builder with the Alphasense IMU wired in — a good follow-up exercise. For a LiDAR-inertial comparison on the same bag with a metrically sound trajectory, see `../fast_lio2/README.md` (37.93 m path) and `../fast_livo2/README.md`.

Note the two resolutions are different knobs and both appear in the outputs: the **submap grid** is 0.1 m (set in the lua), while `cartographer_pbstream_to_ros_map` writes the **map** at 0.05 m/px.

## Visualizing the result

`map_server` is **not installed** in this image (`rosrun map_server map_server` fails with `package 'map_server' not found`), so the usual map_server + rviz recipe does not work as-is. `rviz` itself is present at `/opt/ros/noetic/bin/rviz`.

The simplest inspection needs no ROS at all — the `.pgm` is a plain image:

```bash
python3 -c "
from PIL import Image; im = Image.open('results/hilti_basement.pgm'); print(im.size, im.mode); im.show()"
```

If you do want `map_server`, install it into a derived image (`apt-get install -y ros-noetic-map-server`) and note that `hilti_basement.yaml` stores an **absolute in-container path** (`image: /out/hilti_basement.pgm`), so the results directory must be mounted at `/out` for the yaml to resolve.

## What the lua does

`hilti_3d.lua` keys for this run:

- `MAP_BUILDER.use_trajectory_builder_2d = true`
- `tracking_frame = "PandarXT-32"`, `published_frame = "PandarXT-32"` (the LiDAR's own `frame_id`, so no TF chain is needed)
- `num_point_clouds = 1` (subscribes to `/points2`, which the run script remaps to `/hesai/pandar`)
- `TRAJECTORY_BUILDER_2D.use_imu_data = false` (LiDAR-only)
- `min_z = -0.5`, `max_z = 2.0` to crop the 3D cloud to the floor-level slab the 2D matcher expects
- `use_online_correlative_scan_matching = true` for robustness on the Hesai's coarse vertical resolution

## Caveats on the legacy KITTI demo

The launch chain resolves — `roslaunch cartographer_ros velodyne_kitti_uamc.launch --files` finds the launch file and `velodyne_kitti_2D.lua` — but it is not runnable as shipped:

- It declares `<node name="rviz" ... required="true">`, so the whole launch **dies immediately without a display**.
- Its `rosbag play` node is commented out, so you must play a bag by hand.
- It needs a KITTI **ROS bag** publishing `/velodyne_points` (e.g. via `kitti2bag`). This host has raw KITTI odometry files, not a bag.

Also note `roslaunch cartographer_ros hilti_3d.launch` fails directly (`is neither a launch file in package [cartographer_ros] nor is [cartographer_ros] a launch file name`): the Dockerfile copies it into a directory `rospack` does not resolve. Use `run_hilti_offline.sh`, which invokes `cartographer_offline_node` with explicit paths.
