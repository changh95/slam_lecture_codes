# Cartographer

Google's real-time 2D and 3D LiDAR SLAM running on ROS Noetic.

- **Repo**: bundled snapshot in `cartographer.tar.xz`
- **Sensors**: 2D laser, 3D LiDAR, optional IMU
- **GPU**: not required

## Build

```bash
podman build -t slam_zero_to_hero:cartographer .
```

The image extracts `cartographer.tar.xz`, builds Abseil from source, then runs `catkin build` over `cartographer`, `cartographer_ros`, and `cartographer_rviz`. Two configurations are baked into the image:

| Config | Mode | Topic | Use |
|---|---|---|---|
| `velodyne_kitti_2D.lua` + `velodyne_kitti_uamc.launch` | 2D | `/velodyne_points` | Legacy KITTI demo (KITTI bag from Google Drive). |
| `hilti_3d.lua` + `run_hilti_offline.sh` | 2D (LiDAR-only) | `/hesai/pandar` | Verified on Hilti 2022 `exp14_basement_2.bag`. |

The `hilti_3d.lua` file is named for backwards compatibility with earlier docs but actually runs the trajectory builder in **2D mode** — the Hesai PandarXT-32 point cloud is projected onto the horizontal plane to drive the 2D scan matcher. This avoids the IMU/TF prerequisites that the 3D builder pulls in.

## Verified run — Hilti 2022 `exp14_basement_2.bag`

```bash
mkdir -p results
podman run --rm \
  -v ~/data/hilti_2022:/data:ro \
  -v "$(pwd)/results":/out:rw \
  slam_zero_to_hero:cartographer \
  /usr/local/bin/run_hilti_offline.sh /data/exp14_basement_2.bag
```

The bundled `run_hilti_offline.sh` starts a local roscore, runs `cartographer_offline_node` against the bag (with `points2:=/hesai/pandar` remap), then converts the resulting pbstream into a 2D occupancy grid via `cartographer_pbstream_to_ros_map`.

Outputs:

| File | Description |
|---|---|
| `results/hilti_basement.pbstream` | Cartographer pose graph + submaps (≈ 1 MB) |
| `results/hilti_basement.pgm` | 2D occupancy projection (≈ 660 KB) |
| `results/hilti_basement.yaml` | ROS map metadata (origin, resolution) |

Last verified: Ryzen 9 7950X. 74-s bag processed offline → 1 MB pbstream + 660 KB occupancy map.

## Visualizing the result

```bash
xhost +local:root
podman run --rm -it \
  --net=host --ipc=host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/results":/out:ro \
  slam_zero_to_hero:cartographer
# inside container:
rosrun map_server map_server /out/hilti_basement.yaml &
rviz   # add a Map display, topic /map
```

## What the lua does

`hilti_3d.lua` keys for the smoke test:

- `MAP_BUILDER.use_trajectory_builder_2d = true`
- `tracking_frame = "PandarXT-32"`, `published_frame = "PandarXT-32"` (= the LiDAR's own frame_id, no TF chain needed)
- `num_point_clouds = 1` (subscribes to `/points2`, which the run script remaps to `/hesai/pandar`)
- `TRAJECTORY_BUILDER_2D.use_imu_data = false` (LiDAR-only)
- `min_z = -0.5`, `max_z = 2.0` to crop the 3D cloud to the floor-level slab the 2D matcher expects
- `use_online_correlative_scan_matching = true` for robustness on the Hesai's relatively coarse vertical resolution
