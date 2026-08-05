# Cartographer

Google's real-time 2D and 3D LiDAR SLAM: local scan matching into submaps, plus a background pose-graph optimization with branch-and-bound loop closure.

- **Repo**: [cartographer-project/cartographer](https://github.com/cartographer-project/cartographer) (bundled here as `cartographer.tar.xz`) — upstream is no longer actively maintained; ROS 2 users are directed to [ros2/cartographer_ros](https://github.com/ros2/cartographer_ros)
- **Paper**: [Real-Time Loop Closure in 2D LIDAR SLAM](https://research.google/pubs/real-time-loop-closure-in-2d-lidar-slam/) — Hess et al., IEEE ICRA 2016. (Covers the 2D system; Cartographer's 3D SLAM has no separate paper.)

## Build

```bash
podman build -t slam_zero_to_hero:cartographer .
```

## Run with the GUI

```bash
./run_carto_live.sh
```

Starts `cartographer_node` plus rviz with cartographer_rviz's `Submaps` display, so you watch submaps appear as the bag plays. The script passes the X11 and GPU flags itself:

```
--runtime=/usr/bin/nvidia-container-runtime
-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
```

No `xhost` change and no `--net=host` are needed. Expect one cosmetic red error on rviz's `RobotModel` display — there is no URDF for display purposes; the map still renders.

For the offline run (no GUI, writes the 3D map and occupancy grids):

```bash
./run_carto.sh
```

## Supported datasets

| Dataset | Config | Status |
|---|---|---|
| **Hilti 2022** `exp14_basement_2.bag` | `config/hilti_3d_lio.lua` — 3D + IMU | ✅ verified: 730 poses, 38.4 m path, **0.084 m RMSE** against FAST-LIO2 |
| same, 2D comparison | `config/hilti_2d_imu.lua` — 2D + IMU | Kept for contrast: tilt compensation fixes the smearing, but 2D cannot represent this sequence's 4.19 m level change |
| Any ROS bag with a LiDAR `PointCloud2` + `sensor_msgs/Imu` | copy `hilti_3d_lio.lua` | Needs the sensor↔IMU extrinsic supplied via `urdf/`, and a per-point `time` field for de-skewing |
| KITTI, 2D (legacy) | `velodyne_kitti_2D.lua` | ⚠️ not verified — needs a `kitti2bag` bag and cannot run headless (`rviz` is `required="true"`) |

The original 2D config produced a badly smeared map; the reason, and the de-skewing bug that mattered more than any parameter, are in [NOTES.md](NOTES.md).
