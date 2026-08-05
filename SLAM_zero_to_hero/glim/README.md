# GLIM

Versatile range-inertial SLAM on GTSAM factor graphs: fixed-lag smoothing odometry, submap-based local mapping, and global factor-graph optimization with GPU-accelerated scan-matching factors.

- **Repo**: [koide3/glim](https://github.com/koide3/glim) (`v1.0.0`)
- **Paper**: [GLIM: 3D Range-Inertial Localization and Mapping with GPU-Accelerated Scan Matching Factors](https://arxiv.org/abs/2407.10344) — Koide et al., Robotics and Autonomous Systems 2024
- Underlying registration: [Voxelized GICP for Fast and Accurate 3D Point Cloud Registration](https://doi.org/10.1109/ICRA48506.2021.9560835) — ICRA 2021

## Build

```bash
podman build -t slam_zero_to_hero:glim .
```

Built with CUDA for sm_120 (RTX 5090). Upstream ships **no executable** — a plain glim build produces only libraries — so this image adds `glim_kitti`, a driver that feeds KITTI scans through the real GLIM pipeline.

## Run with the GUI

```bash
podman run --rm -it \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e GLIM_KITTI_VIEWER=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/kitti_vo_slam/extracted/dataset/sequences/00:/data/seq00:ro \
  -v "$(pwd)/results":/output \
  slam_zero_to_hero:glim \
  glim_kitti /usr/local/share/glim_kitti/config /data/seq00 /output/dump
```

An iridescence window (`screen`) shows the map and trajectory building up, and stays open when mapping finishes so you can inspect the result — close it to exit. Closing it mid-run stops early and still saves.

Without `GLIM_KITTI_VIEWER=1` the run is fully headless. No `xhost` change and no `--net=host` are needed.

```
glim_kitti <config_dir> <kitti_sequence_dir> <dump_dir> [max_scans] [stamp_offset]
```

For a **ROS 2 bag**, use upstream's `glim_rosbag` instead — the image ships ROS 2 Jazzy and `glim_ros2` v1.0.0:

```bash
mkdir -p out
podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/Korea_drive/KOREA_DRIVE:/bag:ro \
  -v "$PWD/out":/tmp/dump \
  slam_zero_to_hero:glim \
  glim_rosbag /bag --ros-args -p config_path:=/usr/local/share/glim_korea/config_viewer
```

Four things are easy to get wrong here, all of them silent:

- Mount the bag **directory**, not the `.db3` — the `.db3`'s embedded `metadata` table names a different filename than the file on disk.
- The dump path is **hard-coded to `/tmp/dump`**, so bind-mount that or you get no output.
- Parameters need `--ros-args -p …`. A bare `-p config_path:=…` is parsed as a *remap rule*, silently falls back to the default config, and then segfaults with no display.
- Add `-p auto_quit:=true` for headless runs (with `config_path:=…/glim_korea/config`), otherwise it calls `rclcpp::spin()` and never exits.

## Supported datasets

| Dataset | Config | Status |
|---|---|---|
| **KITTI odometry** seq 00 | `/usr/local/share/glim_kitti/config` | ✅ verified: 4531 poses, 3708 m, ATE 11.5 m over 3.7 km |
| **KITTI odometry** seq 04 | same | ✅ verified: ATE **2.60 m** over 394 m (0.66 % of path) |
| same, best global optimization | `config_posegraph` | Explicit loop detection; best seq 00 ATE (10.2 m) |
| same, CUDA path | `config_gpu` | GPU VGICP factors. Same accuracy, **no speedup on KITTI** — GPU odometry needs an IMU, which this data lacks |
| **Korea_drive** — ROS 2 bag, 27 min vehicle drive (Hesai 109k pts/scan + 100 Hz IMU + GNSS) | `/usr/local/share/glim_korea/config` (headless) or `config_viewer` | ✅ verified: 16,367 poses over 11.06 km, **ATE 5.3 m 2D / 16.5 m 3D** against the GNSS track |
| Any other LiDAR **+ IMU** ROS 2 bag | copy `config_korea/` | Set the topics in `config_ros.json` and `T_lidar_imu` in `config_sensors.json`. This is where the GPU pays off — see below. |

KITTI requires velodyne `.bin` scans plus `times.txt`; sequences 00 and 04 are extracted on this host.

**With an IMU, the GPU finally earns its keep.** On KITTI it could not: `OdometryEstimationGPU` hard-requires an IMU, so velodyne-only data left the GPU carrying only mapping factors, at no speedup. On Korea_drive it is **1.5–1.8× faster** (77.7 vs 49.6 scans/s) *and* **~4× more accurate** (2.18 m vs 8.65 m 3D rmse on a 400 s slice) than CPU odometry — same config apart from one line.

Why CUDA 12.9.1 rather than 13.x, why the GPU buys nothing here, and the KITTI-specific config edits (one of which is load-bearing) are in [NOTES.md](NOTES.md).
