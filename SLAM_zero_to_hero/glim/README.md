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

## Supported datasets

| Dataset | Config | Status |
|---|---|---|
| **KITTI odometry** seq 00 | `/usr/local/share/glim_kitti/config` | ✅ verified: 4531 poses, 3708 m, ATE 11.5 m over 3.7 km |
| **KITTI odometry** seq 04 | same | ✅ verified: ATE **2.60 m** over 394 m (0.66 % of path) |
| same, best global optimization | `config_posegraph` | Explicit loop detection; best seq 00 ATE (10.2 m) |
| same, CUDA path | `config_gpu` | GPU VGICP factors. Same accuracy, **no speedup on KITTI** — GPU odometry needs an IMU, which this data lacks |
| LiDAR **+ IMU** bags | — | Needs `glim_ros1`/`glim_ros2`, which are not in this image. This is where the GPU would actually pay off. |

Requires velodyne `.bin` scans plus `times.txt`; sequences 00 and 04 are extracted on this host.

Why CUDA 12.9.1 rather than 13.x, why the GPU buys nothing here, and the KITTI-specific config edits (one of which is load-bearing) are in [NOTES.md](NOTES.md).
