# KISS-SLAM

LiDAR SLAM built on KISS-ICP: point-to-point ICP odometry with an adaptive threshold, local maps, a density-map loop detector, and g2o pose-graph optimization.

- **Repo**: [PRBonn/kiss-slam](https://github.com/PRBonn/kiss-slam) (`v0.0.2`)
- **Paper**: [KISS-SLAM: A Simple, Robust, and Accurate 3D LiDAR SLAM System With Enhanced Generalization Capabilities](https://arxiv.org/abs/2503.12660) — Guadagnino et al., IEEE/RSJ IROS 2025
- Front end: [KISS-ICP: In Defense of Point-to-Point ICP](https://arxiv.org/abs/2209.15397) — Vizzo et al., IEEE RA-L 2023

## Build

```bash
podman build -t slam_zero_to_hero:kiss_slam .
```

Bundles `kiss-slam==0.0.2` plus the pure-Python `rosbags` reader, so the `rosbag` dataloader needs no ROS install, no roscore, and no network.

## Run with the GUI

```bash
podman run --rm -it \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results":/out -w /out \
  slam_zero_to_hero:kiss_slam \
  kiss_slam_pipeline --visualize --dataloader kitti --sequence 00 /data
```

An Open3D window (`RegistrationVisualizer`) shows the local map and growing trajectory.

**It starts paused** — press `space` to run, `n` to step, `c` to re-centre the view, `esc` to quit. The hint is printed only to stdout, so the window looks frozen until you do. Drop `--visualize` for a headless run; the viewer costs roughly 60 % of throughput.

No `xhost` change and no `--net=host` are needed.

## Supported datasets

| Dataset | Command | Status |
|---|---|---|
| **KITTI odometry** | `--dataloader kitti --sequence 00 /data` | ✅ verified: seq 00 ATE **5.59 m / 0.58 %** with 7 loop closures; seq 04 ATE 0.59 m. The only loader here that gives you ground truth for free. |
| **Hilti 2022** | `--config config/hilti_indoor.yaml --dataloader rosbag --topic /hesai/pandar <bag>` | ✅ verified **only with that config** — the stock defaults are tuned for outdoor driving and diverge badly indoors |
| Any folder of `.bin` / `.pcd` / `.ply` | `--dataloader generic <dir>` | Writes frame indices as timestamps rather than real seconds |
| ROS 1 `.bag` / ROS 2 `.db3` | `--dataloader rosbag --topic <topic> <bag>` | via the bundled `rosbags` reader |
| TUM, MulRan, nuScenes, NCLT, Apollo, Ouster, mcap, … | `--dataloader <name>` | 14 loaders inherited from KISS-ICP |

Not applicable to EuRoC (vision-only, no LiDAR).

Why the Hilti defaults fail, and how that was established against the raw scans, is in [NOTES.md](NOTES.md).
