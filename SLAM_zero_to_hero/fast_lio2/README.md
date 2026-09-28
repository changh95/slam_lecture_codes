# FAST-LIO2

Tightly-coupled LiDAR-inertial odometry on an iterated error-state Kalman filter, registering raw points directly against an incrementally-built ikd-Tree map. The default dataset is the **Hilti SLAM Challenge 2022 `exp14_basement_2.bag`** (Hesai PandarXT-32 + Alphasense IMU, 74 s handheld basement walk).

- **Repo**: [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) — the master branch *is* FAST-LIO2 (built at commit `7cc4175`)
- **Paper**: [FAST-LIO2: Fast Direct LiDAR-Inertial Odometry](https://arxiv.org/abs/2107.06829) — Xu et al., IEEE T-RO 2022
- Predecessor: [FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter](https://arxiv.org/abs/2010.08196) — RA-L 2021

![FAST-LIO2 map and trajectory on Hilti 2022 exp14_basement_2](results/exp14/rviz_map.png)

## Data

```bash
python3 ../download_hilti_2022.py exp14_basement_2      # -> ~/data/hilti_2022/exp14_basement_2.bag (5.8 GB)
```

As of 2026-09 the script's S3 URL answers `403 AccessDenied`; Hilti moved the dataset to Hugging Face. Until the script is updated, fetch the bag and its ground truth directly:

```bash
HF=https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022/resolve/main
mkdir -p ~/data/hilti_2022 && cd ~/data/hilti_2022
curl -L -O $HF/rosbags/exp14_basement_2.bag                  # 6,260,771,085 bytes
curl -L -O $HF/ground_truth/exp14_basement_2_imu.txt         # TUM format, IMU frame, 689 poses
```

## Build

```bash
podman build -t localhost/slam_zero_to_hero:rviz_unified_controls ../rviz_unified_controls   # once, if missing
podman build -t localhost/slam_zero_to_hero:fast_lio2 .
```

Bakes ROS Noetic, Livox-SDK v1, `livox_ros_driver` and FAST_LIO (all pinned) into `/catkin_ws`, plus `rviz` (with the course's unified mouse controls, built from `localhost/slam_zero_to_hero:rviz_unified_controls`), `Xvfb`, `xwd` and ImageMagick for the screenshots. CPU only; no GPU needed.

## Run

```bash
mkdir -p results/exp14
timeout 900 podman run --rm \
  -v ~/data/hilti_2022:/data:ro \
  -v "$PWD/results/exp14":/out \
  -v "$PWD/config/hilti_pandarxt32.yaml":/catkin_ws/src/FAST_LIO/config/hilti_pandarxt32.yaml:ro \
  -v "$PWD/config/hilti.rviz":/catkin_ws/src/FAST_LIO/rviz_cfg/hilti.rviz:ro \
  -v "$PWD/launch/mapping_hilti.launch":/catkin_ws/src/FAST_LIO/launch/mapping_hilti.launch:ro \
  -v "$PWD/scripts":/scripts:ro \
  -v "$PWD/run_hilti_offline.sh":/run.sh:ro \
  -e CONFIG=hilti_pandarxt32 -e SCREENSHOT=1 \
  localhost/slam_zero_to_hero:fast_lio2 bash /run.sh > results/exp14/console.log 2>&1

python3 scripts/traj_stats.py results/exp14/fastlio_traj_tum.txt
python3 scripts/ate.py results/exp14/fastlio_traj_tum.txt ~/data/hilti_2022/exp14_basement_2_imu.txt
```

`run_hilti_offline.sh` starts a container-private `roscore`, launches `mapping_hilti.launch`, plays the bag at real time, logs `/Odometry` to `fastlio_traj_tum.txt`, then SIGINTs the mapper so it flushes `pos_log.txt`. `SCREENSHOT=1` also starts rviz (`config/hilti.rviz`: accumulated `/cloud_registered` coloured by height + `/path` in magenta) on a private Xvfb and saves `rviz_midway.png` (35 s into the bag) and `rviz_map.png` (end). Drop `-e SCREENSHOT=1` for a plain headless run. Other knobs: `RATE`, `DURATION` (seconds of bag), `SAVE_PCD=1` (dumps `scans.pcd`, ~370 MB), `RELAY=1` (see below).

No `xhost` change and no `--net=host`: the ROS master lives in the container's own network namespace, so several ROS containers can run at once.

### Watching it live on your desktop

Replace `-e SCREENSHOT=1` with the X11 + GPU flags and `-e RVIZ=true`:

```bash
mkdir -p results/gui
timeout 900 podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/hilti_2022:/data:ro \
  -v "$PWD/results/gui":/out \
  -v "$PWD/config/hilti_pandarxt32.yaml":/catkin_ws/src/FAST_LIO/config/hilti_pandarxt32.yaml:ro \
  -v "$PWD/config/hilti.rviz":/catkin_ws/src/FAST_LIO/rviz_cfg/hilti.rviz:ro \
  -v "$PWD/launch/mapping_hilti.launch":/catkin_ws/src/FAST_LIO/launch/mapping_hilti.launch:ro \
  -v "$PWD/scripts":/scripts:ro \
  -v "$PWD/run_hilti_offline.sh":/run.sh:ro \
  -e RVIZ=true -e CONFIG=hilti_pandarxt32 \
  localhost/slam_zero_to_hero:fast_lio2 bash /run.sh
```

Mouse in rviz (course-wide [unified controls](../rviz_unified_controls/)): left drag rotates, wheel zooms, right or middle drag pans. `config/hilti.rviz` and the image's upstream `loam_livox.rviz` both use `slam_zero_to_hero/UnifiedOrbit`.

## Results — Hilti 2022 `exp14_basement_2`

Measured 2026-09-27 with the Run command above (Ryzen 9 7950X, host shared with other jobs):

| Metric | Value |
|---|---|
| Poses | 737 from 740 scans (1 empty first scan + 2 for IMU init) |
| Path length | 37.934 m (GT: 37.803 m) |
| **ATE RMSE** vs `exp14_basement_2_imu.txt` (SE(3) aligned, 686 matched poses) | **0.050 m** (median 0.045, max 0.147); a second, plain headless run gave 0.057 m |
| Mapper time per scan (`ave total`) | 16.5 ms with software-rendered rviz, 20.7 ms headless on the loaded host; 6.05 ms on an idle host (2026-08-05) |
| Wall clock | 78 s playback of the 74 s bag at `RATE=1.0` |

The run-to-run spread in ATE (5.0–5.7 cm) comes from real-time playback: which IMU samples land before each scan varies slightly. Outputs of the default run are in [`results/exp14/`](results/exp14/) (`traj_stats.txt`, `ate.txt`, logs, screenshots).

## Supported datasets

| Dataset | Config | Status |
|---|---|---|
| **Hilti 2022** `exp14_basement_2.bag` (default) | `config/hilti_pandarxt32.yaml` | ✅ verified 2026-09-27: 737 poses, 37.93 m path, **ATE 0.050 m**. Hesai PandarXT-32 + Alphasense IMU. |
| same, real per-point stamps | `config/hilti_pandarxt32_relay.yaml` + `RELAY=1` | ✅ verified 2026-08-05; agrees with the above to within 8.5 cm over 38 m |
| Livox Avia / Horizon / Mid-360 | upstream `avia.yaml`, `horizon.yaml`, `mid360.yaml` | Shipped by upstream, not verified here |
| Ouster-64, Velodyne | upstream `ouster64.yaml`, `velodyne.yaml` | Shipped by upstream, not verified here |
| Any LiDAR + IMU ROS bag | copy a config | Set `lid_topic`/`imu_topic`, `lidar_type` (1 Livox, 2 Velodyne, 3 Ouster), `scan_line`, `scan_rate`, and the LiDAR↔IMU extrinsic |

There is no upstream Hesai config; why the Velodyne branch is the right home for it, and why the 740 `Failed to find match for field 'time'` warnings are expected, are in [NOTES.md](NOTES.md).
