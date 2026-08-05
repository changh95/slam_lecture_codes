# FAST-LIO2

Tightly-coupled LiDAR-inertial odometry on an iterated error-state Kalman filter, registering raw points directly against an incrementally-built ikd-Tree map.

- **Repo**: [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) — the master branch *is* FAST-LIO2
- **Paper**: [FAST-LIO2: Fast Direct LiDAR-Inertial Odometry](https://arxiv.org/abs/2107.06829) — Xu et al., IEEE T-RO 2022
- Predecessor: [FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter](https://arxiv.org/abs/2010.08196) — RA-L 2021

## Build

```bash
podman build -t slam_zero_to_hero:fast_lio2 .
```

Bakes ROS Noetic, Livox-SDK v1, `livox_ros_driver` and FAST_LIO into `/catkin_ws`, plus `rviz` for the GUI.

## Run with the GUI

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
  -v "$PWD/launch/mapping_hilti.launch":/catkin_ws/src/FAST_LIO/launch/mapping_hilti.launch:ro \
  -v "$PWD/scripts":/scripts:ro \
  -v "$PWD/run_hilti_offline.sh":/run.sh:ro \
  -e RVIZ=true -e CONFIG=hilti_pandarxt32 \
  slam_zero_to_hero:fast_lio2 bash /run.sh
```

rviz shows the point cloud accumulating and the body frame moving as the bag plays. Drop `-e RVIZ=true` for a headless run — everything else is identical, and the trajectory lands in `results/gui/fastlio_traj_tum.txt` either way.

No `xhost` change and no `--net=host` are needed; the container runs its own roscore in a private network namespace, so several ROS containers can run at once.

## Supported datasets

| Dataset | Config | Status |
|---|---|---|
| **Hilti 2022** `exp14_basement_2.bag` | `config/hilti_pandarxt32.yaml` | ✅ verified: 737 poses, 37.93 m path, 6.05 ms/scan. Hesai PandarXT-32 + Alphasense IMU. |
| same, real per-point stamps | `config/hilti_pandarxt32_relay.yaml` + `RELAY=1` | ✅ verified; agrees with the above to within 8.5 cm over 38 m |
| Livox Avia / Horizon / Mid-360 | upstream `avia.yaml`, `horizon.yaml`, `mid360.yaml` | Shipped by upstream, not verified here |
| Ouster-64, Velodyne | upstream `ouster64.yaml`, `velodyne.yaml` | Shipped by upstream, not verified here |
| Any LiDAR + IMU ROS bag | copy a config | Set `lid_topic`/`imu_topic`, `lidar_type` (1 Livox, 2 Velodyne, 3 Ouster), `scan_line`, `scan_rate`, and the LiDAR↔IMU extrinsic |

There is no upstream Hesai config; why the Velodyne branch is the right home for it, and why the `Failed to find match for field 'time'` warnings are expected, are in [NOTES.md](NOTES.md).
