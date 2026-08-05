# FAST-LIVO2

Tightly-coupled LiDAR-inertial-visual odometry: a single error-state Kalman filter fusing direct sparse image alignment with LiDAR-inertial odometry, producing a colourized map.

- **Repo**: [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)
- **Paper**: [FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/abs/2408.14035) — Zheng et al., IEEE T-RO 2025
- Predecessor: [FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/abs/2203.00893) — IROS 2022

## Build

```bash
podman build -t slam_zero_to_hero:fast_livo2 .
```

Bakes ROS Noetic, a non-templated Sophus, Livox-SDK v1, `rpg_vikit` and FAST-LIVO2 into `/catkin_ws`, plus `rviz` and `compressed_image_transport` for the GUI. A post-build check fails the image if `fastlivo_mapping` is missing.

## Run with the GUI

Fetch the demo sequence first (1.8 GB):

```bash
python3 ../download_fast_livo2.py Retail_Street
```

```bash
mkdir -p results/gui
timeout 1800 podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/fast_livo2:/data:ro \
  -v "$PWD/results/gui":/catkin_ws/src/FAST-LIVO2/Log/result:rw \
  -v "$PWD/results/gui":/out:rw \
  -v "$PWD/config/avia_retail_street.yaml":/catkin_ws/src/FAST-LIVO2/config/avia.yaml:ro \
  -v "$PWD/run_avia.sh":/run.sh:ro \
  -e RVIZ=true \
  slam_zero_to_hero:fast_livo2 bash /run.sh
```

rviz shows the colourized map building up alongside the `/rgb_img` camera view — the clearest way to see this is LiDAR-*visual* odometry and not LIO alone. Drop `-e RVIZ=true` for a headless run.

No `xhost` change and no `--net=host` are needed. Expect one cosmetic red error on rviz's `RobotModel` display (no URDF for this rig).

For the Hilti sequence instead, use `run_hilti.sh` with `~/data/hilti_2022` — FAST-LIVO2 ships a Hesai/Hilti config.

## Supported datasets

| Dataset | Launch / config | Status |
|---|---|---|
| **FAST-LIVO2-Dataset** `Retail_Street` | `mapping_avia.launch` + `config/avia_retail_street.yaml` | ✅ **the demo sequence.** 1351 poses, 67.4 m closed loop, 4 cm end-to-start = 0.06 % drift |
| **Hilti 2022** `exp14_basement_2.bag` | bundled `mapping_hesaixt32_hilti22.launch` | ✅ verified: 738 poses, 37.94 m. Grayscale fisheye, images decimated 4×. |
| 19 more FAST-LIVO2-Dataset sequences | `python3 ../download_fast_livo2.py --list` | ⚠️ only `CBD_Building_01` and `Bright_Screen_Wall` share Retail_Street's calibration — the others need their own block from `calibration.yaml` (`--calib` shows the groups) |
| MARS-LVIG, NTU VIRAL | `mapping_avia_marslvig.launch`, `mapping_ouster_ntu.launch` | Shipped by upstream, not verified here |

Note that `avia.yaml` ships `pose_output_en: false`, so a stock run writes no trajectory — that, the per-sequence calibration trap, and how to generate input for [Global-LVBA](https://github.com/xuankuzcr/Global-LVBA) are in [NOTES.md](NOTES.md).
