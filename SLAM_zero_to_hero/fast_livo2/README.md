# FAST-LIVO2

Tightly-coupled LiDAR-inertial-visual odometry: a single error-state Kalman filter fusing direct sparse image alignment with LiDAR-inertial odometry, producing a colourized map. The default demo runs on **FAST-LIVO2-Dataset `Retail_Street.bag`** (handheld Livox Avia + built-in IMU + RGB camera, 135 s).

- **Repo**: [hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)
- **Paper**: [FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/abs/2408.14035) — Zheng et al., IEEE T-RO 2025
- Predecessor: [FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/abs/2203.00893) — IROS 2022

![FAST-LIVO2 on Retail_Street: colourized map, path (cyan) and the tracked camera view](results/retail_street_rviz.png)

## Build

```bash
podman build -t slam_zero_to_hero:fast_livo2 .
```

Bakes ROS Noetic, a non-templated Sophus, Livox-SDK v1, `rpg_vikit` and FAST-LIVO2 (all pinned to commits) into `/catkin_ws`, plus `rviz`, `compressed_image_transport` and Xvfb/ImageMagick for GUI and headless screenshots. CPU only, no GPU needed. A post-build check fails the image if `fastlivo_mapping` is missing.

## Data

```bash
python3 ../download_fast_livo2.py Retail_Street     # 1.8 GB into ~/data/fast_livo2, resumable
```

## Run (headless)

```bash
mkdir -p results/avia
timeout 1800 podman run --rm \
  -v ~/data/fast_livo2:/data:ro \
  -v "$PWD/results/avia":/catkin_ws/src/FAST-LIVO2/Log/result:rw \
  -v "$PWD/results/avia":/out:rw \
  -v "$PWD/config/avia_retail_street.yaml":/catkin_ws/src/FAST-LIVO2/config/avia.yaml:ro \
  -v "$PWD/run_avia.sh":/run.sh:ro \
  slam_zero_to_hero:fast_livo2 bash /run.sh
```

`run_avia.sh` starts roscore, launches `mapping_avia.launch rviz:=false`, plays the bag at 1×, then SIGINTs FAST-LIVO2 so it flushes `results/avia/Retail_Street.txt` (TUM format). About 3 min wall-clock.

## Visualization

**Screenshot, no host display** — rviz renders on Xvfb inside the container; `capture_rviz.sh` grabs it at the given seconds of playback (writes `results/gui/rviz_t<sec>.png`; the image at the top is the t = 132 s frame). `config/fast_livo2_overview.rviz` is upstream's rviz config with only the camera changed to a view over the whole walk:

```bash
mkdir -p results/gui
timeout 1800 podman run --rm \
  -v ~/data/fast_livo2:/data:ro \
  -v "$PWD/results/gui":/catkin_ws/src/FAST-LIVO2/Log/result:rw \
  -v "$PWD/results/gui":/out:rw \
  -v "$PWD/config/avia_retail_street.yaml":/catkin_ws/src/FAST-LIVO2/config/avia.yaml:ro \
  -v "$PWD/config/fast_livo2_overview.rviz":/catkin_ws/src/FAST-LIVO2/rviz_cfg/fast_livo2.rviz:ro \
  -v "$PWD/run_avia.sh":/run.sh:ro \
  -v "$PWD/capture_rviz.sh":/capture.sh:ro \
  slam_zero_to_hero:fast_livo2 bash /capture.sh 60 132
```

**Live rviz on your desktop** — same run with X11 + GPU flags and `RVIZ=true`:

```bash
mkdir -p results/gui_host
timeout 1800 podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root -e DISABLE_ROS1_EOL_WARNINGS=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/fast_livo2:/data:ro \
  -v "$PWD/results/gui_host":/catkin_ws/src/FAST-LIVO2/Log/result:rw \
  -v "$PWD/results/gui_host":/out:rw \
  -v "$PWD/config/avia_retail_street.yaml":/catkin_ws/src/FAST-LIVO2/config/avia.yaml:ro \
  -v "$PWD/config/fast_livo2_overview.rviz":/catkin_ws/src/FAST-LIVO2/rviz_cfg/fast_livo2.rviz:ro \
  -v "$PWD/run_avia.sh":/run.sh:ro \
  -e RVIZ=true \
  slam_zero_to_hero:fast_livo2 bash /run.sh
```

rviz shows the RGB-coloured map (`/cloud_registered`, accumulated), the current scan in red, the path (`/path`, cyan) and the `/rgb_img` camera view with tracked visual patches — the clearest sign this is LiDAR-*visual* odometry and not LIO alone. On the RTX 5090 host this renders at ~31 fps ([`results/gui_host/rviz_host.png`](results/gui_host/rviz_host.png)). No `xhost` change and no `--net=host` are needed. Drop the `fast_livo2_overview.rviz` mount for upstream's drone-following close-up view.

**Mouse controls** (course-wide, from [`../rviz_unified_controls`](../rviz_unified_controls)): left drag rotates, wheel zooms, right drag (or middle drag) pans. The image installs the plugin and switches `config/fast_livo2_overview.rviz` to `UnifiedOrbit` and upstream's `rviz_cfg/{fast_livo2,hilti,M300,ntu_viral}.rviz` to `UnifiedThirdPersonFollower`. In the follower views, start a pan with the pointer on the ground, not the sky: like stock rviz, that controller pans by dragging a point on the ground plane.

## Results (Retail_Street, measured 2026-09-27)

| Measured | Value |
|---|---|
| Poses | **1351** of 1355 scans, 10.0 Hz over 135.0 s |
| Path length | **67.43 m** (out-and-back along the street, farthest point 26.1 m from start) |
| Start → end | **0.040 m** — the walk returns to its start, so ≈ **0.06 % drift** with no GT or loop closure |
| Images used | 1355 / 1355, 1349 VIO updates from the visual sparse map |
| Per-frame cost | LIO 36.6 ms, VIO 7.8 ms average (32-core host shared with other jobs; 14.1 / 4.7 ms on an idle Ryzen 9 7950X) |

No ground truth exists for this dataset, so there is no ATE. Three runs this session (headless, Xvfb capture, desktop GUI) gave 67.43 / 67.46 / 67.48 m, all 1351 poses and 4.0–4.1 cm end-to-start.

## Supported datasets

| Dataset | Launch / config | Status |
|---|---|---|
| **FAST-LIVO2-Dataset** `Retail_Street` (default) | `mapping_avia.launch` + `config/avia_retail_street.yaml` | ✅ 1351 poses, 67.43 m, 4 cm end-to-start (0.06 %); rviz screenshot above |
| **Hilti 2022** `exp14_basement_2.bag` | bundled `mapping_hesaixt32_hilti22.launch`, `run_hilti.sh` + `~/data/hilti_2022` | ✅ verified earlier: 738 poses, 37.94 m. Grayscale fisheye, images decimated 4×. |
| 19 more FAST-LIVO2-Dataset sequences | `python3 ../download_fast_livo2.py --list` | ⚠️ only `CBD_Building_01` and `Bright_Screen_Wall` share Retail_Street's calibration — the others need their own block from `calibration.yaml` (`--calib` shows the groups) |
| MARS-LVIG, NTU VIRAL | `mapping_avia_marslvig.launch`, `mapping_ouster_ntu.launch` | Shipped by upstream, not verified here |

Note that upstream `avia.yaml` ships `pose_output_en: false`, so a stock run writes no trajectory — that, the per-sequence calibration trap, and how to generate input for [Global-LVBA](https://github.com/xuankuzcr/Global-LVBA) are in [NOTES.md](NOTES.md).
