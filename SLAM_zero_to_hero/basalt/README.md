# Basalt-VIO

Visual-inertial odometry and mapping for stereo + IMU, using square-root sliding-window optimization. The default dataset is **EuRoC MAV `MH_01_easy`**; a Monado SLAM VR-headset sequence is also baked into the image for the [VR/AR section](#vrar-headsets-monado-slam-baked-into-the-image).

- **Repo**: [mateosss/basalt](https://gitlab.freedesktop.org/mateosss/basalt) (the Monado XR fork, pinned to `a90a57d7`) — of [VladyslavUsenko/basalt](https://gitlab.com/VladyslavUsenko/basalt)
- **Paper**: [Visual-Inertial Mapping with Non-Linear Factor Recovery](https://arxiv.org/abs/1904.06504) — Usenko et al., IEEE RA-L 2020
- Also relevant: [Square Root Marginalization for Sliding-Window Bundle Adjustment](https://arxiv.org/abs/2109.02182) — the marginalization Basalt's VIO actually uses

## Download the data

```bash
python3 ../download_euroc_mav.py    # MH_01_easy by default; --list for the other 10
# -> ~/data/euroc_mav/MH_01_easy/mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}
```

The downloader takes no arguments and deletes each outer zip after extracting it. The ETH server rate-limits (`HTTP 429`); rerun later if a zip fails.

`MH_01_easy` is 3682 stereo pairs (752×480, 20 Hz), 36,820 IMU samples (200 Hz) and Vicon/Leica ground truth, 184 s of data.

## Build

```bash
podman build -t slam_zero_to_hero:basalt .
```

CPU only; no CUDA. The build also bakes in a 4.3 GB Monado SLAM sequence at `/MIPB07_beatsaber_fitbeat_expertplus_2` (used by Final Project 6), which makes the image 22 GB.

## Run with the GUI

```bash
podman run --rm -it \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/euroc_mav/MH_01_easy:/dataset:ro \
  slam_zero_to_hero:basalt \
  basalt_vio --show-gui 1 \
    --dataset-path /dataset \
    --dataset-type euroc \
    --cam-calib /usr/local/share/basalt/euroc_eucm_calib.json \
    --config-path /usr/local/share/basalt/euroc_config.json
```

One Pangolin window (`Main`) opens: the stereo images with tracked features (top left), the 3D trajectory and landmarks (top right), and position plots (bottom). It starts running immediately and stays open at the end; close it to exit.

No `xhost` change and no `--net=host` are needed. The two NVIDIA lines give hardware GL; without them the image falls back to Mesa software rendering, which also works.

![basalt_vio GUI on EuRoC MH_01_easy, host display](results/euroc_mh01_gui_hostdisplay.png)

Above: the command as written, on the host X display with RTX 5090 GL, 75 s in (frame 2682). Green is the estimate, yellow the ground truth, blue dots the landmarks. `results/euroc_mh01_gui.png` is the same view taken headless at t+110 s with the command below (frame 1983 on the rebuilt image under heavy CPU load; software GL slows the estimator, so raise the delay to capture the whole sequence).

## Run headless (trajectory + ATE)

```bash
mkdir -p results
podman run --rm \
  -v ~/data/euroc_mav/MH_01_easy:/dataset:ro \
  -v "$(pwd)/results":/out:rw -w /out \
  slam_zero_to_hero:basalt \
  basalt_vio --show-gui 0 \
    --dataset-path /dataset \
    --dataset-type euroc \
    --cam-calib /usr/local/share/basalt/euroc_eucm_calib.json \
    --config-path /usr/local/share/basalt/euroc_config.json \
    --result-path /out/euroc_mh01_metrics.json \
    --save-trajectory tum \
    --save-trajectory-fn euroc_mh01_traj.txt
```

Writes `results/euroc_mh01_traj.txt` (TUM format) and `results/euroc_mh01_metrics.json` (RMS ATE vs. the EuRoC ground truth, SE(3)-aligned).

Measured on this host (2026-09-28): **3682 / 3682 frames**, **RMS ATE 0.076 m** (3638 poses matched to GT, SE(3)-aligned), estimated path length 80.1 m. Runtime was 22 s on an idle box (8.5× real time, 2026-08-05); on 2026-09-28, with ~19 other jobs sharing the CPU, it was 77–175 s.

To screenshot the GUI with no display at all (Xvfb + software GL inside the container):

```bash
podman run --rm \
  -v ~/data/euroc_mav/MH_01_easy:/dataset:ro \
  -v "$(pwd)/scripts":/scripts:ro -v "$(pwd)/results":/out:rw \
  slam_zero_to_hero:basalt \
  /scripts/capture_gui.sh /out/euroc_mh01_gui.png 110 \
    --dataset-path /dataset --dataset-type euroc \
    --cam-calib /usr/local/share/basalt/euroc_eucm_calib.json \
    --config-path /usr/local/share/basalt/euroc_config.json
```

## Supported datasets

| Dataset | Calib + config | Status |
|---|---|---|
| **EuRoC MAV `MH_01_easy`** (default) | `euroc_eucm_calib.json` + `euroc_config.json` | ✅ 3682 frames, RMS ATE **0.076 m**, GUI screenshot above. Other EuRoC sequences work the same way; also available: `euroc_ds_calib.json` (double sphere), `euroc_rt8_calib.json` (radial-tangential). |
| **Monado SLAM** `MIPB07` (Valve Index, baked in) | `msdmi_calib.json` + `msdmi_config.json` | ✅ 8105 frames, RMS ATE **0.062 m** (see below). `msdmo` = Odyssey+, `msdmg` = the third headset. |
| TUM-VI 512×512 | `tumvi_512_eucm_calib.json` + `tumvi_512_config.json` | Same `--dataset-type euroc`; not run here. |

VO mode (no IMU) is the same binary with `euroc_config_vo.json`, or `--use-imu 0`.

## VR/AR headsets: Monado SLAM (baked into the image)

The image ships `MIPB07_beatsaber_fitbeat_expertplus_2` from the [Monado SLAM Datasets](https://huggingface.co/datasets/collabora/monado-slam-datasets): a Valve Index HMD playing Beat Saber, 8105 stereo frames of 960×960 at 54 Hz plus 1 kHz IMU. No bind mount is needed:

```bash
podman run --rm -it \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  slam_zero_to_hero:basalt \
  basalt_vio --show-gui 1 \
    --dataset-path /MIPB07_beatsaber_fitbeat_expertplus_2 \
    --dataset-type euroc \
    --cam-calib /usr/local/share/basalt/msdmi_calib.json \
    --config-path /usr/local/share/basalt/msdmi_config.json
```

Verified RMS ATE ≈0.062 m against the sequence's own ground truth; the headset stays within a 2.4 × 1.7 × 1.3 m box. Headless command and details in [NOTES.md](NOTES.md).

![basalt_vio GUI on Monado MIPB07](results/msd_beatsaber_gui.png)

Above: headless capture 60 s in (frame 1929 of 8105) — the 960×960 fisheye pair with tracked features and the head trajectory.
