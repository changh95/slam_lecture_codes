# Basalt-VIO

Visual-inertial odometry and mapping for stereo + IMU, using square-root sliding-window optimization.

- **Repo**: [mateosss/basalt](https://gitlab.freedesktop.org/mateosss/basalt) (the Monado XR fork) — of [VladyslavUsenko/basalt](https://gitlab.com/VladyslavUsenko/basalt)
- **Paper**: [Visual-Inertial Mapping with Non-Linear Factor Recovery](https://arxiv.org/abs/1904.06504) — Usenko et al., IEEE RA-L 2020
- Also relevant: [Square Root Marginalization for Sliding-Window Bundle Adjustment](https://arxiv.org/abs/2109.02182) — the marginalization Basalt's VIO actually uses

## Build

```bash
podman build -t slam_zero_to_hero:basalt .
```

The build also bakes in a 4.3 GB Monado SLAM sequence at `/MIPB07_beatsaber_fitbeat_expertplus_2`, so the image is runnable with no dataset mounted.

## Run with the GUI

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

One Pangolin window (`Main`) opens: a frame slider, the stereo images with tracked features, and the 3D trajectory and landmarks.

No `xhost` change and no `--net=host` are needed. The two NVIDIA lines give hardware GL; without them this image falls back to software rendering. For a headless run use `--show-gui 0` with `--save-trajectory tum`.

## Supported datasets

| Dataset | Calib + config | Notes |
|---|---|---|
| **Monado SLAM** (verified: `MIPB07`, baked in) | `msdmi_calib.json` + `msdmi_config.json` | `msdmi` = Valve Index. `msdmo` = Odyssey+, `msdmg` = the third headset. |
| **EuRoC MAV** (verified: `MH_01_easy`) | `euroc_eucm_calib.json` + `euroc_config.json` | Mount with `-v ~/data/euroc_mav/MH_01_easy:/dataset:ro`. Also available: `euroc_ds_calib.json` (double sphere), `euroc_rt8_calib.json` (radial-tangential). |
| **TUM-VI** 512×512 | `tumvi_512_eucm_calib.json` + `tumvi_512_config.json` | Same `--dataset-type euroc`. |

VO mode (no IMU) is the same binary with `euroc_config_vo.json`, or `--use-imu 0`.

Verified accuracy: **RMS ATE ≈0.062 m** on Monado `MIPB07` (8105 frames), **≈0.074 m** on EuRoC `MH_01_easy`. Details in [NOTES.md](NOTES.md).
