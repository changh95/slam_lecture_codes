# Basalt-VIO

Visual-Inertial Odometry from the [Monado-VIT](https://gitlab.freedesktop.org/mateosss/basalt) fork of [Basalt](https://gitlab.com/VladyslavUsenko/basalt). VIO + VO modes for stereo + IMU.

- **Repo**: https://gitlab.freedesktop.org/mateosss/basalt
- **Sensors**: stereo + IMU (EuRoC, TUM-VI, Monado SLAM datasets)
- **GPU**: not required for VIO; OpenGL needed for `--show-gui 1`

## Build

```bash
podman build -t slam_zero_to_hero:basalt .
```

The image bundles Basalt's stereo-VIO and stereo-VO binaries plus EuRoC, TUM-VI, and Monado-SLAM calibration JSON files installed under `/usr/local/share/basalt/`. EuRoC calibrations available out of the box:

| Camera model | Calib file |
|---|---|
| EUCM (recommended) | `/usr/local/share/basalt/euroc_eucm_calib.json` |
| Double-sphere | `/usr/local/share/basalt/euroc_ds_calib.json` |
| Radial 8-tangent | `/usr/local/share/basalt/euroc_rt8_calib.json` |

## Verified run — EuRoC MH_01_easy (stereo + IMU)

```bash
mkdir -p results
podman run --rm \
  -v ~/data/euroc_mav/MH_01_easy:/dataset:ro \
  -v "$(pwd)/results":/out:rw \
  -w /out \
  slam_zero_to_hero:basalt \
  basalt_vio --show-gui 0 \
    --dataset-path /dataset \
    --dataset-type euroc \
    --cam-calib /usr/local/share/basalt/euroc_eucm_calib.json \
    --config-path /usr/local/share/basalt/euroc_config.json \
    --result-path /out/euroc_mh01_metrics.json \
    --save-trajectory tum \
    --save-trajectory-fn euroc_mh01_traj.txt \
    --marg-data /out/euroc_mh01_marg
```

Output:

| File | Description |
|---|---|
| `results/euroc_mh01_traj.txt` | Trajectory in TUM format (`t tx ty tz qx qy qz qw`) |
| `results/euroc_mh01_metrics.json` | RMS ATE + frame counts vs. EuRoC ground truth |
| `results/euroc_mh01_marg/` | Marginalization data (for downstream BA / re-optimization) |
| `results/stats_{vio,sums,all}.json` | Per-block runtime stats (linearize, QR, solve, marginalize, ...) |

Last verified: Ryzen 9 7950X. **3682 frames in 20.3 s wall (≈ 9× real-time)**, **RMS ATE = 0.073 m** vs. the EuRoC Vicon ground truth — in line with published Basalt-VIO numbers for `MH_01_easy`.

For visualization, drop `--show-gui 0` and add the X11 forwarding flags:

```bash
xhost +local:root
podman run --rm \
  --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/euroc_mav/MH_01_easy:/dataset:ro \
  slam_zero_to_hero:basalt \
  basalt_vio --show-gui 1 \
    --dataset-path /dataset --dataset-type euroc \
    --cam-calib /usr/local/share/basalt/euroc_eucm_calib.json \
    --config-path /usr/local/share/basalt/euroc_config.json
```

## Other configs

- **VO mode (no IMU)**: swap `--config-path` to `/usr/local/share/basalt/euroc_config_vo.json`.
- **TUM-VI**: `--dataset-type euroc --cam-calib /usr/local/share/basalt/tumvi_512_eucm_calib.json --config-path /usr/local/share/basalt/tumvi_512_config.json` against the TUM-VI 512×512 dataset.
- **Monado SLAM datasets**: bundled under `/basalt/data/msd/` inside the image.
