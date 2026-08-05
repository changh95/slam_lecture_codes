# Basalt-VIO

Visual-Inertial Odometry from the [Monado-VIT](https://gitlab.freedesktop.org/mateosss/basalt) fork of [Basalt](https://gitlab.com/VladyslavUsenko/basalt). VIO and VO modes for stereo + IMU.

- **Repo**: https://gitlab.freedesktop.org/mateosss/basalt
- **Sensors**: stereo + IMU (EuRoC, TUM-VI, Monado SLAM datasets)
- **GPU**: not required — `--show-gui 0` needs no display at all; OpenGL is only for `--show-gui 1`

## Build

```bash
podman build -t slam_zero_to_hero:basalt .
```

The image installs one estimator entry point, **`basalt_vio`** (VO is the same binary with a VO config or `--use-imu 0`, not a separate executable), alongside 14 other tools (`basalt_calibrate`, `basalt_kittidata_to_stereo_dataset`, `basalt_convert_kitti_calib.py`, …) and calibration JSONs under `/usr/local/share/basalt/`:

| Dataset / model | Calib file |
|---|---|
| EuRoC, EUCM (recommended) | `/usr/local/share/basalt/euroc_eucm_calib.json` |
| EuRoC, double-sphere | `/usr/local/share/basalt/euroc_ds_calib.json` |
| EuRoC, radial 8-tangent | `/usr/local/share/basalt/euroc_rt8_calib.json` |
| Monado SLAM, Valve Index | `msdmi_calib.json` + `msdmi_config.json` |
| Monado SLAM, Odyssey+ | `msdmo_calib.json` + `msdmo_config.json` |
| Monado SLAM, third headset | `msdmg_calib.json` + `msdmg_config.json` |

The Monado files exist in **both** `/usr/local/share/basalt/` and `/basalt/data/msd/`; prefer the former for consistency with the EuRoC paths.

The build also **bakes in a 4.3 GB Monado SLAM sequence** so the image has an out-of-the-box demo dataset. It unzips at the filesystem *root*, so the real path is `/MIPB07_beatsaber_fitbeat_expertplus_2` (with `mav0/` directly inside) — not under `/basalt/data` or `/datasets`.

## Verified run — Monado SLAM, Valve Index (baked into the image)

`MIPB07_beatsaber_fitbeat_expertplus_2` from the [Monado SLAM Datasets](https://huggingface.co/datasets/collabora/monado-slam-datasets): a Valve Index HMD playing Beat Saber. EuRoC-like layout, 8105 stereo frames of 960×960 grayscale at 54 Hz, 150,170 IMU samples at 1000 Hz, 150.1 s of data. No host bind mount needed.

```bash
mkdir -p results/msd_beatsaber
podman run --rm \
  -v "$(pwd)/results/msd_beatsaber":/out:rw -w /out \
  slam_zero_to_hero:basalt \
  basalt_vio --show-gui 0 \
    --dataset-path /MIPB07_beatsaber_fitbeat_expertplus_2 \
    --dataset-type euroc \
    --cam-calib /usr/local/share/basalt/msdmi_calib.json \
    --config-path /usr/local/share/basalt/msdmi_config.json \
    --result-path /out/msd_beatsaber_metrics.json \
    --save-trajectory tum \
    --save-trajectory-fn msd_beatsaber_traj.txt
```

Use the **`msdmi_*`** pair: `msdmi` = MI_valve_index (kb4 fisheye, 960×960 stereo — matches the baked PNGs exactly). `msdmo` is the Odyssey+ pair shown in the Dockerfile's trailing comment, and that comment also references an `MOO09_short_1_updown` sequence which is **not** in this image.

Last verified: Ryzen 9 7950X, 2026-08-05. **8105 / 8105 frames**, **RMS ATE ≈ 0.062 m** against the sequence's own ground truth (8103 of 8105 poses matched), peak RSS 663 MiB. Trajectory path length 100.55 m inside a 2.42 × 1.65 × 1.25 m box — it is a person standing and swinging their arms, so the whole trajectory stays within ~2.2 m of the origin.

Two runs of the exact command above gave RMS ATE 0.0618 and 0.0630 m, with internal runtimes of 69.6 s (idle box) and 76.3 s (five other SLAM containers running) — i.e. **106–116 frames/s, ~2.0–2.2× real time** over 150.1 s of data. See the note on `--deterministic 0` under the EuRoC run.

Tracking is not perfectly continuous: 7 of 8101 recorded frames have zero landmarks *and* zero observations (brief windows with no visual observations at all), plus one with no camera. That is expected for fast HMD motion and does not break the run.

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
    --save-trajectory-fn euroc_mh01_traj.txt
```

Last verified: Ryzen 9 7950X, 2026-08-05. **3682 frames** (matching the 3682 PNGs in the host's `cam0`), internal runtime **21.55 s** (wall 22.33 s) = **8.5× real time**, peak RSS 337 MiB, **RMS ATE ≈ 0.074 m** against the EuRoC Vicon ground truth — in line with published Basalt-VIO figures for `MH_01_easy`.

Quote the ATE to two significant figures. `basalt_vio` defaults to `--deterministic 0` with an automatic thread count, so the value moves between runs: 0.0732 in an earlier session, 0.0743 here.

Output:

| File | Description |
|---|---|
| `<name>_traj.txt` | Trajectory in TUM format (`t tx ty tz qx qy qz qw`), one header line + one line per frame |
| `<name>_metrics.json` | RMS ATE + frame counts vs. ground truth |
| `stats_{vio,sums,all}.json` | Per-block runtime stats (linearize, QR, solve, marginalize, …). `stats_all.json` reaches ~6.7 MB. |

### `--marg-data` is optional and expensive

The marginalization dump (`--marg-data /out/<name>_marg`) is only useful for downstream BA or re-optimization. It costs real time and disk: **663 MiB / 918 `.cereal` files** for EuRoC MH_01, and **4.1 GB** for the 8105-frame Monado sequence. Omitting it cut the Monado run from 85.1 s to 69.6 s — **18 % faster**. Both verified runs above leave it out deliberately.

## Watching it run (GUI on your desktop)

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

That runs the baked-in Monado sequence, so it needs no bind-mounted dataset at all. For EuRoC instead, add `-v ~/data/euroc_mav/MH_01_easy:/dataset:ro` and point `--dataset-path` at `/dataset` with the `euroc_eucm_calib.json` / `euroc_config.json` pair.

One Pangolin window titled **`Main`** opens: a `show_frame` slider, the stereo images with tracked features, and the 3D trajectory and landmark view, plus toggles for the feature and highlight menus.

Three notes on the flags, each verified on this host:

- **No `xhost +local:root`** — unnecessary (rootless podman already authenticates as your uid) and a needless security downgrade. Earlier revisions of this file said to run it.
- **No `--net=host`** — the X connection goes over the bind-mounted socket.
- **The `--runtime` and `NVIDIA_*` lines matter.** Without them this image renders through Mesa `swrast` software rasterization even though it is built `FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04`; with them you get `NVIDIA GeForce RTX 5090` and OpenGL 4.6. podman 3.4 predates CDI, so `--device nvidia.com/gpu=all` does not work here.

To confirm the window mapped, use `xwininfo -root -tree | grep Main` — `-root -children` misses it because the window manager reparents it.

## Other configs

- **VO mode (no IMU)**: swap `--config-path` to `/usr/local/share/basalt/euroc_config_vo.json`.
- **TUM-VI**: `--dataset-type euroc --cam-calib /usr/local/share/basalt/tumvi_512_eucm_calib.json --config-path /usr/local/share/basalt/tumvi_512_config.json` against the TUM-VI 512×512 dataset.
- **Other Monado sequences**: same `msdmi_*` configs; download further sequences from the Monado SLAM Datasets and bind-mount them.
