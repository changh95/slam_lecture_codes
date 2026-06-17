# FAST-LIVO2

Tightly-coupled LiDAR–Inertial–Visual Odometry from HKU-MARS.

- **Repo**: https://github.com/hku-mars/FAST-LIVO2
- **Sensors**: Livox / Velodyne / Hesai LiDAR + IMU + Camera
- **GPU**: not required (pure CPU)

## Build

```bash
podman build -t slam_zero_to_hero:fast_livo2 .
```

The image bakes:

| Component | Pin |
|---|---|
| Base | `ros:noetic` |
| Sophus | `strasdat/Sophus@a621ff` (non-templated; patched for `complex.real()` + adds `SophusConfig.cmake`) |
| Livox-SDK (v1) | upstream HEAD |
| `livox_ros_driver` (v1) | upstream HEAD — **not** SDK2/driver2; FAST-LIVO2 still expects v1 |
| `rpg_vikit` (FAST-LIVO2 fork) | `xuankuzcr/rpg_vikit` HEAD |
| `FAST-LIVO2` | `hku-mars/FAST-LIVO2` HEAD |

A post-build smoke test (`rospack find fast_livo` + `test -x fastlivo_mapping`) fails the image if the entrypoint binary is missing, so `podman build` succeeding implies the runtime exists.

## Verified run — Hilti 2022 `exp14_basement_2.bag`

FAST-LIVO2 **already ships** a Hilti-2022/Hesai-PandarXT-32 config — `mapping_hesaixt32_hilti22.launch` + `HILTI22.yaml` + `camera_fisheye_HILTI22.yaml`. Topics, extrinsics, IMU noise, and camera intrinsics are all pre-tuned for this dataset.

```bash
mkdir -p results
chmod +x run_hilti.sh
podman run --rm \
  --net=host --ipc=host \
  -v ~/data/hilti_2022:/data:ro \
  -v "$(pwd)/results":/catkin_ws/src/FAST-LIVO2/Log/result:rw \
  -v "$(pwd)/results":/out:rw \
  -v "$(pwd)/run_hilti.sh":/run.sh:ro \
  slam_zero_to_hero:fast_livo2 \
  bash /run.sh /data/exp14_basement_2.bag
```

The script starts roscore, launches `mapping_hesaixt32_hilti22.launch` headless, plays the bag with `--clock`, then sends SIGINT so FAST-LIVO2 flushes its trajectory to disk.

Outputs land in `results/`:

| File | Description |
|---|---|
| `results/exp09_cupola.txt` | Trajectory in TUM format (`t tx ty tz qx qy qz qw`). Filename comes from `evo.seq_name` baked into `HILTI22.yaml`; rename it after the run if you process multiple bags. |
| `results/fastlivo.log` | Full stdout/stderr from `fastlivo_mapping` (LIO + VIO state messages, ~2 MB) |

Last verified: 738 trajectory poses captured on a Ryzen 9 7950X, timestamps 1649764528.34 → 1649764601.x — covers the 74-second basement loop end-to-end.

## Other reference datasets

The HKU-MARS team distributes additional FAST-LIVO2 reference bags via OneDrive (Livox Avia hardware). Links rotate, so this repo doesn't bake the URL — fetch the bag manually and bind-mount it:

| Bag | Sensors | Launch |
|---|---|---|
| `Retail_Street.bag` | Livox Avia + pinhole cam + IMU | `mapping_avia.launch` |
| `HKUST_Red_Sculpture.bag` | Livox Avia + pinhole cam + IMU | `mapping_avia.launch` |
| `MARS_LVIG_*` | Livox + cam + IMU | `mapping_avia_marslvig.launch` |
| NTU VIRAL | Ouster + cam + IMU | `mapping_ouster_ntu.launch` |

```bash
./download_fast_livo2_dataset.sh Retail_Street
```

(prompts for the current OneDrive direct-download URL).
