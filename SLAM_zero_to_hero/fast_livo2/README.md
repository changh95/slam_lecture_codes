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

## Verified run — FAST-LIVO2-Dataset `Retail_Street` (the demo sequence)

**`Retail_Street` is the sequence to use for this course's FAST-LIVO2 demo.** Everything below runs on it with no calibration work, and it is the only sequence needed — the rest of the dataset is documented further down as reference.

This is the dataset FAST-LIVO2 was built around: a handheld **Livox Avia** (LiDAR + built-in IMU) plus a proper **RGB pinhole camera**, which exercises the visual half far better than Hilti's grayscale fisheye. Fetch just that one sequence with the downloader at the repo root:

```bash
python3 ../download_fast_livo2.py Retail_Street     # 1.8 GB, resumable
```

```bash
mkdir -p results/avia
timeout 1800 podman run --rm \
  -v ~/data/fast_livo2:/data:ro \
  -v "$(pwd)/results/avia":/catkin_ws/src/FAST-LIVO2/Log/result:rw \
  -v "$(pwd)/results/avia":/out:rw \
  -v "$(pwd)/config/avia_retail_street.yaml":/catkin_ws/src/FAST-LIVO2/config/avia.yaml:ro \
  -v "$(pwd)/run_avia.sh":/run.sh:ro \
  slam_zero_to_hero:fast_livo2 \
  bash /run.sh
```

Bag contents (135 s): `/livox/lidar` 1355 × `livox_ros_driver/CustomMsg`, `/livox/imu` 27,447 msgs (~203 Hz), `/left_camera/image` 1355 raw `sensor_msgs/Image` — matching `avia.yaml`'s topic names exactly.

Last verified: Ryzen 9 7950X, 2026-08-05.

| Measured | Value |
|---|---|
| Poses | **1351** of 1355 scans (4 go to IMU/VIO init) |
| Pose rate | 10.0155 Hz over a 135.000 s span |
| Path length | **67.417 m** |
| Start → end | **0.040 m** — this sequence *is* a closed loop, so 4 cm over 67 m ≈ **0.06 % drift** |
| Extent | x [−17.70, 1.36], y [−19.21, 3.10], z [−0.00, 1.08] m (a planar ~19 × 22 m walk) |
| LIO cost | `Average Total Time` **14.12 ms**/frame |
| VIO cost | `Average Total Time` **4.71 ms**/frame |
| Quaternion norms | 1.000000 ± 2.8e-07, zero NaNs |

Because it returns to its start, this is the sequence to use when you want a drift number out of pure odometry — no ground truth or loop-closure module needed. Contrast Hilti below, where start and end are 21.4 m apart.

The visual pipeline is fully engaged here: **1355 of 1355 images accepted** (no decimation — unlike the Hilti config's `hilti_en` 4× drop) and 1349 `Retrieve N points from visual sparse map` steps.

Two things to know about this configuration:

- **`config/avia_retail_street.yaml` exists only because upstream's `avia.yaml` sets `evo.pose_output_en: false`.** A stock `mapping_avia.launch` run writes **no trajectory at all**. That file is upstream's `avia.yaml` with exactly two changes: `pose_output_en: true` and `seq_name: "Retail_Street"`.
- **`rviz` defaults to `true`** in `mapping_avia.launch`, so `rviz:=false` is mandatory for a headless run. `run_avia.sh` handles that, plus the SIGINT-then-wait needed for FAST-LIVO2 to flush its trajectory as `main()` unwinds.

### The timestamps really do say the year 2000

The output trajectory starts at `946685437.899828` (2000-01-01), while `rosbag info` reports `start: Sep 27 2022`. Nothing is broken: the message header stamps *in the bag* are the Livox device clock, which was never wall-synced —

```
/livox/lidar[0]  header.stamp = 946685437.499511
                 timebase     = 946685277499511000  ->  946685277.499511
```

— whereas `rosbag info`'s "start" is the bag *record* time. FAST-LIVO2 faithfully echoes the header stamps. Align by index, or fix the clock, if you need to compare against anything external.

## Verified run — Hilti 2022 `exp14_basement_2.bag`

FAST-LIVO2 **already ships** a Hilti-2022 / Hesai-PandarXT-32 config — `mapping_hesaixt32_hilti22.launch` + `HILTI22.yaml` + `camera_fisheye_HILTI22.yaml`. Topics, extrinsics, IMU noise, and camera intrinsics are all pre-tuned for this dataset, so nothing in this directory needs to override them.

```bash
mkdir -p results
chmod +x run_hilti.sh
podman run --rm \
  -v ~/data/hilti_2022:/data:ro \
  -v "$(pwd)/results":/catkin_ws/src/FAST-LIVO2/Log/result:rw \
  -v "$(pwd)/results":/out:rw \
  -v "$(pwd)/run_hilti.sh":/run.sh:ro \
  slam_zero_to_hero:fast_livo2 \
  bash /run.sh /data/exp14_basement_2.bag
```

`--net=host` is **not** needed (earlier revisions of this file mandated it). The container starts its own roscore inside its private network namespace, verified by running this alongside other ROS containers with no port-11311 contention. Drop it so parallel runs stay isolated.

The script starts roscore, launches `mapping_hesaixt32_hilti22.launch` headless, plays the bag with `--clock`, then sends SIGINT so FAST-LIVO2 flushes its trajectory to disk.

Outputs land in `results/`:

| File | Description |
|---|---|
| `results/exp09_cupola.txt` | Trajectory in TUM format (`t tx ty tz qx qy qz qw`), 738 lines / 62,445 B. The name comes from `evo.seq_name` baked into `HILTI22.yaml` — rename it after the run if you process several bags. |
| `results/fastlivo.log` | Full stdout/stderr from `fastlivo_mapping` (LIO + VIO state messages, ~2.2 MB) |

Last verified: Ryzen 9 7950X, 2026-08-05, whole container 107 s wall (74 s bag + roscore, launch settle, and the post-SIGINT flush).

| Measured | Value |
|---|---|
| Poses | 738 (of 740 Hesai sweeps; the first two go to IMU init) |
| Pose rate | ~10.0011 Hz — dt min 0.099984 s, max 0.099995 s. One pose per LiDAR sweep, but **not** bit-uniform. |
| Timestamps | 1649764528.346119 → 1649764602.038165 (span 73.692 s) |
| Path length | 37.94 m (37.9414 / 37.9404 m over two runs) |
| Start → end | 21.39–21.40 m — this leg is **not a closed loop** |
| Extent | x [−0.907, 5.196], y [−21.394, 0.001], z [−1.008, 3.361] m |
| LIO cost | `Average Total Time` 19.34 ms/frame (final log block) |
| VIO cost | `Average Total Time` 4.25 ms/frame |

The 37.94 m path length independently corroborates FAST-LIO2's 37.93 m on the same bag from a separate estimator (`../fast_lio2/README.md`), and both are backed by a voxel-sharpness check against the raw scans.

Two honest caveats. The trajectory spans **4.37 m in z** for a basement walk; without ground truth (Hilti withheld it for `exp14_basement_2`) there is no way to separate genuine stairs/ramp from vertical drift, so publish no ATE for this sequence. And the run is not bit-reproducible: two runs gave the same 738 lines, same first line, and same final timestamp, but final positions differing by ~3 cm.

### The visual half really does engage

Worth checking, because a LIVO system silently degrading to LIO-only looks identical in the trajectory file. In `results/fastlivo.log`:

- 740 `Get image` lines — FAST-LIVO2 uses **only `/alphasense/cam0`**, and `hilti_en: true` decimates the bag's 40 Hz images to 10 Hz (`if (++frame_counter % 4 != 0) return;`), so 2960 published images become 740 accepted ones. Seeing 740 rather than 2960 is correct, not dropped frames.
- 736 `Retrieve N points from visual sparse map` steps, the last few retrieving 60–66 points.
- `IMU Initializing: 3.3 % … 86.7 %` then `IMU Initials: Gravity: -0.1767 -0.1664 9.8070`, with `[ LIO ]: No point!!!` / `[ VIO ] No point!!!` for the two frames consumed by initialization.

### Gotcha in `run_hilti.sh`

Its second positional argument (`SEQ_NAME`) is a **silent no-op**. The script passes `evo/seq_name:=...` to `roslaunch`, but that launch file declares no such arg, so roslaunch treats it as an undeclared parameter and the name baked into `HILTI22.yaml` wins. That is why the output is always `exp09_cupola.txt` regardless of what you pass.

## The FAST-LIVO2-Dataset — and its per-sequence calibration trap

20 Livox Avia sequences, now on Google Drive with stable file IDs. FAST-LIVO2's own README redirects to the [Global-LVBA](https://github.com/xuankuzcr/Global-LVBA) repository (Section IV) for them, because the older HKU SharePoint links rotate.

```bash
python3 ../download_fast_livo2.py --list      # all 20 sequences with sizes
python3 ../download_fast_livo2.py --calib     # which share a calibration block
python3 ../download_fast_livo2.py --popular   # Retail_Street + Red_Sculpture + CBD_Building_01
```

The downloader resumes interrupted transfers and checks the `#ROSBAG V2.0` magic bytes, because a silently truncated multi-GB download otherwise only shows up as a mysteriously bad SLAM run. (`download_fast_livo2_dataset.sh` is now a deprecation shim forwarding to it.)

### No Drive-specific client needed — it's plain curl/wget

`https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t` is an ordinary HTTPS GET that answers `206 Partial Content` with a `Content-Range`, so standard tools handle these files including resume. The script picks **curl → wget → stdlib urllib**, so it has no third-party Python dependencies; override with `--backend`. All three were verified to produce byte-identical output and to resume correctly from a 100 MB truncation up to the exact expected length.

Doing it by hand is a one-liner:

```bash
ID=1DKOfH8pfObRenoWf4-IfkTMHcliXnigD          # Retail_Street.bag
U="https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t"

curl -fL --retry 5 --retry-delay 2 -C - -o Retail_Street.bag "$U"   # resumes
wget -c --tries=5 --timeout=60 -O Retail_Street.bag "$U"            # also resumes
```

Two traps worth knowing:

- **`curl` without `-f` is dangerous here.** On an error Drive returns an HTML page with HTTP 4xx; plain `curl -sSL` writes that 1652-byte page to your `.bag` **and exits 0**. With `-f` it exits 22 and writes nothing. `wget` exits 8 but leaves a 0-byte file. Either way, verify the length afterwards — which is why the script always re-checks the exact byte count and the bag magic regardless of backend.
- **`gdown` does not work for this folder at all**, failing with *"Cannot retrieve the public link of the file… or have had many accesses"* even though the files are public and curl fetches them fine.

Note that `curl` is **not** installed in the `kiss_slam` or `glim` images, though `wget` is present in all of them — so if you download from inside a container, the fallback chain matters.

**Calibration is per sequence.** The dataset's `calibration.yaml` opens with: *"we recalibrate the device parameters every three months, so different sequences have different intrinsic and extrinsic parameters."* It contains **four** distinct `Rcl`/`Pcl` + intrinsics blocks, and FAST-LIVO2's shipped `avia.yaml` + `camera_pinhole.yaml` encode only the first:

| Group | Sequences | Stock config |
|---|---|---|
| 1 | `Retail_Street`, `CBD_Building_01`, `Bright_Screen_Wall` | ✅ correct as shipped (fx 1293.56944, cx 626.91359) |
| 2 | `HKU_Landmark`, `HKU_Centennial_Garden_01/02`, `HKU_Main_Building`, `HKU_Cultural_Center_01/02`, `HKU_Lecture_Center_01/02`, `CBD_Building_02/03` | ❌ fx 1176.287…, different `Rcl`/`Pcl` |
| 3 | `HIT_Graffiti_Wall_01`–`04`, `SYSU_01/02` | ❌ fx 1311.895… |
| 4 | `Red_Sculpture` | ❌ fx 1294.727…, `Pcl [-0.00077, 0.04809, -0.00133]` |

Running a group 2–4 sequence against the stock config will quietly use the wrong camera–LiDAR extrinsic and intrinsics — it *runs*, it just degrades the visual half. Copy the matching block out of `calibration.yaml` first. `--list` tags which sequences are safe; the downloader warns after fetching one that isn't.

Sizes worth knowing: `Bright_Screen_Wall` is only 360 MB (a good smoke test), `Retail_Street` 1.8 GB, and `HIT_Graffiti_Wall_02` is 22 GB. All 20 total ~150 GB.

Other launch files in the image target hardware not in this dataset: `mapping_avia_marslvig.launch` (MARS-LVIG), `mapping_ouster_ntu.launch` (NTU VIRAL).

## Generating input for Global-LVBA

[Global-LVBA](https://github.com/xuankuzcr/Global-LVBA) is a **post-processing** stage that refines FAST-LIVO2's output — reducing point-cloud layering and pushing camera poses to pixel-level reprojection accuracy. Its *LVBA-Dataset* is therefore FAST-LIVO2's **output**, not an input you can feed back in: per-frame PNG + PCD + pose text files, no rosbag and no raw IMU.

FAST-LIVO2 emits exactly that layout. Set, in the sequence's config:

```yaml
pcd_save:
  pcd_save_en: true      # NOT false -- see below
  type: 1                # body frame; also forces one .pcd per scan
image_save:
  img_save_en: true
  interval: 1
```

which produces, under `Log/`:

```
image/<timestamp>.png   +  image/image_poses.txt     # t x y z qx qy qz qw
pcd/<timestamp>.pcd     +  pcd/lidar_poses.txt
```

**Global-LVBA's README says to set `pcd_save_en: false`, which is wrong.** Verified on 15 s of the Hilti bag:

| Config | PNG | `image_poses.txt` | PCD | `lidar_poses.txt` |
|---|---|---|---|---|
| `pcd_save_en: false` (as documented upstream) | 148 | 148 | **0** | **missing** |
| `pcd_save_en: true` | 148 | 148 | 148 | 148 |

`LIVMapper.cpp:1203` wraps the entire per-frame PCD *and* `lidar_poses.txt` writing in `if (pcd_save_en)`; `type: 1` only forces `pcd_save_interval = 1` **inside** that block. With it false you get images and camera poses but no geometry — half the required structure.

Budget roughly **250 MB per 15 s** of bag (73 MB images + 181 MB point clouds), so ~1.2 GB for the full 74 s Hilti sequence. Bind-mount the whole `Log/` directory to capture it, and pre-create `image/`, `pcd/`, `result/` inside your host directory — the code opens those paths without creating them, and a missing directory fails silently.
