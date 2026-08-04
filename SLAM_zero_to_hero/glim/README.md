# GLIM

Versatile range-based SLAM built on GTSAM factor graphs: fixed-lag smoothing odometry, submap-based local mapping, and global factor-graph optimization.

- **Repo**: https://github.com/koide3/glim (`v1.0.0`)
- **Sensors**: 3D LiDAR, optionally + IMU (LiDAR-only via the CT-ICP odometry module)
- **GPU**: not required — this image builds with `BUILD_WITH_CUDA=OFF`

## Build

```bash
podman build -t slam_zero_to_hero:glim .
```

## Upstream ships no executable — read this first

A plain (non-ROS) glim build produces **only shared libraries**. There is no `add_executable` and no `main()` anywhere in glim v1.0.0, so `/usr/local/bin` comes out empty and the image cannot read a dataset at all. Upstream's data-ingesting binaries (`glim_rosnode`, `glim_rosbag`, `offline_viewer`) live in the separate [`koide3/glim_ros1`](https://github.com/koide3/glim_ros1) package, which needs ROS 1 Noetic and therefore Ubuntu 20.04 — impossible on this 24.04 CUDA base.

This directory therefore adds `kitti/glim_kitti.cpp`, a ~250-line driver that feeds KITTI velodyne `.bin` scans through the **real** GLIM pipeline, mirroring `GlimROS::insert_frame()`:

```
read .bin -> glim::TimeKeeper -> glim::CloudPreprocessor -> AsyncOdometryEstimation
          -> AsyncSubMapping -> AsyncGlobalMapping -> GlobalMapping::save()
```

The odometry, sub-mapping, and global-mapping modules are `dlopen`-ed through glim's own `load_module()` factory exactly as upstream does, so behaviour matches a stock `glim_rosbag` run. The Dockerfile compiles it to `/usr/local/bin/glim_kitti` and installs two config sets, so the image is self-contained:

| Path | Global mapping module | Use |
|---|---|---|
| `/usr/local/share/glim_kitti/config` | `libglobal_mapping.so` (VGICP matching-cost factors) | Default. Verified on seq 04 and seq 00. |
| `/usr/local/share/glim_kitti/config_posegraph` | `libglobal_mapping_pose_graph.so` (explicit loop detection) | Larger global corrections; best seq 00 ATE. |

```
glim_kitti <config_dir> <kitti_sequence_dir> <dump_dir> [max_scans] [stamp_offset]
```

## Verified run — KITTI odometry seq 04 (271 scans)

```bash
mkdir -p results
timeout 900 podman run --rm --network none \
  -v ~/data/kitti_vo_slam/extracted/dataset/sequences/04:/data/seq04:ro \
  -v "$PWD/results":/output \
  slam_zero_to_hero:glim \
  glim_kitti /usr/local/share/glim_kitti/config /data/seq04 /output/dump_seq04
```

No GPU, no X11, no ROS, no network (`--network none`) — GLIM's viewers are separate `dlopen`-ed modules that this driver simply never loads, so there is no headless flag to pass.

Evaluate against ground truth:

```bash
D=~/data/kitti_vo_slam/extracted/dataset
python3 kitti/eval_kitti.py results/dump_seq04 $D/sequences/04 $D/poses/04.txt
```

| | poses | path | ATE mean | ATE rmse | ATE max |
|---|---|---|---|---|---|
| ground truth | 271 | 393.65 m | — | — | — |
| `odom_lidar.txt` | 261 | 376.61 m | **2.60 m** | 3.57 m | 8.41 m |
| `traj_lidar.txt` | 261 | 376.61 m | 2.66 m | 3.65 m | 8.63 m |

Last verified: Ryzen 9 7950X, 2026-08-05. 18 submaps, 21 VGICP factors, **31 scans/s** on an idle box (17 scans/s measured while five other SLAM containers were running). ATE is 0.66 % of path length.

The ATE is deliberately strict: `eval_kitti.py` transforms ground truth into the Velodyne frame with `Tr` from `calib.txt`, re-anchors it to frame 0, and compares **without** any Umeyama alignment — so start-of-sequence yaw error propagates into every later pose. Treat it as an upper bound.

GLIM emits 261 of 271 poses: the newest frames are still inside the fixed-lag smoother and never get marginalized out. Expect `n < scans`, not a bug.

## Verified run — KITTI odometry seq 00 (4541 scans)

Seq 04 is a straight 394 m drive with no revisits, so it cannot exercise global optimization. Seq 00 is 3.7 km with heavy revisits:

```bash
timeout 2400 podman run --rm --network none \
  -v ~/data/kitti_vo_slam/extracted/dataset/sequences/00:/data/seq00:ro \
  -v "$PWD/results":/output \
  slam_zero_to_hero:glim \
  glim_kitti /usr/local/share/glim_kitti/config /data/seq00 /output/dump_seq00
```

| Global mapping module | poses | path | ATE mean (odom → traj) |
|---|---|---|---|
| `libglobal_mapping.so` (default config) | 4531 | 3708.1–3708.2 m | 11.9–12.0 → **11.5–11.6 m** |
| `libglobal_mapping_pose_graph.so` (`config_posegraph`) | 4531 | 3708.2 m | 11.9 → **10.2 m** |

Both rows are two independent runs each. Ground truth path is 3723.24 m, so odometry is within **0.4 % on scale**, and the pose-graph module's final ATE is **0.27 % of path length**. Throughput ranged 24–31 scans/s (146–188 s wall) depending on how loaded the machine was — still ~2.5–3× real time for a 10 Hz sensor.

Run-to-run ATE varies by a few tenths of a metre because the Async\* stages are threaded and submap boundaries shift slightly; don't expect bit-identical repeats.

### Loop closure does not engage here — and that is the interesting part

With the default config, all **213 VGICP factors are between adjacent submaps; zero connect non-adjacent ones**:

```bash
grep -c "^matching_cost" results/dump_seq00/graph.txt   # 213
python3 - <<'EOF'
import re
p=[tuple(map(int,m.groups())) for l in open('results/dump_seq00/graph.txt')
   if (m:=re.match(r'matching_cost vgicp (\d+) (\d+)',l))]
print(sum(1 for i,j in p if abs(i-j)>2), 'non-adjacent of', len(p))
EOF
```

GLIM's *implicit* loop closure creates a factor when two submaps are within `max_implicit_loop_distance` (100 m) **and** overlap by `min_implicit_loop_overlap` (0.2). After ~12 m of LiDAR-only drift over 3.7 km, revisited submaps no longer overlap by 20 %, so no factor is ever born — the drift has to already be smaller than the submap overlap scale. The `pose_graph` module's explicit detector has the same chicken-and-egg problem in its geometric gate (`max_neighbor_dist` defaults to **5 m**, `gicp_max_correspondence_dist` 2 m).

`config_kitti_posegraph/` widens that gate (`max_neighbor_dist` 40 m, `min_inliear_fraction` 0.25, `gicp_max_correspondence_dist` 5 m) and does produce much larger global corrections — trajectory-vs-odometry displacement rises to a max of 18.0 m (mean 5.9 m) against 3.4 m (mean 0.83 m) for the default module — along with the best ATE in the table. But that module's dump does not enumerate its loop factors (`num_matching_cost_factors` belongs to the other module and reads 0), so **the loop-closure count here is unverified**; only the corrective effect is measured.

The honest takeaway for a lecture: GLIM is designed as a LiDAR-**inertial** system. An IMU keeps drift inside the geometric gates that its loop machinery relies on. Strip the IMU and the global optimizer still smooths, but place recognition never triggers — GLIM has no appearance-based loop detector to fall back on.

## The KITTI-specific config, and why each edit matters

`config_kitti/` is upstream `glim/config/` with exactly six files changed (`diff -rq` confirms):

| File | Change | Why |
|---|---|---|
| `config.json` | odometry → `config_odometry_ct.json`; sub/global mapping → `*_cpu.json` | Shipped config points at `*_gpu.json`, but `libodometry_estimation_gpu.so` does not exist in a CUDA-off build, so `load_module` would abort. |
| `config_sensors.json` | **`global_shutter_lidar: true`**; `T_lidar_imu` identity | The critical one — see below. |
| `config_sub_mapping_cpu.json` | `enable_imu: false`; `keyframe_update_strategy: DISPLACEMENT` | Without the first, `sub_mapping` builds IMU preintegration factors over an empty buffer. See the `OVERLAP` note below. |
| `config_global_mapping_cpu.json` | `enable_imu: false`; **`between_registration_type: NONE`** | The second prevents a hard crash — see below. |
| `config_preprocess.json` | `distance_near_thresh` 0.5→1.5, `downsample_resolution` 1.0→0.5, `random_downsample_target` 10000→20000 | Ego-vehicle returns removed; KITTI's 64-beam clouds carry enough points to justify finer downsampling. |
| `config_ros.json` | `extension_modules: []` | Headless. Unused by this driver, kept consistent. |

**`global_shutter_lidar: true` is load-bearing.** KITTI `.bin` files are **ring-major, not azimuth/time ordered** — measured on `000000.bin`: 64 azimuth wraps, one per laser, with elevation decreasing monotonically with index. GLIM's `TimeKeeper` therefore synthesizes per-point timestamps "based on the order of points" that are physically meaningless on KITTI, and CT-ICP happily deskews with them. Leave it `false` and ATE collapses from 2.60 m to **77.8 m** (path 171.85 m vs 393.65 m): odometry tracks ground truth for ~90 frames, then falls apart. Setting it `true` makes `CloudPreprocessor` zero all per-point times, i.e. treat each scan as a global-shutter snapshot.

An azimuth-derived alternative (KISS-ICP convention, `GLIM_KITTI_TIME_MODE=azimuth`) is implemented in the driver and reaches ATE 3.64 m, but needs `constant_velocity_inf_scale` dropped from 1e3 to 1e-3 as well. Global shutter is one change instead of two and scores better, so it is the default.

## Known upstream defects

- **`between_registration_type: GICP` (upstream default) can kill the process.** `GlobalMapping::create_between_factors()` runs an LM/GICP alignment between consecutive submaps with a hard-coded 0.5 m correspondence distance and no `try`/`catch`. On KITTI that is often degenerate and throws `gtsam::IndeterminantLinearSystemException` from `AsyncGlobalMapping`'s worker thread → `terminate called` → **exit 139 in 1 of 3 identical runs**. `NONE` takes the early-return branch (plain `BetweenFactor` from the odometry delta) and gave 5/5 clean runs.
- **`keyframe_update_strategy: OVERLAP` (upstream default) behaves differently on Ubuntu 24.04**, producing only 3 giant submaps (~87 frames each) instead of 14–18. Barely-overlapping submaps leave the global ISAM2 underconstrained → `IndeterminantLinearSystemException` on `x1` and `num_matching_cost_factors: 0`. `DISPLACEMENT` restores 18 submaps deterministically. Root cause of the platform difference was not isolated.
- **`find_package(glim)` alone fails to link**, with `cannot find -lGTSAM::GTSAM`. GTSAM 4.2.0 exports the target `gtsam`, never `GTSAM::GTSAM`; the wrapper comes from a `FindGTSAM.cmake` that *gtsam_points* installs, and `glim-config.cmake` calls `find_dependency(GTSAM)` **before** `find_dependency(gtsam_points)`. `kitti/CMakeLists.txt` works around it by calling `find_package(gtsam_points)` first. Any other consumer of this image hits the same wall.
- **CUDA is off only by accident.** glim's `CMakeLists.txt` defaults `BUILD_WITH_CUDA` to `ON`; it ends up off because `gtsam_points-config.cmake` sets it `OFF` and is pulled in first. Passing `-DBUILD_WITH_CUDA=OFF` explicitly would make that intentional.
- **The image is not reproducible across rebuilds**: `koide3/iridescence` is cloned without a tag (currently 1.0.3). Pin a tag before relying on byte-identical rebuilds. Also note the glim source clone must stay at `/glim` — the non-ROS build never installs its config JSONs (that `install(DIRECTORY config ...)` rule sits inside a ROS-only branch), so `/glim/config` is the only upstream copy.

## Outputs

`GlobalMapping::save()` writes GLIM's standard dump — the same format `glim_rosbag` produces:

| Path | Contents |
|---|---|
| `traj_lidar.txt`, `odom_lidar.txt` | TUM-format trajectories (optimized and raw odometry) |
| `traj_imu.txt`, `odom_imu.txt` | Identical to the LiDAR ones here: `T_lidar_imu` is identity and CT odometry tags frames as LIDAR |
| `graph.bin`, `values.bin`, `graph.txt` | Serialized GTSAM factor graph; `graph.txt` summarizes submap/frame/factor counts |
| `000000/`…`NNNNNN/` | Per-submap `points_compact.bin`, `covs_compact.bin`, `data.txt` (47 MB for seq 04's 18 submaps) |

## Other datasets

The driver is KITTI-specific, but the pipeline is not: any directory of velodyne-style `.bin` scans works if you supply a `times.txt` (it falls back to a synthetic 10 Hz clock and warns). For LiDAR + IMU sequences, GLIM's IMU-based odometry (`config_odometry_cpu.json`, `enable_imu: true`) is the intended path and is where its loop closure actually earns its keep — but feeding IMU needs either an extension of this driver or `glim_ros1`/`glim_ros2` in a ROS-based image.
