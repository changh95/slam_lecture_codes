# UAMC notes

Findings from getting FAST-LIVO2-ROS2 to run on the COEX recording: first on
`lvi_set_2_restamped` (2026-09-27/28, the sections after this one), then on the same
recording with its original stamps, `lvi_coex_set_2` (2026-10-26, the demo's default now).
Everything here was measured on this host; nothing is taken from upstream. No ground truth
exists for these bags, so "works" means "does not diverge" plus a plausible map, not an ATE.

## `lvi_coex_set_2`: upstream's configs fit as they are

Decoding messages at t = 5, 100 and 250 s (header stamp minus bag receive time, and the
device time inside the message minus its header):

| Stream | Header − receive | Inside the message |
|---|---|---|
| Avia `CustomMsg` | +1200.16 … +1200.18 s | `timebase` − header = 0.0000 s |
| Mid-360 `PointCloud2` | +1200.17 … +1200.18 s | first point's `timestamp` − header = 0.0000 s |
| Avia IMU | +1200.20 … +1200.24 s | — |
| Mid-360 IMU | +1200.27 … +1200.29 s | — |
| Oak-D image | −0.15 … −0.12 s | — |

So both LiDARs and both IMUs are on the Livox clock, and the camera is on the host clock.
Against `lvi_set_2_restamped`, the restamping moved the LiDAR headers by 1200.183 s and the
Avia IMU by 1200.2527 s (upstream's `img_time_offset`), which is the −69.7 ms IMU error the
sections below chase. Here `imu_time_offset: 0.0` is right, and the Mid-360 point times
are consistent with their headers, so `mid360_handler` works without a code change.

**Camera → LiDAR clock offset, measured.** The camera's rotation rate, from LK-tracked
features between consecutive frames (rotation-only Kabsch on undistorted bearings; the
essential matrix is degenerate for these near-pure rotations), cross-correlated with the
Mid-360 gyro norm over 70 s windows:

| Bag, window | Offset (image header → Mid-360 IMU clock) | Correlation |
|---|---|---|
| `lvi_coex_set_2`, 40–110 s | 1200.288 s | 0.962 |
| `lvi_coex_set_2`, 180–250 s | 1200.298 s | 0.971 |
| `lvi_ghm_set`, 20–90 s (control) | −0.004 s | 0.927 |

The Gwanghwamun control comes out at zero, as its one-clock stamps predict. The coloured
map uses 1200.293 s (`IMG_OFFSET`). Upstream's 1200.2527 is 40 ms short of it, and the LiDAR
offset 1200.183 is 110 ms short.

**Avia ↔ Mid-360 extrinsic.** `uamc_geom.R_IA` was estimated on `lvi_ghm_set` six weeks
earlier. On COEX, a Kabsch fit of the Avia IMU gyro to the Mid-360 IMU gyro (8.5 k samples per
window) lands 0.24° and 0.17° from it, with the same residual (0.020–0.021 rad/s), and the
Mid-360 IMU lags the Avia IMU by 5 ms, as on Gwanghwamun. The rig did not change, so
`uamc_geom.py` serves both bags.

**Runs** (whole bag, real time; all configs upstream apart from outputs and `use_sim_time`):

| Config | Result |
|---|---|
| `coex_mid360_lvi.yaml` | 6065 poses, 319.1 m path, start-end 0.03 m, fastest 0.5 s window 2.3 m/s. After one rigid alignment (89.5° yaw: the Mid-360 sits rotated from the Avia) it matches the September Avia LIO run on the restamped bag to 0.10 m RMSE over the first 90 s, and the September Avia LVI run to 0.70 m at 120 s |
| `coex_mid360_lio.yaml` | 3342 poses, 318.2 m, start-end 0.03 m; at most 0.16 m (mean 0.03 m) from the LVI run |
| `coex_mid360_lio.yaml` + `COLOR_MAP` | Avia 52.3 M + Mid-360 53.2 M points, 31.9 % inside the camera image, 7.44 M-point multi-resolution map (0.5–16 cm), built in 35 s |
| `coex_avia_lvi.yaml` | follows the LIO runs for 45 s, then climbs 3–6 m on a flat floor, 7 m/s at 60.8 s, > 15 m/s at 218.7 s; stopped at 230 s, 505 m from the start. The same as `img +0.0697` on the restamped bag below, i.e. upstream's `img_time_offset` is the likely cause (not verified) |
| `avia_lvi.yaml` with `img_en: 0` (Avia LIO, not kept in `config/`) | whole bag without diverging, 3320 poses, 335.0 m, but ends 18.3 m from its start and 11.3 m below it. For its first 90 s it is the September restamped run with `imu_time_offset: -0.0697` to 0.00 m, which confirms the two bags differ only in their stamps; then the same 5 m/s jump at 93.5 s, and from ~210 s a sink the September run did not have (not bit-reproducible across the two) |

The LIO runs are deterministic: the same config rerun with RViz on the GPU instead of
llvmpipe matched to 1.7 mm over 3169 poses.

## What "restamped" means, measured

The bag is `lvi_coex_set_2` with its header stamps rewritten. Decoding the messages:

| Stream | Header stamp | Device time inside the message |
|---|---|---|
| Avia `CustomMsg` | host clock | `timebase` = header **+ 1200.183 s** |
| Mid-360 `PointCloud2` | host clock | per-point `timestamp` = header **+ 1200.183 s** |
| Avia IMU, Oak-D image | host clock | none |

Bag receive time equals the header stamp for every topic (median lag 0.000 s).
Upstream's `avia_lvi.yaml` carries `img_time_offset: 1200.252661`, which only makes sense
for un-restamped headers (LiDAR on device time, camera on host time). On the restamped bag
it must go.

`lvi_ghm_set` is different again: there the Avia header already equals `timebase` and the
camera is on the same clock, so no offset is needed at all.
Re-measured 2026-10-21 by decoding the first and 3000th messages: Avia `timebase` − header
= 0.0000 s, and the IMU, LiDAR and camera headers all sit 0.03–0.13 s behind the bag
receive time, so no offset is needed there. An Avia LiDAR-inertial run with
`imu_time_offset: 0.0` gets through the whole 841.8 s bag (636.0 m path) but with speed spikes
up to 15.4 m/s; the Mid-360 runs (README) never exceed 1.5 m/s.

## IMU runs 69.7 ms early

With `imu_time_offset: 0.0` the LiDAR-inertial filter diverges at t = 52.7 s, the first
time the rig is swung hard (gyro up to 2.3 rad/s). The same run at `-r 0.5` gives a
bit-identical trajectory, so it is not message loss. Sweeping `imu_time_offset`
(LIO, `img_en: 0`; FAST-LIVO2 *subtracts* it from IMU stamps):

| `imu_time_offset` | Result (full 334.7 s) |
|---|---|
| +0.10 / +0.05 / +0.02 | diverge at 44.8 / 48.6 / 49.7 s (90 s test) |
| 0.0 (upstream) | diverges at 52.7 s, ends 200 km away |
| −0.06 | 320.8 m path, start-end 14.0 m |
| **−0.0697** | **336.3 m path, start-end 6.6 m** |
| −0.08 | 339.1 m path, start-end 11.3 m |
| −0.10 | diverges at 155 s |

−0.0697 s is exactly 1200.2527 − 1200.183: consistent with the restamping tool having
shifted the IMU by upstream's camera offset and the LiDAR by its own `timebase` offset.
That explanation is inferred; the sweep is measured.

## The Avia IMU drops ~38 % of its samples

It is sampled on a 5 ms grid (200 Hz) but only 123 Hz arrive, with gaps up to 71 ms,
in `lvi_ghm_set` as well as COEX. It is a recording-side loss, not a playback one, and
the likely reason this rig is sensitive to IMU timing at all.

## LiDAR-visual-inertial is not stable on this bag

All with `imu_time_offset: -0.0697` unless noted, full bag at `-r 0.5`:

| Variant | First jump > 5 m/s | End |
|---|---|---|
| upstream LVI settings, `img 0.0`, `imu 0.0` | 57.7 s | diverged (km) |
| `img +0.0697` (rolling shutter + td estimate on) | 57.2 s | diverged |
| `img 0.0` (rolling shutter + td estimate on) | 155.3 s | diverged |
| `img −0.035 / −0.0697` | 57 / 45 s | diverged |
| td estimate off only, `img +0.0697` | 45.0 s | diverged |
| rolling shutter off only, `img 0.0` | 44.8 s | diverged |
| rolling shutter on, td estimate off, `img 0.0` | 150.8 s | diverged |
| **rolling shutter off, td estimate off, `img 0.0`** (`coex_avia_lvi.yaml` as of September) | 154.6 s | 413 m path, start-end 69.7 m, 14.8 m vertical extent |
| same, `img +0.0697` | 44.6 s | diverged |
| same, `img_point_cov 5000` (rolling shutter + td on) | 57.2 s | diverged |

So that config was only trusted for the first ~150 s. Against the LIO run it
agrees to 0.34 m up to t = 90 s; then the LIO run takes a 3.6 m jump at 94 s and the two
are 20.5 m apart by 146 s (README runs). Which is right cannot be told without ground truth.
The README's coloured-map demo therefore plays 150 s.

## Mid-360 does not work on the restamped bag

`preprocess.cpp::mid360_handler` computes each point's time as
`pt.timestamp * 1e-9 - header` — 1200 s on this bag, so every point is "in the future"
and the first scan leaves 0 effective features. The shipped `mid360_lvi.yaml` (which is
set up for this rig's `/livox/lidar_192_168_1_150`) jumps 16 m in the first 10 s of a
static start. Fixing it needs a code change (use the offset from the first point), not a
config change; not done here.

## Upstream bugs worked around in the image

- **vikit cannot read the RationalPolynomial camera.** `camera_loader.cpp` at `6f213c7`
  reads only the Pinhole model through `getRemoteParam()`; every other model, including
  the `RationalPolynomial` used by all four FAST-LIVO2-ROS2 launches here, uses
  `getParam(nh, "parameter_blackboard/cam_width")`, a local lookup on the laserMapping
  node that never succeeds. fx = 0 → `fastlivo_mapping` dies with SIGFPE (exit -8) before
  the first scan. `patches/vikit_remote_camera_params.patch` switches all models to
  `getRemoteParam()` and lengthens its 100 ms service wait to 10 s. The camera is loaded
  even with `img_en: 0`, so the LIO run needs this too.
- **The PCD map is never downsampled.** `savePCD()` runs one `pcl::VoxelGrid` at 0.15 m
  over the whole map; at ~500 m extent the voxel index overflows int32, PCL prints
  "Integer indices would overflow" and returns the input, so `all_downsampled_points.pcd`
  is a second copy of the ~1 GB raw cloud. `scripts/voxel_pcd.py` redoes it with 64-bit
  keys into `pcd/map.pcd` (0.1 m).
- **The map is written only on SIGINT.** A bash script's background jobs start with SIGINT
  ignored, so `kill -INT` never reached the node and no PCD appeared. `run_coex.sh` turns
  job control on (`set -m`) first. If the node is still behind the bag when SIGINT
  arrives it can crash on shutdown (`could not create publisher: context is invalid`)
  and lose the PCD; the trajectory file is written per pose and survives.
- The launch files start an `image_transport republish` node for `/left_camera/image`
  (a Retail_Street leftover); it is harmless here and segfaults on shutdown.

## Throughput

At `-r 1.0` on this host (32 cores shared with other jobs) the LIO run keeps up
(10 Hz, 3320 poses for 3325 scans). FAST-LIVO2 warns "IMU and LiDAR not synced" whenever
its single-threaded spin falls behind; those warnings do not change the result
(`-r 1.0` and `-r 0.5` trajectories are identical).

## RViz on the host display

With `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`, rviz2 (humble) often dies on the final
SIGINT with `Aborted` (`pthread_mutex_lock ... Assertion`) or `Segmentation fault`. It
happens after the last screenshot, and the trajectory, map and stats are still written.
It is not the course view controller: in an A/B of three 40 s runs each (2026-09-28, load
average ~40), `UnifiedOrbit` crashed 2/3 and stock `rviz_default_plugins/Orbit` 2/3. On
the private Xvfb display every shutdown was clean.

Since 2026-10-26 `run_coex.sh` no longer sends RViz that SIGINT on a host display: it writes
everything first and leaves RViz up with the final map until the user closes the window
(`HOLD_RVIZ`), so the crash no longer shows up there.

**GPU rendering needs `compute`.** With `NVIDIA_DRIVER_CAPABILITIES=graphics,utility`, the
container gets `libGLX_nvidia.so` but not `libnvidia-gpucomp.so` (driver 580), which the NVIDIA
toolkit only mounts with `compute`. `libGLX_nvidia` then fails to load, glvnd falls back to
Mesa without an error, and RViz draws with llvmpipe even though `__GLX_VENDOR_LIBRARY_NAME=nvidia`
is set and the GPU devices are in the container. With `graphics,compute,utility` `glxinfo -B` reports the RTX 5090,
and RViz went from 10 to 31 fps on the COEX runs. `run.log` prints the renderer RViz got.
