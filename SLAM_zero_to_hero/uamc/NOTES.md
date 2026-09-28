# UAMC notes

Findings from getting FAST-LIVO2-ROS2 to run on the COEX bag `lvi_set_2_restamped`
(2026-09-27/28). Everything here was measured on this host; nothing is taken from upstream.
No ground truth exists for these bags, so "works" means "does not diverge" plus a
plausible map, not an ATE.

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
| **rolling shutter off, td estimate off, `img 0.0`** (`coex_avia_lvi.yaml`) | 154.6 s | 413 m path, start-end 69.7 m, 14.8 m vertical extent |
| same, `img +0.0697` | 44.6 s | diverged |
| same, `img_point_cov 5000` (rolling shutter + td on) | 57.2 s | diverged |

So `coex_avia_lvi.yaml` is only trusted for the first ~150 s. Against the LIO run it
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
