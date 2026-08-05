# FAST-LIO2

Tightly-coupled LiDAR-inertial odometry on an iterated error-state Kalman filter, from HKU-MARS.

- **Repo**: https://github.com/hku-mars/FAST_LIO (built at commit `7cc4175`)
- **Sensors**: Livox / Velodyne / Ouster LiDAR + IMU (no camera)
- **GPU**: not required (pure CPU)

## Build

```bash
podman build -t slam_zero_to_hero:fast_lio2 .
```

The image bakes `ros:noetic`, Livox-SDK v1, `livox_ros_driver`, and FAST_LIO into `/catkin_ws`, producing `/catkin_ws/devel/lib/fast_lio/fastlio_mapping`. Stock configs ship for Avia, Horizon, Mid-360, Ouster-64, Velodyne, and MARSIM — **there is no Hesai config upstream**, so this directory adds one.

## Verified run — Hilti 2022 `exp14_basement_2.bag`

Hesai PandarXT-32 on `/hesai/pandar` + Alphasense IMU on `/alphasense/imu`, 74 s handheld basement walk. Nothing is baked into the image; the config, launch file, and helper scripts are bind-mounted, so no rebuild is needed to change them.

```bash
mkdir -p results/fullA
timeout 900 podman run --rm \
  -v ~/data/hilti_2022:/data:ro \
  -v "$PWD/results/fullA":/out \
  -v "$PWD/config/hilti_pandarxt32.yaml":/catkin_ws/src/FAST_LIO/config/hilti_pandarxt32.yaml:ro \
  -v "$PWD/launch/mapping_hilti.launch":/catkin_ws/src/FAST_LIO/launch/mapping_hilti.launch:ro \
  -v "$PWD/scripts":/scripts:ro \
  -v "$PWD/run_hilti_offline.sh":/run.sh:ro \
  -e RATE=1.0 -e SAVE_PCD=1 -e CONFIG=hilti_pandarxt32 \
  slam_zero_to_hero:fast_lio2 bash /run.sh
```

`run_hilti_offline.sh` starts a container-private `roscore`, launches `mapping_hilti.launch` headless, logs `/Odometry` to TUM format while `rosbag play` streams the bag at real time, then SIGINTs the node so it flushes its map and pose log. **No `--net=host`** — the ROS master lives in the container's own network namespace, so several ROS containers can run at once without fighting over port 11311.

Env knobs: `RATE` (playback rate), `DURATION` (seconds, empty = whole bag; `-e DURATION=20` gives a ~57 s smoke test), `CONFIG`, `SAVE_PCD`, `RELAY`.

Outputs in `results/fullA/`:

| File | Description |
|---|---|
| `fastlio_traj_tum.txt` | Trajectory in TUM format (`t tx ty tz qx qy qz qw`), 737 lines |
| `odometry_raw.csv` | Full `nav_msgs/Odometry` incl. covariance |
| `pos_log.txt` | FAST-LIO's own state log: time, angle, position, velocity, gyro/accel bias, gravity |
| `scans.pcd` | Accumulated map, 11,493,611 points (~368 MB, only with `SAVE_PCD=1`) |
| `fastlio_stdout.log` | Full node stdout incl. the per-scan `[ mapping ]` timing lines |

Last verified: Ryzen 9 7950X, 2026-08-05. **737 poses from 740 scans**, **37.934 m** path length, **6.05 ms/scan** average (match 3.07 ms, ICP 1.63 ms, solve 0.49 ms, map increment 0.05 ms) — about 16× real-time headroom. The 74.01 s bag played back in 75 s wall-clock at `RATE=1.0`.

Sanity numbers to compare against, from `scripts/traj_stats.py`:

```bash
python3 scripts/traj_stats.py results/fullA/fastlio_traj_tum.txt
# poses 737, path 37.934 m, end-start 21.350 m
# median inter-frame step 0.0540 m (= 0.54 m/s walk), max 0.1696 m
```

There is **no ATE for this sequence** — Hilti withheld ground truth for `exp14_basement_2`, and `evo` is not in the image. Judge the run by self-consistency instead: path length, inter-frame smoothness, and map crispness.

## Watching it run (GUI on your desktop)

Set `RVIZ=true` and add the X11 + GPU flags. `mapping_hilti.launch` then starts rviz with FAST_LIO's own `rviz_cfg/loam_livox.rviz`, so you watch the point cloud accumulate and the body frame move as the bag plays:

```bash
mkdir -p results/gui
timeout 900 podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/hilti_2022:/data:ro \
  -v "$PWD/results/gui":/out \
  -v "$PWD/config/hilti_pandarxt32.yaml":/catkin_ws/src/FAST_LIO/config/hilti_pandarxt32.yaml:ro \
  -v "$PWD/launch/mapping_hilti.launch":/catkin_ws/src/FAST_LIO/launch/mapping_hilti.launch:ro \
  -v "$PWD/scripts":/scripts:ro \
  -v "$PWD/run_hilti_offline.sh":/run.sh:ro \
  -e RVIZ=true -e CONFIG=hilti_pandarxt32 \
  slam_zero_to_hero:fast_lio2 bash /run.sh
```

`rviz` is installed for exactly this (it is not in `ros:noetic`), and renders on the RTX 5090 via the `--runtime` flags — without them it falls back to software GL. **No `xhost` change and no `--net=host` are needed**: podman here is rootless, so container root maps to your uid, which X already authorizes, and the connection goes over the bind-mounted socket.

`XDG_RUNTIME_DIR` only silences rviz's `QStandardPaths: XDG_RUNTIME_DIR not set` warning. To confirm the window mapped, use `xwininfo -root -tree | grep -i rviz` — `-root -children` shows only a stray `Tool Properties` dock and reads as a failed launch.

## Why the Hesai needs its own config

`config/hilti_pandarxt32.yaml` sets `lidar_type: 2` (the **Velodyne** branch). The reasoning matters more than the value:

| Point field | Hesai `/hesai/pandar` | What FAST-LIO wants |
|---|---|---|
| `x`,`y`,`z`,`intensity` | float32 ✅ | float32 |
| `ring` | uint16, 0–31 ✅ | uint16 |
| per-point time | **absent** — Hilti publishes an *absolute* float64 `timestamp` instead | float32 `time`, relative to the scan start |

So `pcl::fromROSMsg` leaves `time` at zero and prints `Failed to find match for field 'time'` — **once per scan, 740 times in a full run. This is expected, not a failure.** `velodyne_handler()` sees `time == 0`, sets `given_offset_time = false`, and rebuilds intra-scan time from each point's azimuth using `omega_l = 0.361 * scan_rate`. That fallback exists precisely for drivers that don't publish per-point time, which makes `lidar_type: 2` a legitimate, self-contained choice.

`lidar_type: 3` (Ouster) would **not** work: `oust64_handler()` reads a `uint32 t` field with no fallback, so every point would get curvature 0 and de-skewing would die silently. `lidar_type: 4` is MARSIM in this commit, not Hesai.

Because the azimuth path is load-bearing, `scan_rate: 10` matters (measured: 10.001 Hz, median dt 0.099996 s). `timestamp_unit` is inert here — there is no float32 `time` field for it to scale, so changing it does nothing.

Extrinsics come from upstream FAST-LIVO2's own Hilti-2022 calibration (`config/HILTI22.yaml` inside the `fast_livo2` image), not from guesswork: `extrinsic_T: [-0.001, -0.00855, 0.055]`, `extrinsic_R: [0,-1,0, -1,0,0, 0,0,-1]` (quaternion `[x,y,z,w] = [0.7071068, -0.7071068, 0, 0]`).

`blind: 0.5` replaces the stock `velodyne.yaml` value of 2.0, which would discard most of the geometry in a narrow basement corridor.

## Variant B — feed the sensor's real per-point timestamps

If you'd rather not rely on the azimuth reconstruction, `scripts/hesai_to_velodyne.py` republishes `/hesai/pandar` as `/velodyne_points` with a genuine float32 `time = timestamp - header.stamp`:

```bash
podman run --rm ... -e CONFIG=hilti_pandarxt32_relay -e RELAY=1 slam_zero_to_hero:fast_lio2 bash /run.sh
```

This silences all 740 PCL warnings and changes the result very little: 737 poses, **37.929 m** vs 37.934 m, with per-pose divergence of median 4.5 cm / max 8.5 cm over 38 m of travel. Variant A is the recommended teaching path (stock FAST-LIO, nothing extra to explain); variant B is the one to reach for on a faster or vehicle-mounted Hesai sequence, where a 100 ms sweep covers much more ground.

## Gotchas

- **`rosbag play --topics` is greedy.** `--topics A B /path/to.bag` swallows the bag path as a third topic and exits with *"You must specify at least 1 bag file to play back"* — while a wrapper script may happily report success. Keep the bag path *before* `--topics`.
- **FAST-LIO writes `scans.pcd` and `pos_log.txt` only after SIGINT unwinds `main()`.** A hard `podman kill`, or too short a wait after the interrupt, loses both. The script waits 15 s.
- **Expect 737 poses, not 740.** The first scan is dropped (`No point, skip this scan!`) and two more go to IMU initialization. A strict `scans == poses` assertion will fail.
- **The world frame is the first IMU body frame, and this IMU's +z points down** (at-rest `acc_z = -9.67`). Growing `z` in the TUM file means the operator is *descending* — don't present the trajectory as z-up.
- Full-rate playback needs ~100 MB/s of decompressed bag throughput. It held here even with concurrent image builds, but on a busier disk lower `RATE` rather than accepting dropped scans.
- `profiling/fast_lio2/hilti_hesai.yaml` (unused by any pipeline) had `lidar_type: 4` labelled as Hesai; 4 is MARSIM in this commit. Use `config/hilti_pandarxt32.yaml`.

## Other datasets

`lidar_type` selects the front end: `1` Livox Avia, `2` Velodyne, `3` Ouster-64, `4` MARSIM. Any ROS1 bag with a LiDAR `PointCloud2` plus a `sensor_msgs/Imu` stream will run — point `lid_topic`/`imu_topic` at it, set `scan_line`/`scan_rate` to the sensor, and supply the LiDAR↔IMU extrinsic. Upstream ships ready-made configs for Avia, Horizon, and Mid-360 in `/catkin_ws/src/FAST_LIO/config/`.
