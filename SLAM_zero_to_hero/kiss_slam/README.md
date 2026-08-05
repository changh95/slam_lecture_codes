# KISS-SLAM

LiDAR SLAM built on KISS-ICP: point-to-point ICP odometry with an adaptive threshold, plus local maps, a density-map loop detector, and g2o pose-graph optimization. Simple, robust, and effective — *when its ranges match your sensor*.

- **Repo**: https://github.com/PRBonn/kiss-slam (`v0.0.2`)
- **Sensors**: LiDAR (any 3D point cloud)
- **GPU**: not required

## Build

```bash
podman build -t slam_zero_to_hero:kiss_slam .
```

The image bundles `kiss-slam==0.0.2` plus the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags) reader, so the `rosbag` dataloader works with **no ROS install, no roscore, no network, and no X11**. Worth stating explicitly: every other ROS-based system in this repo needs a `roscore`, and this one does not — so it never contends for port 11311.

## Verified run — KITTI odometry (the sensor KISS-ICP's defaults are tuned for)

```bash
mkdir -p results
podman run --rm \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results":/out -w /out \
  slam_zero_to_hero:kiss_slam \
  kiss_slam_pipeline --dataloader kitti --sequence 00 /data
```

The `kitti` loader knows the ground truth, so it writes GT twins next to the estimate and reports accuracy itself:

| Sequence | scans | est. path | GT path | ATE mean | ATE rmse | ATE max | trans. err | closures | rate |
|---|---|---|---|---|---|---|---|---|---|
| 04 | 271 | 393.11 m | 393.65 m | 0.594 m | 0.644 m | 0.954 m | 0.384 % | 0 | 45 Hz |
| 00 | 4541 | 3726.18 m | 3724.19 m | 5.588 m | 6.121 m | 11.484 m | 0.575 % | **7** | 56–67 Hz |

ATE above is computed directly against the GT twin with no alignment (the `kitti` loader's poses are already in the GT frame):

```bash
python3 - <<'EOF'
import numpy as np
d='results/slam_output/<timestamp>'; s='00'
e=np.loadtxt(f'{d}/{s}_poses_tum.txt')[:,1:4]; g=np.loadtxt(f'{d}/{s}_gt_tum.txt')[:,1:4]
err=np.linalg.norm(e-g,axis=1)
print('ATE mean %.3f rmse %.3f max %.3f m'%(err.mean(),np.sqrt((err**2).mean()),err.max()))
EOF
```

Two independent runs of the seq-00 command produced identical accuracy (same ATE to three decimals, same 7 closures, same 0.575 % translation error); only the rate moved with machine load.

Seq 00 is the interesting one: **7 loop closures detected and optimized**, which is what a 3.7 km sequence with revisits should produce. For contrast, GLIM's geometric loop gate finds zero on the same data — see `../glim/README.md`.

Do **not** quote the tool's own `Absolute Rotational Error` for seq 04: it prints 0.620 rad (35.5°) while the same log reports 0.001 deg/m, and the actual per-frame quaternion deviation against GT is 0.044°. The translation metrics are sound; that one is not.

`--dataloader kitti` needs `velodyne/` scans — present here for sequences 00 (4541) and 04 (271). Note that the claim `download_kitti.py` will fetch data on demand is **wrong**: that script ships in the kiss-icp GitHub repo, not the pip package, and is not in this image.

## Verified run — Hilti 2022 `exp14_basement_2.bag` (needs a tuned config)

The out-of-the-box defaults **fail on this sequence**, and they fail quietly. Use the config in this directory:

```bash
mkdir -p results
podman run --rm \
  -v ~/data/hilti_2022:/data:ro \
  -v "$(pwd)/results":/out \
  -v "$(pwd)/config/hilti_indoor.yaml":/cfg.yaml:ro \
  -w /out \
  slam_zero_to_hero:kiss_slam \
  kiss_slam_pipeline --config /cfg.yaml --dataloader rosbag --topic /hesai/pandar /data/exp14_basement_2.bag
```

| | default config | `config/hilti_indoor.yaml` | FAST-LIO2 / FAST-LIVO2 reference |
|---|---|---|---|
| poses | 740 | 740 | 737 / 738 |
| path length | 111.41 m | **45.87 m** | 37.93 m / 37.94 m |
| start→end distance | 29.86 m | **23.10 m** | 21.35 m / 21.40 m |
| median inter-frame step | 0.112 m | **0.053 m** | 0.054 m |
| max inter-frame step | 0.895 m (≈ 9 m/s!) | **0.297 m** | 0.170 m |
| steps > 0.5 m | 27 | **0** | 0 |
| rate | 182 Hz / 6 ms | 155 Hz / 7 ms | — |

Last verified: Ryzen 9 7950X, 2026-08-05, 740/740 frames, 0 loop closures — correct here, because the basement leg is not a closed loop (start and end are 23 m apart).

### How we know the default run is wrong, not merely different

A trajectory can't be judged by eyeballing its length, so this was arbitrated against the raw scans. Aggregate 30 consecutive `/hesai/pandar` scans into one frame using each trajectory's poses, then count occupied 0.10 m voxels: correct poses land surfaces on top of each other, wrong poses smear them across more voxels. "Identity" means pretending the sensor never moved — the floor any real SLAM system must beat.

| Window | identity | FAST-LIO2 | KISS default | KISS tuned |
|---|---|---|---|---|
| scans 100–129 | 1.000 | 0.587 | 0.983 | **0.539** |
| scans 300–329 | 1.000 | 0.485 | 0.767 | **0.403** |
| scans 500–529 | 1.000 | 0.434 | 0.959 | **0.503** |

The default configuration scores 0.77–0.98 — barely distinguishable from assuming zero motion. Tuned, it matches or beats FAST-LIO2 on two of the three windows. Script: `../fast_lio2/scripts/traj_arbiter.py`. The inverse pose convention was tested too and scores *worse* than identity, confirming the file really is sensor→world rather than mis-inverted.

### Why the defaults fail here

Two settings do almost all the damage:

1. **`max_range: 100.0` combined with `mapping.voxel_size: null`.** When `voxel_size` is null, KISS-ICP derives it from `max_range` — so a 100 m range yields a **1.0 m voxel**, discarding exactly the fine geometry a narrow basement corridor offers. `max_range: 25.0` plus an explicit `voxel_size: 0.3` fixes it. `min_range: 0.0` also admits returns off the operator's own body; 1.0 m drops them.
2. **`deskew: true` makes it worse, not better** — path 58.75 m with, 45.87 m without, and 4 non-physical >0.5 m jumps versus 0.

That second one is counter-intuitive, so it's worth being precise about what it is *not*. The bag's per-point timestamps are **fine**: measured on scan 0 of `/hesai/pandar`, 52,109 points, `timestamp` is float64 absolute unix time, strictly monotonic with zero wraps, spanning 0.099985 s, and the cloud is azimuth-major (rings cycle 0–31 within each column). Normalization to [0,1] happens per scan inside KISS-ICP's C++ preprocessor, so absolute values are handled correctly. The likely cause is the motion model instead: KISS-ICP deskews using the previous frame-to-frame delta as a constant-velocity prior, which fits a walking operator's rotation poorly across a 100 ms sweep.

One genuine latent bug surfaced while checking this, in `kiss_icp/tools/point_cloud2.py::read_point_cloud`: NaN rows are filtered from `points` but **not** from `timestamps`, so the two arrays desynchronize on any cloud containing NaNs. Harmless for this bag (`is_dense: True`), but it would silently corrupt deskewing on a sensor that publishes NaN returns.

## Watching it run (GUI on your desktop)

`--visualize` / `-v` opens an Open3D window titled `RegistrationVisualizer` showing the local map and the growing trajectory:

```bash
podman run --rm -it \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results":/out -w /out \
  slam_zero_to_hero:kiss_slam \
  kiss_slam_pipeline --visualize --dataloader kitti --sequence 00 /data
```

No extra packages are needed — Open3D 0.19.0 and its X11/GL dependencies are already in the image, and it renders on the RTX 5090 via the `--runtime` flags. As elsewhere, **no `xhost` change and no `--net=host`**. `XDG_RUNTIME_DIR` suppresses a misleading `error: XDG_RUNTIME_DIR not set in the environment.` line that GLFW prints while probing for Wayland; the X11 path works regardless.

Four things will confuse you if nobody says them first:

- **It starts paused.** The window sits at `0/N` until you press `space`, and the instruction is printed only to stdout — so a student watching the window concludes it hung. Press `space` to run, `n` to step one frame, `esc` to quit.
- **`--visualize`'s help text is wrong.** It claims "Visualize Ground Truth Loop Closures"; no ground truth and no closures are drawn. The closure-drawing code is dead in v0.0.2 — `RegistrationVisualizer.__init__` sets `self.closures = []` and nothing ever appends to it, so the red closure edges can never appear. Upstream bug, not something to fix here.
- **The camera does not follow.** The viewpoint is set once at startup, so on a long sequence the map slides out of frame and the window looks frozen. Press `c` to re-centre.
- **It costs ~60 % throughput** — 71 Hz headless versus ~28 Hz with the viewer on KITTI 00 — because every odometry pose is removed and re-added as an individual sphere mesh on each keypose update. Accuracy is unaffected, so keep runs you intend to measure headless.

To confirm the window mapped, use `xwininfo -root -tree | grep RegistrationVisualizer`; `-root -children` misses it because the window manager reparents the GLFW window.

## Outputs

Filenames derive from the loader and sequence, so they are not fixed. With `--dataloader rosbag` they are named after the bag; with `--dataloader kitti --sequence 04` you get `04_poses_*` **plus** `04_gt_*` ground-truth twins.

| File | Description |
|---|---|
| `<name>_poses_tum.txt` | Trajectory in TUM format (`t tx ty tz qx qy qz qw`) |
| `<name>_poses_kitti.txt` | Trajectory in KITTI 12-float-per-row format |
| `<name>_poses.npy` | Numpy `(N,4,4)` pose array |
| `<name>_gt_*.txt` / `.npy` | Ground truth — **only** for loaders that have it (e.g. `kitti`) |
| `config.yml` | The effective configuration. Check this first when a run misbehaves. |
| `trajectory.png` | Top-down trajectory plot |
| `trajectory.g2o` | Pose graph. Sequential-only when no closures fire: N vertices, N−1 edges, and a `FIX` on both the first and the last vertex. |
| `local_maps/plys/NNNNNN.ply` | One PLY **per local map**, not a single fused map: 1 for Hilti, 4 for KITTI 04, 32 for KITTI 00. Use `--refuse-scans` for one global map. |
| `local_maps/local_map_graph.g2o` | Local-map graph. Holds **N+1** vertices for N PLYs — the last local map is empty, and Open3D warns `Write PLY failed: point cloud has 0 points`. |
| `result_metrics.log` | Frequency / runtime / closure count, plus accuracy for loaders with GT |

Two caveats on the artifacts. The `slam_output/latest` symlink points at an **in-container absolute path** (`/out/slam_output/...`), so it dangles when browsed from the host. And runs are not bit-reproducible: two identical seq-00 runs differed on 444 of 4541 lines, though only by a max of 0.000173 m.

Throughput figures move with machine load — the same Hilti run measured 219 Hz in an earlier session and 182 Hz here on the same CPU — so treat Hz and ms/frame as indicative, and frame counts, closure counts, and ATE as the reproducible numbers.

## Other dataloaders

KISS-SLAM inherits KISS-ICP's loaders — **14** of them: `apollo`, `boreas`, `generic`, `helipr`, `kitti`, `kitti_raw`, `mcap`, `mulran`, `ncd`, `nclt`, `nuscenes`, `ouster`, `rosbag`, `tum`. The handy ones here:

- `--dataloader generic <dir>` — any folder of `.bin` / `.pcd` / `.ply`. Writes integer frame indices as timestamps rather than real seconds.
- `--dataloader rosbag --topic <topic> <bag>` — ROS1 `.bag` or ROS2 `.db3`, via `rosbags`.
- `--dataloader kitti --sequence NN <kitti_root>` — the only path here that gives you ground truth for free.
- `--dataloader tum` — usable against `~/data/tum_rgbd/`.

For the EuRoC stereo+IMU dataset, KISS-SLAM is **not applicable** (vision-only, no LiDAR).
