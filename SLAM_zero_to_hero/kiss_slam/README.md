# KISS-SLAM

LiDAR SLAM built on KISS-ICP: point-to-point ICP odometry with an adaptive threshold, local maps, a density-map loop detector, and g2o pose-graph optimization. The default demo runs on **KITTI odometry sequence 00** (Velodyne HDL-64E, 4541 scans, 3.7 km with revisits).

- **Repo**: [PRBonn/kiss-slam](https://github.com/PRBonn/kiss-slam) (`v0.0.2`)
- **Paper**: [KISS-SLAM: A Simple, Robust, and Accurate 3D LiDAR SLAM System With Enhanced Generalization Capabilities](https://arxiv.org/abs/2503.12660) — Guadagnino et al., IEEE/RSJ IROS 2025
- Front end: [KISS-ICP: In Defense of Point-to-Point ICP](https://arxiv.org/abs/2209.15397) — Vizzo et al., IEEE RA-L 2023
- CPU only; no GPU needed for SLAM or for the headless viewer capture.

## Data

The `kitti` loader needs `sequences/00/velodyne/*.bin`, `sequences/00/times.txt`, `sequences/00/calib.txt`, and `poses/00.txt`. The expected location is:

```
~/data/kitti_vo_slam/extracted/dataset/
├── poses/00.txt                    # 4541 GT poses
└── sequences/00/{calib.txt,times.txt,velodyne/000000.bin … 004540.bin}   # 4541 scans, 8.3 GB
```

`../download_kitti.py` fetches the official KITTI odometry zips from S3 into `~/data/kitti_vo_slam/`. The Velodyne zip alone is about 80 GB and covers all 22 sequences, so extract only `sequences/00/velodyne/` from it if disk is tight.

## Build

```bash
podman build -t localhost/slam_zero_to_hero:kiss_slam .
```

Bundles `kiss-slam==0.0.2` (`kiss-icp==1.3.0`, `open3d==0.19.0`), the pure-Python `rosbags` reader (so the `rosbag` loader needs no ROS install), Xvfb, and `scripts/capture_viewer.py`.

## Run (headless, KITTI 00)

```bash
mkdir -p results
podman run --rm \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results":/out -w /out \
  localhost/slam_zero_to_hero:kiss_slam \
  kiss_slam_pipeline --dataloader kitti --sequence 00 /data
```

Output lands in `results/slam_output/<timestamp>/`: poses in TUM/KITTI/npy, `00_gt_*` ground-truth twins, `trajectory.png`, `trajectory.g2o`, 32 local-map PLYs, and `result_metrics.log`. File-by-file details are in [NOTES.md](NOTES.md#outputs).

Measured on 2026-09-27/28 (Ryzen 9 7950X, shared with other jobs), two runs, before and after the image rebuild:

| Sequence | scans | est. path | GT path | ATE mean / rmse / max (unaligned) | trans. err | closures | rate | wall time |
|---|---|---|---|---|---|---|---|---|
| **00** | 4541 | 3726.18 m | 3724.19 m | **5.588 / 6.121 / 11.484 m** | 0.575 % | **7** | 54 Hz / 27 Hz (heavy load) | 4 min 11 s / 4 min 36 s |

Both runs match each other and the 2026-08-05 run in path, ATE, and closures; only the rate moves with machine load. The tool prints its own `ATE 0.917 m`, which is a different (aligned) definition; the table uses the per-frame distance to the GT twin (script in NOTES.md).

## Visualization

### On your desktop

```bash
podman run --rm -it \
  -e DISPLAY=$DISPLAY -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results":/out -w /out \
  localhost/slam_zero_to_hero:kiss_slam \
  kiss_slam_pipeline --visualize --dataloader kitti --sequence 00 /data
```

The `--visualize` viewer is an Open3D window (`RegistrationVisualizer`), not polyscope. It shows the current local map in yellow, keyposes as blue spheres joined by edges, and odometry poses in green.

**It starts playing** — the image patches upstream's paused start (set `-e KISS_SLAM_AUTOPLAY=0` to get it back). Press `space` to pause/resume, `n` to step while paused, `c` to re-centre (the camera does not follow), and `esc` to quit. The viewer costs about 60 % of throughput. For GPU rendering, add `--runtime=/usr/bin/nvidia-container-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`.

Verified 2026-09-28: the command above, run with `-n 600` on the host X display (`DISPLAY=:1`) and no GPU flags, opened the window and played by itself to the end. The window was grabbed with `xwd` mid-run (the camera stays at its start pose until `c` is pressed):

![Desktop viewer, KITTI 00](results/viewer/desktop_000600.png)

### Headless screenshots

`scripts/capture_viewer.py` runs the same pipeline with the same viewer, but auto-plays it. Inside its own Xvfb (Mesa llvmpipe, no GPU), it saves the window every `CAPTURE_EVERY` frames and once more on the last frame:

```bash
podman run --rm \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results":/out -w /out \
  -e CAPTURE_EVERY=1000 -e CAPTURE_DIR=/out/viewer \
  localhost/slam_zero_to_hero:kiss_slam \
  python3 /opt/kiss_slam/capture_viewer.py --visualize --dataloader kitti --sequence 00 /data
```

The full-sequence capture is slow. Mesa llvmpipe renders on the CPU, and the viewer re-adds one sphere per pose on every keypose update, so the rate falls from about 3 fps to about 0.5 fps. The full 4541-scan run on 2026-09-28 took 1 h 43 min (load average 15 to 60 from other jobs) and the container held about 27 GB of memory by scan 3900 (it grows with the pose count). It exited 0 with the same 0.575 % / 7-closure result as a headless run, and wrote `viewer_001000.png` to `viewer_004000.png` plus `viewer_004541.png` for the last frame. For a quick look, cap the run with `-n`, as in this command (verified on 2026-09-28: 3 min 24 s, writes `viewer_000500.png` and `viewer_001000.png`):

```bash
podman run --rm \
  -v ~/data/kitti_vo_slam/extracted/dataset:/data:ro \
  -v "$(pwd)/results":/out -w /out \
  -e CAPTURE_EVERY=500 -e CAPTURE_DIR=/out/viewer \
  localhost/slam_zero_to_hero:kiss_slam \
  python3 /opt/kiss_slam/capture_viewer.py --visualize --dataloader kitti --sequence 00 -n 1000 /data
```

| Scan 1000: yellow current local map | Scan 4541 (last frame): the full blue keypose graph over 3.7 km, green odometry, loops closed |
|---|---|
| ![](results/viewer/viewer_001000.png) | ![](results/viewer/viewer_004541.png) |

The pipeline also writes `slam_output/<timestamp>/trajectory.png`, a top-down plot of the optimized trajectory (black) with the 7 loop-closure edges in red. Below is a downscaled copy (the original is 12800x9600):

![KITTI 00 trajectory](results/kitti00_trajectory.png)

## Supported datasets

| Dataset | Command | Status |
|---|---|---|
| **KITTI odometry seq 00** (default) | `--dataloader kitti --sequence 00 /data` | ✅ verified 2026-09-27: 4541 scans, ATE 5.59 m mean / 0.575 %, 7 loop closures, viewer screenshots above |
| KITTI odometry seq 04 | `--dataloader kitti --sequence 04 /data` | ✅ verified 2026-08-05: 271 scans, ATE 0.59 m, 0 closures |
| Hilti 2022 `exp14_basement_2.bag` | `--config /cfg.yaml --dataloader rosbag --topic /hesai/pandar /data/exp14_basement_2.bag` | ✅ verified 2026-08-05, **only with [`config/hilti_indoor.yaml`](config/hilti_indoor.yaml)**. The stock defaults are tuned for outdoor driving and diverge indoors ([NOTES.md](NOTES.md)). |
| Any folder of `.bin` / `.pcd` / `.ply` | `--dataloader generic <dir>` | not verified here; writes frame indices as timestamps |
| ROS 1 `.bag` / ROS 2 `.db3` | `--dataloader rosbag --topic <topic> <bag>` | via the bundled `rosbags` reader |
| TUM, MulRan, nuScenes, NCLT, Apollo, Ouster, mcap, … | `--dataloader <name>` | 14 loaders inherited from KISS-ICP; not verified here |

Not applicable to EuRoC (vision-only, no LiDAR).
