# PIN-SLAM

LiDAR (and RGB-D) SLAM whose map is a cloud of **neural points**: each point carries a learned feature, and a tiny shared MLP decodes nearby features into a signed distance field. Tracking is scan-to-implicit-map registration against that SDF, loops are found with a neural-point descriptor and closed by pose-graph optimization — and because the map is made of points, a loop correction simply moves the points with their keyframes. A mesh can be extracted at any resolution by marching cubes.

- **Repo**: [PRBonn/PIN_SLAM](https://github.com/PRBonn/PIN_SLAM) (`v1.1.1`, commit `5976f3f`)
- **Paper**: [PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency](https://arxiv.org/abs/2401.09101) — Pan et al., IEEE T-RO 2024

Default dataset: **KITTI odometry sequence 00, Velodyne HDL-64E** (4541 scans, GT poses).

## Build

```bash
podman build -t localhost/slam_zero_to_hero:pin_slam .
```

CUDA 12.8 base + PyTorch 2.7.1 **cu128** wheels, so it runs on RTX 50xx (sm_120) as well as older GPUs. PIN-SLAM is pure PyTorch — there is no custom CUDA extension to compile. The image also carries Xvfb + ImageMagick for headless screenshots, and this folder's `scripts/` at `/PIN_SLAM/demo_scripts/`.

## Data

KITTI odometry: velodyne + poses + calib, laid out as `dataset/sequences/00/{velodyne,calib.txt,times.txt}` and `dataset/poses/00.txt`. On this machine it is at `~/data/kitti_vo_slam/extracted/dataset` (only sequences 00 and 04 have `velodyne/` extracted). To fetch it elsewhere, use [`../download_kitti.py`](../download_kitti.py). Be careful: that script takes **no arguments** (`--list`/`--help` are ignored and it runs in full). It downloads all five zips (~170 GB), extracts them to `~/data/kitti_vo_slam/dataset/` (not `extracted/`), and then **deletes every zip, including ones that failed to extract**. See [NOTES.md](NOTES.md).

The container mounts that folder at `/PIN_SLAM/data/kitti`, which is exactly the path upstream's `config/lidar_slam/run_kitti.yaml` expects, so the stock config is used unmodified.

## Run (headless)

```bash
podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v ~/data/kitti_vo_slam/extracted/dataset:/PIN_SLAM/data/kitti:ro \
  -v "$PWD/results":/PIN_SLAM/experiments \
  localhost/slam_zero_to_hero:pin_slam \
  python3 pin_slam.py ./config/lidar_slam/run_kitti.yaml -s -m
```

`-s` saves the neural point map, `-m` the marching-cubes mesh. Append `--range 0 1000 1` (start, end, step) to process a subset. Each run writes `results/test_kitti_<timestamp>/` with `slam_poses_kitti.txt`, `odom_poses_kitti.txt`, `gt_poses.ply`, `slam_poses.ply`, `mesh/mesh_*cm.ply`, `map/neural_points.ply` and a log; the ATE against `poses/00.txt` is printed at the end.

## Run with the GUI

On the host display:

```bash
podman run --rm -it \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/data/kitti_vo_slam/extracted/dataset:/PIN_SLAM/data/kitti:ro \
  -v "$PWD/results":/PIN_SLAM/experiments \
  localhost/slam_zero_to_hero:pin_slam \
  python3 pin_slam.py ./config/lidar_slam/run_kitti.yaml -v -m
```

`-v` opens PIN-SLAM's Open3D GUI (a separate process) with the current scan, the local neural-point map, the trajectory, loop edges and the periodic mesh. The window stays open after SLAM ends; close it (or Ctrl-C) to exit.

Headless, the same GUI can be run on a virtual display and screenshotted every N seconds:

```bash
podman run --rm \
  --runtime=/usr/bin/nvidia-container-runtime \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v ~/data/kitti_vo_slam/extracted/dataset:/PIN_SLAM/data/kitti:ro \
  -v "$PWD/results":/PIN_SLAM/experiments \
  localhost/slam_zero_to_hero:pin_slam \
  demo_scripts/xvfb_capture.sh experiments/gui 30 ./config/lidar_slam/run_kitti.yaml -v -m -s --range 0 1000 1
```

It writes `results/gui/gui_NNNN.png` every 30 s and `results/gui/gui_final.png` once the final mesh is built; the run itself lands in `results/test_kitti_<timestamp>/` as usual.

A finished run rendered to PNG (mesh + SLAM trajectory in red + GT in green; the script starts its own Xvfb, no GPU needed):

```bash
podman run --rm \
  -v "$PWD/results":/PIN_SLAM/experiments \
  localhost/slam_zero_to_hero:pin_slam \
  python3 demo_scripts/render_result.py experiments/test_kitti_<timestamp> experiments/pin_slam_kitti00_mesh.png
```

## Supported datasets

| Dataset | Command | Status |
|---|---|---|
| **KITTI odometry seq 00** (Velodyne) | `run_kitti.yaml` (stock) | ✅ **the default.** All 4541 scans: SLAM **ATE 0.85 m** over 3724 m, 32 loop corrections, 13.2 fps on an RTX 5090 |
| Other KITTI sequences | `run_kitti.yaml kitti NN -d -i data/kitti` (data loader) | not run here; needs `velodyne/` extracted for that sequence (only 00 and 04 in `extracted/`) |
| TUM RGB-D, Replica | `config/rgbd_slam/run_replica.yaml <loader> <seq> -d -i <path>` (`tum`, `replica` loaders) | shipped by upstream, not verified here |
| MulRan, Newer College, Hilti, nuScenes, HeLiPR, ROS bags | `config/lidar_slam/run_*.yaml`, `-d` loaders | shipped by upstream, not verified here |

## Results

Measured on an RTX 5090 (container, cu128), KITTI 00, stock `run_kitti.yaml`:

| Run | Frames | Path | ATE (SLAM) | ATE (odometry only) | KITTI drift (SLAM / odom) | Loops | Speed |
|---|---|---|---|---|---|---|---|
| Headless, `-s -m` | 4541 (all) | 3724.2 m | **0.847 m**, 0.72° | 5.581 m | 0.609 % / 0.555 % | 32 | 75.5 ms/frame = **13.2 fps** (tracking 28 ms, mapping 36 ms); 6 min 41 s wall including mesh |
| GUI under Xvfb, `-v -m -s --range 0 1000 1` | 1000 | 713.6 m | 0.356 m | 0.356 m | 0.742 % | 0 (no revisit yet) | 105.8 ms/frame = 9.5 fps with the GUI on; 4 min 34 s wall |

Loop closure is what brings the ATE down from 5.6 m to 0.85 m. The KITTI relative drift barely changes, because that metric only looks at 100–800 m segments. Final map: 47 MB of neural points, and the 24 cm mesh has 6.96 M vertices and 11.3 M triangles (481 MB PLY).

Full run: mesh plus SLAM trajectory (red) and GT (green), top-down (`render_result.py`):

![PIN-SLAM KITTI 00 mesh](results/pin_slam_kitti00_mesh.png)

PIN-SLAM's own evo plot of the same run (`results/test_kitti_<timestamp>/traj_plot_2d.png`):

![PIN-SLAM KITTI 00 trajectory](results/kitti00_traj_plot_2d.png)

PIN-SLAM's Open3D GUI at the end of the 1000-frame run (neural point map coloured by geometric feature; the info panel shows 655 k neural points and 713 m travelled):

![PIN-SLAM GUI](results/gui/gui_final.png)

Implementation notes, the evo/Tk crash, and why `xvfb-run` is not used are in [NOTES.md](NOTES.md).

