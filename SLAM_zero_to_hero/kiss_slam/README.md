# KISS-SLAM

LiDAR SLAM based on KISS-ICP. Simple, robust, and effective.

- **Repo**: https://github.com/PRBonn/kiss-slam (`v0.0.2`)
- **Sensors**: LiDAR (any 3D point cloud)
- **GPU**: not required

## Build

```bash
podman build -t slam_zero_to_hero:kiss_slam .
```

The image bundles `kiss-slam==0.0.2` plus the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags) reader so the `rosbag` dataloader works without a ROS install.

## Verified run — Hilti 2022 `exp14_basement_2.bag`

Hardware: Ryzen 9 7950X. Bag is the Hilti SLAM Challenge basement sequence (Hesai PandarXT-32 LiDAR, 740 scans, 74 s).

```bash
mkdir -p results
podman run --rm \
  -v ~/data/hilti_2022:/data:ro \
  -v "$(pwd)/results":/out:rw \
  -w /out \
  slam_zero_to_hero:kiss_slam \
  kiss_slam_pipeline --dataloader rosbag --topic /hesai/pandar /data/exp14_basement_2.bag
```

Outputs land in `results/slam_output/<timestamp>/`:

| File | Description |
|---|---|
| `exp14_basement_2_poses_tum.txt` | Trajectory in TUM format (`t tx ty tz qx qy qz qw`) |
| `exp14_basement_2_poses_kitti.txt` | Trajectory in KITTI 12-float-per-row format |
| `exp14_basement_2_poses.npy` | Numpy `(N,4,4)` pose array |
| `trajectory.png` | Top-down trajectory plot |
| `trajectory.g2o` | Pose graph (g2o format) |
| `local_maps/plys/000000.ply` | Final dense local map |
| `result_metrics.log` | Frequency / runtime / loop closure summary |

Last verified: 740-frame run on a Ryzen 9 7950X, **219 Hz average, ~5 ms/frame**, 0 loop closures (basement loop is too short for closure detection — expected).

## Other dataloaders

KISS-SLAM inherits all KISS-ICP dataloaders. With no ROS install in the image, the easy ones are:

- `--dataloader generic <dir-of-pcds-or-bin>` — any folder of `.bin` / `.pcd` / `.ply`
- `--dataloader rosbag --topic <topic> <bag>` — ROS1 `.bag` or ROS2 `.db3` (via `rosbags` lib)
- `--dataloader kitti --sequence 00 <kitti_root>` — needs the KITTI VO velodyne dump (`download_kitti.py` will fetch the ~80 GB zip on demand)

For the EuRoC stereo+IMU dataset, KISS-SLAM is **not applicable** (vision-only, no LiDAR).
