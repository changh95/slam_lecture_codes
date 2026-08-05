# ICP Point Cloud Registration using PCL

Code exercise for point-to-point and point-to-plane ICP registration, alignment
visualization, and sequential LiDAR odometry using PCL.

The three registration demos run on the **Stanford bunny** and the odometry demo
runs on a **KITTI odometry sequence**, so every exercise has ground truth to
score against.

---

## Project Structure

```
part2_ch03_06/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   ├── bun_zipper_res3.ply    # Stanford bunny - default input for demos 1-3
│   ├── 000000.bin             # KITTI velodyne scan (single frame)
│   └── scene.pcd
└── examples/
    ├── demo_common.hpp           # Cloud loading (.ply/.pcd/.bin), scale helpers, pose error
    ├── icp_basic.cpp             # Point-to-point ICP registration
    ├── icp_point_to_plane.cpp    # Point-to-plane ICP with normals
    ├── icp_visualization.cpp     # Visualize ICP alignment process
    └── lidar_odometry.cpp        # Sequential scan registration for LiDAR odometry
```

---

## Build

Dependencies:
- **PCL 1.10+** (`common`, `io`, `filters`, `registration`, `features`, `visualization`, `kdtree`, `search`) — required.
- **Eigen3 3.3+** — required.
- **MPI** — required (used by VTK/PCL visualization).

All four executables are always built; there are no optional targets.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch03_06
```

---

## Run

Every demo accepts `--help`. Demos 1-3 need no arguments: they load
`data/bun_zipper_res3.ply`, centre it, and build the source cloud by applying a
known transform, so the estimate is compared against the exact answer.

### 1-3. ICP on the Stanford bunny

```bash
# Point-to-point ICP
./build/icp_basic

# Point-to-plane ICP, side by side with point-to-point
./build/icp_point_to_plane

# Visualize the alignment (green: source, blue: target, red: aligned)
./build/icp_visualization

# Step-by-step: every keystroke in the viewer window runs one more ICP
# iteration, so you can watch the source cloud converge onto the target
# one step at a time ('q' quits; the window stays open once the iteration
# limit is reached)
./build/icp_visualization --step
```

Your own clouds work too — `.ply`, `.pcd`, and KITTI `.bin` are all accepted:

```bash
./build/icp_basic source.pcd target.pcd   # register a pair
./build/icp_basic my_model.ply            # single cloud + a known transform
```

`--generate` still builds synthetic inputs (half-sphere, indoor box, torus) for
running without any data at all:

```bash
./build/icp_basic --generate
./build/icp_visualization --generate --step
```

**Scale-relative parameters.** The bunny is only ~0.25 m across, so absolute
values tuned for LiDAR scans (0.5 m correspondence distance, 0.1 m normal
radius) would be larger than the whole model and make ICP meaningless. Voxel
size, correspondence distance, viewer camera distance and axis size are all
derived from the bounding-box diagonal instead, so the same demos work on the
bunny and on room- or street-scale clouds.

Expected result on the bunny: both methods recover the transform to
0.0000 deg / 0.000000 m, with point-to-plane roughly 3x faster and about an
order of magnitude lower in fitness score.

### 4. LiDAR odometry on a KITTI sequence

Point the demo at a KITTI odometry sequence directory. It finds `velodyne/`,
`calib.txt`, and `../../poses/NN.txt` by itself:

```bash
./build/lidar_odometry ~/data/kitti_vo_slam/extracted/dataset/sequences/04

# Shorter run while experimenting
./build/lidar_odometry .../sequences/04 --max-frames 60

# A plain velodyne directory also works (no ground-truth comparison)
./build/lidar_odometry /path/to/velodyne/
```

Options: `--max-frames N`, `--voxel S`, `--no-prediction`, `--generate`.

What the demo does per frame:

1. crop returns closer than 2.5 m (the ego vehicle, which moves with the sensor
   and biases matching towards zero motion) and beyond 80 m,
2. voxel-downsample,
3. align current scan to previous with point-to-point ICP, seeded with the
   previous frame-to-frame motion as a constant-velocity initial guess,
4. accumulate `T_global = T_global * T_relative`.

KITTI ground truth is given in the left-camera frame, so it is mapped into the
velodyne frame with the calib `Tr` matrix before being compared.

Outputs: `trajectory_kitti.txt`, `trajectory_tum.txt` (with real timestamps from
`times.txt`), and `trajectory_gt_kitti.txt`.

```bash
evo_traj kitti trajectory_kitti.txt --plot
evo_ape kitti trajectory_gt_kitti.txt trajectory_kitti.txt -va --plot
```

**Voxel size drives both speed and drift.** Measured on sequence 04, 60 frames:

| Voxel size | Final position error |
|---|---|
| 0.2 m (default) | 1.0 % of path length |
| 0.3 m | 2.4 % |
| 0.5 m | 4.3 % |

Larger leaves are faster but discard the vertical structure that constrains
forward motion.

**What drifts is heading, not distance.** Over the full 271-frame sequence 04
(~52 s on 8 cores, voxel 0.2 m):

```
Ground-truth path length: 393.645 m
Estimated path length:    395.27 m     <- 0.4 % scale error
ATE (translation RMSE):   10.705 m
Rotation RMSE:            4.223 deg
Final position error:     24.762 m     (6.29 % of path length)
  along track:            0.655 m      <- distance is essentially right
  across track:           24.754 m     <- accumulated heading error
Final rotation error:     6.982 deg
```

The per-frame translation is accurate, but a fraction of a degree of yaw error
per frame integrates into a large across-track offset. Scan-to-scan ICP has no
loop closure and no local map, so nothing ever corrects it.
[part2_ch03_07](../part2_ch03_07) compares GICP / NDT / KISS-ICP on the same
data.

### Docker

```bash
# Demos 1-3 (X11 for the visualizer)
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch03_06

# Demo 4 needs the KITTI dataset mounted
docker run -it --rm \
    -v ~/data/kitti_vo_slam:/kitti:ro \
    slam_zero_to_hero:part2_ch03_06 \
    ./lidar_odometry /kitti/extracted/dataset/sequences/04
```

---

## Getting the KITTI odometry data

Demo 4 needs the velodyne laser data and, for the ground-truth comparison, the
poses and calibration from the
[KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):

- `data_odometry_velodyne.zip` (80 GB) — extract only the sequence you want
- `data_odometry_poses.zip` — ground truth for sequences 00-10 only
- `data_odometry_calib.zip` — provides the `Tr` velodyne-to-camera transform

Sequence 04 is a good starting point: 271 scans, 393 m, mostly straight driving.

---

## References

- [PCL `registration` module](https://pointclouds.org/documentation/group__registration.html) (`IterativeClosestPoint`, `IterativeClosestPointWithNormals`)
- [PCL ICP tutorial](https://pcl.readthedocs.io/projects/tutorials/en/latest/iterative_closest_point.html)
- [PCL `visualization` module](https://pointclouds.org/documentation/group__visualization.html)
- [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/) (source of the bunny)
- [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
