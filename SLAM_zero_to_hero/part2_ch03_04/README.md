# Introduction to Point Cloud Processing using PCL

Code exercise demonstrating point cloud I/O, filtering, normal estimation, segmentation, and visualization using PCL.

---

## Project Structure

```
part2_ch03_04/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── sample_clouds/           # Sample point cloud files
└── examples/
    ├── basic_io.cpp             # Read/write PCD and PLY files
    ├── filtering.cpp            # VoxelGrid, StatisticalOutlierRemoval, PassThrough filters
    ├── normal_estimation.cpp    # Compute surface normals (standard and OpenMP)
    ├── segmentation.cpp         # RANSAC plane segmentation and Euclidean clustering
    └── visualization.cpp        # PCL visualizer demo
```

---

## Build

Dependencies:
- **PCL 1.10+** (`common`, `io`, `filters`, `features`, `segmentation`, `visualization`) — required.
- **MPI** — required (used by VTK, which PCL visualization depends on).
- **Eigen3** — pulled in transitively by PCL.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch03_04
```

---

## Run

### Local

```bash
# Basic I/O (read/write PCD and PLY)
./build/basic_io input.pcd output.pcd

# Filtering (VoxelGrid downsampling, SOR, PassThrough)
./build/filtering cloud.pcd --voxel 0.1 --sor 1.0

# Normal estimation
./build/normal_estimation cloud.pcd --radius 0.1

# Segmentation (RANSAC ground plane + Euclidean clustering)
./build/segmentation cloud.pcd --ground --cluster

# Visualization
./build/visualization cloud.pcd
```

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:part2_ch03_04
```

---

## References

- [PCL Documentation](https://pointclouds.org/documentation/)
- [PCL Tutorials](https://pcl.readthedocs.io/projects/tutorials/en/latest/)
