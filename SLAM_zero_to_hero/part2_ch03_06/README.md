# ICP Point Cloud Registration using PCL

Code exercise for point-to-point and point-to-plane ICP registration, alignment
visualization, and sequential LiDAR odometry using PCL.

---

## Project Structure

```
part2_ch03_06/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── sample_clouds/
└── examples/
    ├── icp_basic.cpp              # Point-to-point ICP registration
    ├── icp_point_to_plane.cpp     # Point-to-plane ICP with normals
    ├── icp_visualization.cpp      # Visualize ICP alignment process
    └── lidar_odometry.cpp         # Sequential scan registration for LiDAR odometry
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

### Local

```bash
# Point-to-point ICP
./build/icp_basic source.pcd target.pcd

# Point-to-plane ICP (requires normals estimation internally)
./build/icp_point_to_plane source.pcd target.pcd

# Visualize alignment
./build/icp_visualization source.pcd target.pcd

# LiDAR odometry on a velodyne scan sequence
./build/lidar_odometry /path/to/velodyne/
```

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:part2_ch03_06
```

---

## References

- [PCL `registration` module](https://pointclouds.org/documentation/group__registration.html) (`IterativeClosestPoint`, `IterativeClosestPointWithNormals`)
- [PCL ICP tutorial](https://pcl.readthedocs.io/projects/tutorials/en/latest/iterative_closest_point.html)
- [PCL `visualization` module](https://pointclouds.org/documentation/group__visualization.html)
