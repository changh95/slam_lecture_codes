# Advanced ICP Methods: GICP, NDT, TEASER++, and KISS-ICP

Code exercise for point cloud registration using GICP, NDT, TEASER++, and KISS-ICP via PCL.

---

## Project Structure

```
part2_ch03_07/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── sample_sequences/
└── examples/
    ├── gicp_demo.cpp        # Generalized ICP (plane-to-plane) registration
    ├── ndt_demo.cpp         # Normal Distributions Transform registration
    ├── teaser_demo.cpp      # TEASER++ global registration (stub without -DUSE_TEASER=ON)
    ├── kiss_icp_demo.cpp    # KISS-ICP odometry (stub without -DUSE_KISS_ICP=ON)
    └── method_comparison.cpp # Compare ICP vs GICP vs NDT
```

---

## Build

Dependencies:
- **PCL 1.10+** (`common`, `io`, `filters`, `features`, `registration`, `visualization`) and **MPI** — required.
- **TEASER++** — optional; `teaser_demo` uses the full solver only when built with `-DUSE_TEASER=ON` and `teaserpp` is found; otherwise a stub is built.
- **KISS-ICP** — optional; `kiss_icp_demo` uses the full pipeline only when built with `-DUSE_KISS_ICP=ON` and `kiss_icp` is found; otherwise a stub is built.

```bash
# Local (core only)
mkdir build && cd build
cmake ..
make -j4

# Local (with optional deps)
cmake -DUSE_TEASER=ON -DUSE_KISS_ICP=ON ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch03_07
```

---

## Run

### Local

```bash
# Generalized ICP
./build/gicp_demo source.pcd target.pcd

# NDT registration (optional resolution argument)
./build/ndt_demo source.pcd target.pcd [resolution]

# Compare ICP vs GICP vs NDT
./build/method_comparison source.pcd target.pcd

# TEASER++ global registration
./build/teaser_demo source.pcd target.pcd

# KISS-ICP odometry (pass directory of scans)
./build/kiss_icp_demo /path/to/sequence_dir
```

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:part2_ch03_07
```

---

## References

- [PCL `registration` module](https://pointclouds.org/documentation/group__registration.html) (`GeneralizedIterativeClosestPoint`, `NormalDistributionsTransform`)
- [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus)
- [KISS-ICP](https://github.com/PRBonn/kiss-icp)
