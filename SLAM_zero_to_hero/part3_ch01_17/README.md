# Kimera-RPGO: Robust Pose Graph Optimization

Code exercise demonstrating Robust Pose Graph Optimization using Kimera-RPGO with GTSAM, including PCM-based outlier rejection and GNC-based robust solving.

---

## Project Structure

```
part3_ch01_17/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── rpgo_basics.cpp            # Basic pose graph construction and optimization with Kimera-RPGO
    └── rpgo_outlier_rejection.cpp # Pose graph with injected outlier loop closures and PCM rejection
```

---

## Build

Dependencies:
- **GTSAM 4.x** — required (built from source in Docker)
- **Kimera-RPGO** — required (built from source in Docker)
- **Boost**, **Eigen3**, **TBB** — required (pulled transitively by GTSAM/Kimera-RPGO)

```bash
# Local (GTSAM and Kimera-RPGO must be installed)
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build -t slam_zero_to_hero:part3_ch01_17 .
```

---

## Run

### Local

```bash
./build/rpgo_basics
./build/rpgo_outlier_rejection
```

### Docker

```bash
docker run -it --rm slam_zero_to_hero:part3_ch01_17

# Inside the container (working directory is /workspace)
./build/rpgo_basics
./build/rpgo_outlier_rejection
```

---

## References

- [Kimera-RPGO](https://github.com/MIT-SPARK/Kimera-RPGO)
- [GTSAM](https://github.com/borglab/gtsam)
