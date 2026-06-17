# g2o: General Graph Optimization

Code exercise for graph-based nonlinear optimization with g2o — 2D pose graph optimization and bundle adjustment with the BAL dataset format.

---

## Project Structure

```
part3_ch01_13/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── g2o_basics.cpp            # 2D pose graph optimization (SE2 vertices + edges)
    └── g2o_bundle_adjustment.cpp # Bundle adjustment with BAL dataset (SE3 + point vertices)
```

---

## Build

Dependencies:
- **g2o** (`core`, `stuff`, `types_slam2d`, `types_sba`, `solver_dense`, `solver_eigen`) — required.
- **Eigen3** — required.
- **spdlog v1.12.0** — required. The Dockerfile rebuilds spdlog from source with bundled fmt to resolve a version mismatch against the system `libfmt`.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part3_ch01_13
```

---

## Run

### Local

```bash
# 2D pose graph optimization
./build/g2o_basics

# Bundle adjustment — requires a BAL dataset file
./build/g2o_bundle_adjustment problem-49-7776-pre.txt
```

Download a BAL dataset (ladybug series): https://grail.cs.washington.edu/projects/bal/

```bash
wget https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2
bunzip2 problem-49-7776-pre.txt.bz2
```

### Docker

The Dockerfile downloads `problem-49-7776-pre.txt` into the build directory automatically.

```bash
docker run -it --rm slam_zero_to_hero:part3_ch01_13

# Inside the container (cwd is /workspace/part3_ch01_13/build)
./g2o_basics
./g2o_bundle_adjustment problem-49-7776-pre.txt
```

---

## References

- [g2o GitHub](https://github.com/RainerKuemmerle/g2o)
- [BAL Dataset](https://grail.cs.washington.edu/projects/bal/)
- [spdlog](https://github.com/gabime/spdlog)
