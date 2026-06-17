# GTSAM: Factor Graph Optimization

Code exercise for 2D pose graph optimization and bundle adjustment using [GTSAM](https://gtsam.org/).

---

## Project Structure

```
part3_ch01_14/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── gtsam_basics.cpp           # 2D pose graph optimization
    └── gtsam_bundle_adjustment.cpp # Bundle adjustment with BAL dataset
```

---

## Build

Dependencies:
- **GTSAM** — required (ships in `slam:base`).
- **libopenmpi-dev** — required at build time to resolve a VTK/MPI dependency in `slam:base`.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part3_ch01_14
```

---

## Run

### Local

```bash
# 2D pose graph optimization
./build/gtsam_basics

# Bundle adjustment — requires a BAL dataset file
# Download from https://grail.cs.washington.edu/projects/bal/
wget https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2
bunzip2 problem-49-7776-pre.txt.bz2
./build/gtsam_bundle_adjustment problem-49-7776-pre.txt
```

### Docker

The Docker image downloads `problem-49-7776-pre.txt` automatically during build.

```bash
docker run -it --rm slam_zero_to_hero:part3_ch01_14

# Inside the container (working directory is build/)
./gtsam_basics
./gtsam_bundle_adjustment problem-49-7776-pre.txt
```

---

## References

- [GTSAM](https://gtsam.org/)
- [BAL (Bundle Adjustment in the Large) datasets](https://grail.cs.washington.edu/projects/bal/)
