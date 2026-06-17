# Ceres-Solver: Nonlinear Least Squares

Code exercise for nonlinear least squares optimization with Ceres Solver — curve fitting, 2D pose graph, and bundle adjustment with the BAL dataset.

---

## Project Structure

```
part3_ch01_15/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── ceres_basics.cpp              # Curve fitting and 2D pose graph optimization
    └── ceres_bundle_adjustment.cpp   # Bundle adjustment with BAL dataset
```

---

## Build

Dependencies:
- **Ceres Solver** — required (ships in `slam:base`).
- **libopenmpi-dev** — required at runtime to resolve a VTK MPI dependency (installed by the Dockerfile).

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part3_ch01_15
```

---

## Run

### Local

```bash
# Curve fitting and 2D pose graph
./build/ceres_basics

# Bundle adjustment — requires a BAL dataset file
./build/ceres_bundle_adjustment problem-49-7776-pre.txt
```

Download the BAL dataset from <https://grail.cs.washington.edu/projects/bal/> (e.g. `ladybug/problem-49-7776-pre.txt.bz2`), then `bunzip2` it.

### Docker

The Dockerfile downloads and extracts `problem-49-7776-pre.txt` automatically into the build directory.

```bash
docker run -it --rm slam_zero_to_hero:part3_ch01_15

# Inside the container (working directory is build/)
./ceres_basics
./ceres_bundle_adjustment problem-49-7776-pre.txt
```

---

## References

- [Ceres Solver](http://ceres-solver.org/)
- [BAL Dataset](https://grail.cs.washington.edu/projects/bal/)
