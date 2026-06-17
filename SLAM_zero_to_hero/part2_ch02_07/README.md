# Triangulation

Code exercise for 3D point reconstruction from stereo/multi-view images using DLT,
mid-point, and OpenGV methods with OpenCV and Eigen.

---

## Project Structure

```
part2_ch02_07/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── triangulation_demo.cpp   # DLT, mid-point, and stereo depth triangulation (OpenCV + Eigen)
    └── triangulation_opengv.cpp # Linear and optimal L2 triangulation using OpenGV
```

---

## Build

Dependencies:
- **OpenCV 4.x** and **Eigen3** — required.
- **OpenGV** — optional. `triangulation_opengv` is built only when OpenGV is found (ships in `slam:base`).

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch02_07
```

---

## Run

### Local

```bash
# DLT, mid-point, and stereo depth methods
./build/triangulation_demo

# OpenGV triangulation methods (built only if OpenGV is available)
./build/triangulation_opengv
```

Both executables run without arguments.

### Docker

```bash
docker run -it --rm slam_zero_to_hero:part2_ch02_07
```

---

## References

- [OpenCV `calib3d` module — `triangulatePoints`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenGV](https://laurentkneip.github.io/opengv/)
- [Eigen3](https://eigen.tuxfamily.org/dox/)
