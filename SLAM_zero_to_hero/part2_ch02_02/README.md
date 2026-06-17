# Epipolar Geometry: Essential & Fundamental Matrix Estimation

Code exercise for estimating the Essential and Fundamental matrices, recovering
relative pose, and visualizing epipolar lines with OpenCV — plus 5-point
relative-pose solvers from PoseLib and OpenGV.

---

## Project Structure

```
part2_ch02_02/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── essential_fundamental_demo.cpp # Essential & Fundamental matrix estimation
    ├── pose_recovery.cpp              # Recover rotation and translation from E
    ├── epipolar_visualization.cpp     # Visualize epipolar lines
    ├── relpose_poselib.cpp            # 5-point relative pose using PoseLib
    └── relpose_opengv.cpp             # 5-point relative pose using OpenGV (Nister, Stewenius)
```

---

## Build

Dependencies:
- **OpenCV 4.x** (`core`, `imgproc`, `imgcodecs`, `features2d`, `calib3d`, `highgui`) and **Eigen3** — required.
- **PoseLib** and **OpenGV** — optional. `relpose_poselib` / `relpose_opengv`
  are built only when the respective library is found (both ship in `slam:base`).

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch02_02
```

---

## Run

### Local

```bash
# Essential & Fundamental matrix estimation
./build/essential_fundamental_demo

# Recover pose (R, t) from the Essential matrix
./build/pose_recovery image1.jpg image2.jpg

# Epipolar line visualization
./build/epipolar_visualization image1.jpg image2.jpg

# 5-point relative pose using PoseLib
./build/relpose_poselib

# 5-point relative pose using OpenGV (Nister, Stewenius solvers)
./build/relpose_opengv
```

`pose_recovery` and `epipolar_visualization` take two image paths as arguments;
the other demos run without arguments.

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:part2_ch02_02
```

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`findFundamentalMat`, `findEssentialMat`, `recoverPose`, `computeCorrespondEpilines`, `triangulatePoints`)
- [PoseLib](https://github.com/PoseLib/PoseLib)
- [OpenGV](https://laurentkneip.github.io/opengv/)
