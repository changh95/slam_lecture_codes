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
├── data/                              # KITTI stereo pair (left.png = cam0, right.png = cam1)
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

All demos default to the bundled KITTI stereo pair (`data/left.png`,
`data/right.png`) using the KITTI seq 00–02 calibration
(`fx=fy=718.856, cx=607.1928, cy=185.2157`). The pose-recovering demos compare
their result against the KITTI stereo ground-truth extrinsic (R = I, baseline
along −X). Pass two image paths to any demo to use a different pair.

### Local

```bash
# Essential & Fundamental matrices, pose recovery vs GT extrinsic
./build/essential_fundamental_demo

# Recover pose (R, t) from E, compared to the GT extrinsic
./build/pose_recovery

# Epipolar line visualization (writes epipolar_visualization.png)
./build/epipolar_visualization

# 5-point relative pose (PoseLib) vs GT extrinsic
./build/relpose_poselib

# 5-point relative pose (OpenGV: Nister, Stewenius) vs GT extrinsic
./build/relpose_opengv

# Override the image pair (works for any demo)
./build/pose_recovery /path/to/left.png /path/to/right.png
```

### Docker

The `data/` pair is copied into the image, so the demos resolve it via `../data`
automatically (working directory is `build/`).

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch02_02

# Inside the container
./pose_recovery
./essential_fundamental_demo
```

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`findFundamentalMat`, `findEssentialMat`, `recoverPose`, `computeCorrespondEpilines`, `triangulatePoints`)
- [PoseLib](https://github.com/PoseLib/PoseLib)
- [OpenGV](https://laurentkneip.github.io/opengv/)
