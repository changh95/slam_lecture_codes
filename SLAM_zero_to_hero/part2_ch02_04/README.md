# Homography for Visual SLAM

Code exercise for homography estimation, H/F model selection (ORB-SLAM style), and image stitching — using OpenCV and optionally PoseLib.

---

## Project Structure

```
part2_ch02_04/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── homography_demo.cpp            # Homography estimation and decomposition
    ├── hf_model_selection.cpp         # H/F model selection (OpenCV)
    ├── hf_model_selection_poselib.cpp # H/F model selection (PoseLib)
    ├── image_stitching.cpp            # Image stitching / panorama (OpenCV)
    └── image_stitching_poselib.cpp    # Image stitching / panorama (PoseLib)
```

---

## Build

Dependencies:
- **OpenCV 4.x** (`core`, `imgproc`, `imgcodecs`, `features2d`, `calib3d`, `highgui`) and **Eigen3** — required.
- **PoseLib** — optional. `hf_model_selection_poselib` / `image_stitching_poselib` are built only when PoseLib is found (ships in `slam:base`).

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch02_04
```

---

## Run

### Local

```bash
# Homography estimation and decomposition
./build/homography_demo

# H/F model selection (OpenCV)
./build/hf_model_selection

# H/F model selection (PoseLib)
./build/hf_model_selection_poselib

# Image stitching (OpenCV)
./build/image_stitching [image1.jpg image2.jpg]

# Image stitching (PoseLib)
./build/image_stitching_poselib [image1.jpg image2.jpg]
```

`image_stitching` and `image_stitching_poselib` accept two optional image paths; the other demos run without arguments.

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:part2_ch02_04
```

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`findHomography`, `getPerspectiveTransform`, `decomposeHomographyMat`, `findFundamentalMat`)
- [PoseLib](https://github.com/PoseLib/PoseLib)
