# PnP and Fiducial Marker Tracking

Code exercise for estimating camera pose from 2D-3D point correspondences (PnP) and detecting/tracking ArUco/ChArUco fiducial markers with OpenCV — plus P3P minimal solvers from PoseLib and OpenGV.

---

## Project Structure

```
part2_ch02_09/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── markers/                    # Generated marker images
└── examples/
    ├── pnp_demo.cpp                # Compare PnP algorithms (P3P, EPnP, DLS, SQPnP)
    ├── pnp_ransac_demo.cpp         # Robust PnP with RANSAC outlier rejection
    ├── pnp_poselib.cpp             # P3P using PoseLib minimal solver (optional)
    ├── pnp_opengv.cpp              # P3P using OpenGV (Kneip, Gao, EPnP) (optional)
    ├── marker_detection.cpp        # ArUco marker detection and generation
    ├── marker_pose_estimation.cpp  # 6-DoF pose from ArUco markers
    ├── charuco_calibration.cpp     # Camera calibration with ChArUco board
    └── ground_truth_collection.cpp # Collect ground-truth poses for SLAM evaluation
```

---

## Build

Dependencies:
- **OpenCV 4.x** (`core`, `imgproc`, `imgcodecs`, `videoio`, `highgui`, `calib3d`, `objdetect`) and **Eigen3** — required. ArUco is part of `objdetect` in OpenCV 4.7+; older installations need `libopencv-contrib-dev`.
- **PoseLib** and **OpenGV** — optional. `pnp_poselib` / `pnp_opengv` are built only when the respective library is found (both ship in `slam:base`).

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch02_09
```

---

## Run

### Local

```bash
# Compare PnP methods (OpenCV)
./build/pnp_demo

# Robust PnP with RANSAC
./build/pnp_ransac_demo

# PoseLib P3P minimal solver (built only if PoseLib found)
./build/pnp_poselib

# OpenGV P3P solvers (built only if OpenGV found)
./build/pnp_opengv

# Generate an ArUco marker image
./build/marker_detection --generate --id 42 --size 200 --output marker_42.png

# Detect markers from a live camera
./build/marker_detection --camera 0

# 6-DoF pose estimation from markers
./build/marker_pose_estimation --camera 0 --calib camera_calib.yaml --size 0.05

# Camera calibration with ChArUco board
./build/charuco_calibration --camera 0 --output calib.yaml

# Generate ChArUco board image
./build/charuco_calibration --generate --cols 5 --rows 7 --board-output charuco_board.png
```

Sample data generated during Docker build:
- `data/marker_0.png`, `data/marker_1.png`, `data/marker_42.png` — ArUco markers
- `data/charuco_board.png` — ChArUco calibration board
- `data/sample_calib.yaml` — sample camera calibration (640×480, f=600px)
- `data/marker_map.txt` — known marker world positions for ground-truth collection

### Docker

```bash
docker run -it --rm \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch02_09
```

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`solvePnP`, `solvePnPRansac`)
- [OpenCV ArUco/objdetect module](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) (`ArucoDetector`, `estimatePoseSingleMarkers`, `CharucoBoard`)
- [PoseLib](https://github.com/PoseLib/PoseLib)
- [OpenGV](https://laurentkneip.github.io/opengv/)
