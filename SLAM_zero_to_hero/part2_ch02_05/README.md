# Simple Monocular Visual Odometry using OpenCV

Code exercise implementing a monocular VO pipeline with KLT feature tracking, Essential matrix estimation, and pose accumulation — evaluated against KITTI ground-truth poses.

---

## Project Structure

```
part2_ch02_05/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── include/
│   └── monocular_vo.hpp          # VO pipeline header
├── src/
│   └── monocular_vo.cpp          # VO pipeline implementation (built as static lib)
└── examples/
    ├── feature_tracking_demo.cpp # Feature detection and KLT tracking demo
    ├── monocular_vo.cpp          # Standalone monocular VO pipeline
    └── run_vo_kitti.cpp          # VO on KITTI dataset with ground-truth scale
```

---

## Build

Dependencies:
- **OpenCV 4.x** (`core`, `imgproc`, `imgcodecs`, `highgui`, `video`, `calib3d`, `features2d`) — required.
- **Eigen3** — optional (used if found; no executables are gated on it).

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build -t slam_zero_to_hero:part2_ch02_05 .
```

---

## Run

### Local

```bash
# Feature detection and KLT tracking demo
./build/feature_tracking_demo <image_dir> [num_frames]

# Standalone monocular VO pipeline
./build/monocular_vo_demo <image_dir> [focal] [cx] [cy]

# VO on KITTI dataset with ground-truth scale recovery
./build/run_vo_kitti <image_dir> <poses_file> [max_frames]
```

Example with KITTI sequence 00:
```bash
./build/run_vo_kitti /data/sequences/00/image_0 /data/poses/00.txt
```

Download KITTI sequences from [KITTI Visual Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and extract to `/data/kitti/`.

### Docker

```bash
docker run -it --rm \
    -v /path/to/kitti:/data \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch02_05 \
    ./run_vo_kitti /data/sequences/00/image_0 /data/poses/00.txt
```

---

## References

- [OpenCV `video` module](https://docs.opencv.org/4.x/d7/df3/group__video.html) (`calcOpticalFlowPyrLK`, `goodFeaturesToTrack`)
- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`findEssentialMat`, `recoverPose`)
- [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
