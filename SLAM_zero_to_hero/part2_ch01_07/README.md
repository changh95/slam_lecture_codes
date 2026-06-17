# Optical Flow using OpenCV

Sparse (Lucas-Kanade) and dense (Farneback) optical flow demos on TUM RGB-D sequences,
with a 4×5 LK parameter sweep and a 4-tile Farneback visualization.

---

## Project Structure

```
part2_ch01_07/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── sparse_optical_flow.cpp  # LK feature tracking with winSize × pyramid-level sweep (20 trackers)
    └── dense_optical_flow.cpp   # Farneback flow: arrows, HSV, warp residual, motion segmentation
```

---

## Build

Dependencies:
- **OpenCV 4.x** (with GUI/GTK support) — required for both executables.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch01_07
```

---

## Run

Both binaries stream frame-by-frame into a `cv::imshow` window. Press **ESC** to stop early;
press any key on the final frame to close.

### Data

Both demos read frames from a TUM RGB-D sequence. Default path (local):
`~/data/tum_rgbd/rgbd_dataset_freiburg1_desk`.
Inside the container the default is `/data/tum_rgbd/rgbd_dataset_freiburg1_desk`.

### Local

```bash
# Sparse optical flow — LK winSize × pyramid-level sweep
./build/sparse_optical_flow [seq_dir] [num_frames]

# Dense optical flow — Farneback 4-tile viewer
./build/dense_optical_flow [seq_dir] [num_frames] [frame_gap]
```

All arguments are optional (defaults apply).

### Docker

```bash
xhost +SI:localuser:root

docker run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/data:/data:ro \
    --net=host \
    slam_zero_to_hero:part2_ch01_07 \
    ./sparse_optical_flow

docker run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/data:/data:ro \
    --net=host \
    slam_zero_to_hero:part2_ch01_07 \
    ./dense_optical_flow
```

---

## References

- [OpenCV `video` module — `calcOpticalFlowPyrLK`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- [OpenCV `video` module — `calcOpticalFlowFarneback`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- [OpenCV `goodFeaturesToTrack`](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html)
