# Local Feature Extraction and Matching using OpenCV

Code exercise for detecting and matching local features (FAST, ORB, SIFT, FAST+TEBLID) with
ANMS-SSC spatial selection, Lowe's ratio test, and RANSAC geometric verification.

---

## Project Structure

```
part2_ch01_03/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── feature_detection.cpp  # FAST, ORB, SIFT detection on synthetic images
    ├── feature_matching.cpp   # Full matching pipeline (ANMS-SSC → ratio test → RANSAC)
    └── feature_profiling.cpp  # ORB vs SIFT vs FAST+TEBLID timing with easy_profiler
```

---

## Build

Dependencies:
- **OpenCV 4.x** (with contrib / `xfeatures2d` for TEBLID) — required for all targets.
- **easy_profiler** — required for `feature_profiling` (included in `slam:base`).

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch01_03
```

---

## Run

### Local

```bash
# FAST, ORB, SIFT detection on synthetic images
./build/feature_detection

# Full matching pipeline (ORB, SIFT, FAST+TEBLID) with optional custom images
./build/feature_matching
./build/feature_matching /path/to/image1.png /path/to/image2.png

# Profiling: ORB vs SIFT vs FAST+TEBLID — saves feature_profiling.prof
./build/feature_profiling
./build/feature_profiling /path/to/image1.png /path/to/image2.png

# View profiling results
profiler_gui feature_profiling.prof
```

`feature_matching` and `feature_profiling` accept two optional image paths; without arguments
they run on built-in synthetic images.

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    slam_zero_to_hero:part2_ch01_03

# Inside container
./feature_detection
./feature_matching
./feature_profiling
```

---

## References

- [OpenCV `features2d` module](https://docs.opencv.org/4.x/da/d9b/group__features2d.html) (FAST, ORB, BFMatcher, FLANN)
- [OpenCV `xfeatures2d` (contrib)](https://docs.opencv.org/4.x/d2/dca/group__xfeatures2d.html) (SIFT, TEBLID)
- [ANMS-Codes reference implementation](https://github.com/BAILOOL/ANMS-Codes)
- [easy_profiler](https://github.com/yse/easy_profiler)
