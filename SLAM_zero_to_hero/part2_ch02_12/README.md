# RANSAC and USAC: Robust Estimation

Code exercise for robust model estimation — homography and fundamental matrix with OpenCV's RANSAC/USAC framework, a custom RANSAC written from scratch, RansacLib's template-based design, and MAGSAC++.

Every demo runs on the **same real data** (ORB correspondences from a EuRoC MAV frame pair, `data/1403636579763555584.png` → `data/1403636584763555584.png`) and is scored by the **same metrics** — mean inlier reprojection error for H, mean squared Sampson distance for F, computed by shared code (`ransac_data.h`) whichever library produced the model, at a 3 px threshold and 0.99 confidence throughout. Line fitting uses a shared fixed-seed synthetic point set (OpenCV has no line RANSAC to compare against).

## The image pair

| | |
|---|---|
| Source | EuRoC MAV `MH_01_easy`, `mav0/cam0/data/` (raw, still distorted) |
| Frames | `1403636579763555584.png` (frame 0) → `1403636584763555584.png` (frame 100) |
| Resolution | 752×480, 8-bit grayscale |
| Baseline | cam0 runs at 20 Hz, so 100 frames = **5.0 s apart** — a wide baseline, *not* consecutive frames |

The filenames are EuRoC's own nanosecond capture timestamps, kept verbatim so each
frame is traceable back to the dataset. The machine-hall scene is fully 3D with no
dominant plane, and that is what makes the pair interesting here: a homography is the
*wrong* model for it and explains only ~560–650 of the 844 matches, while the
fundamental matrix — the right model — keeps ~825. The two demos are a direct
illustration of model mis-specification showing up as a high outlier ratio.

Reference intrinsics from `mav0/cam0/sensor.yaml` (printed by `ransac_fundamental`,
but not consumed by anything — F is uncalibrated): `fx=458.654, fy=457.296,
cx=367.215, cy=248.375`, radial-tangential distortion with `k1=-0.2834`.

---

## Project Structure

```
part2_ch02_12/
├── README.md
├── CMakeLists.txt
├── Dockerfile                 # also builds MAGSAC++ from source
├── data/                      # EuRoC MH_01_easy cam0 frames 0 and 100 (5 s apart)
├── results.csv                # benchmark results from the docker image
└── examples/
    ├── ransac_data.h          # shared ORB pipeline, metrics, synthetic line data, viz
    ├── ransac_homography.cpp  # H: 8 OpenCV RANSAC/USAC variants
    ├── ransac_fundamental.cpp # F: 10 OpenCV methods with full USAC configuration
    ├── ransac_custom.cpp      # Line/H/F RANSAC from scratch (Eigen) vs OpenCV RANSAC
    ├── ransac_ransaclib.cpp   # Line/H/F solvers plugged into RansacLib LO-MSAC
    └── ransac_magsac.cpp      # H/F via MAGSAC++ vs OpenCV RANSAC
```

---

## Build

Dependencies:
- **OpenCV 4.5+** — required. USAC support requires OpenCV >= 4.5 (detected at configure time).
- **Eigen3** — required. `ransac_custom` does all of its computation in Eigen (OpenCV is only its interface), and `ransac_ransaclib` and `ransac_magsac` need it too.
- **RansacLib** (header-only) — optional. `ransac_ransaclib` is built only when found.
- **MAGSAC** — optional locally; the Dockerfile builds and installs it automatically, so `ransac_magsac` is always available in the image.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch02_12
```

---

## Run

### Local

```bash
# Homography estimation with RANSAC / USAC methods
./build/ransac_homography

# Fundamental matrix estimation with USAC configuration
./build/ransac_fundamental

# Custom RANSAC implementation (line / homography / fundamental)
./build/ransac_custom

# RansacLib template-based RANSAC (line / homography / fundamental)
./build/ransac_ransaclib

# MAGSAC++ threshold-free estimation (homography / fundamental)
./build/ransac_magsac
```

All five run without arguments, loading the bundled pair from `data/`. Each also
accepts a replacement pair as `argv[1] argv[2]` — both paths are required, since a
single argument is ignored and the defaults are used instead.

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch02_12
```

### Visualization

When a display is available (`DISPLAY` set, X11 socket mounted in Docker), the demos
open result windows; press any key to close them. The same visualizations are always
saved as JPEGs next to the executable:

| Demo | Windows / files |
|------|-----------------|
| `ransac_homography` | RANSAC vs USAC_MAGSAC match masks (`homography_ransac_matches.jpg`, `homography_usac_magsac_matches.jpg`) |
| `ransac_fundamental` | USAC_MAGSAC match mask + epipolar lines (`fundamental_usac_magsac_matches.jpg`, `fundamental_epipolar_lines.jpg`) |
| `ransac_custom` | Line fit + custom H/F match masks (`custom_ransac_line.jpg`, `custom_ransac_homography.jpg`, `custom_ransac_fundamental.jpg`) |
| `ransac_ransaclib` | RansacLib H match mask (`ransaclib_h_matches.jpg`) |
| `ransac_magsac` | MAGSAC++ H match mask (`magsac_h_matches.jpg`) |

Match visualizations draw inliers in green and outliers in red. Without a display
(headless run), the windows are skipped and only the JPEGs are written.

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`findHomography`, `findFundamentalMat`, `UsacParams`)
- [OpenCV Homography tutorial](https://docs.opencv.org/4.x/d1/de0/tutorial_homography.html)
- [RansacLib](https://github.com/tsattler/RansacLib)
- [MAGSAC GitHub](https://github.com/danini/magsac)
