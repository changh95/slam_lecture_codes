# RANSAC and USAC: Robust Estimation

Code exercise for homography and fundamental matrix estimation using OpenCV's RANSAC and USAC framework, plus a custom RANSAC implementation — with optional RansacLib and MAGSAC++ examples.

---

## Project Structure

```
part2_ch02_12/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── ransac_homography.cpp  # Homography estimation with various RANSAC/USAC methods
    ├── ransac_fundamental.cpp # Fundamental matrix estimation with full USAC configuration
    ├── ransac_custom.cpp      # Custom RANSAC implementation from scratch
    ├── ransac_ransaclib.cpp   # Template-based RANSAC via RansacLib (optional)
    └── ransac_magsac.cpp      # Threshold-free estimation via MAGSAC++ (optional)
```

---

## Build

Dependencies:
- **OpenCV 4.5+** — required. USAC support requires OpenCV >= 4.5 (detected at configure time).
- **Eigen3** — required for `ransac_ransaclib` and `ransac_magsac`.
- **RansacLib** (header-only) — optional. `ransac_ransaclib` is built only when found.
- **MAGSAC** — optional. `ransac_magsac` is built only when found (links `graph_cut_ransac`/`gcransac` if present).

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

# Custom RANSAC implementation (line fitting and homography)
./build/ransac_custom

# RansacLib template-based RANSAC (if built)
./build/ransac_ransaclib

# MAGSAC++ threshold-free estimation (if built)
./build/ransac_magsac
```

All executables run without arguments.

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
| `ransac_custom` | Line fit + homography reprojection (`custom_ransac_line.jpg`, `custom_ransac_homography.jpg`) |

Match visualizations draw inliers in green and outliers in red. Without a display
(headless run), the windows are skipped and only the JPEGs are written.

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`findHomography`, `findFundamentalMat`, `UsacParams`)
- [OpenCV Homography tutorial](https://docs.opencv.org/4.x/d1/de0/tutorial_homography.html)
- [RansacLib](https://github.com/tsattler/RansacLib)
- [MAGSAC GitHub](https://github.com/danini/magsac)
