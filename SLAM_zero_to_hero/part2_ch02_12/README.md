# RANSAC and USAC: Robust Estimation

Code exercise for robust model estimation — homography and fundamental matrix with OpenCV's RANSAC/USAC framework, a custom RANSAC written from scratch, RansacLib's template-based design, and MAGSAC++.

Every demo runs on the **same real data** (ORB correspondences from a EuRoC MAV frame pair, `data/000024.png` → `data/000025.png`) and is scored by the **same metrics**, so the numbers in `results.csv` are directly comparable across estimators. Line fitting uses a shared fixed-seed synthetic point set (OpenCV has no line RANSAC to compare against).

### The image pair

| | |
|---|---|
| Source | EuRoC MAV `MH_01_easy`, `mav0/cam0/data/` (raw, still distorted) |
| `data/000024.png` | `1403636579763555584.png` — frame 0 |
| `data/000025.png` | `1403636584763555584.png` — frame 100 |
| Resolution | 752×480, 8-bit grayscale |
| Baseline | cam0 runs at 20 Hz, so 100 frames = **5.0 s apart** — a wide baseline, *not* consecutive frames |

The filenames follow the KITTI `%06d.png` convention but the content is EuRoC; the
machine-hall scene is fully 3D with no dominant plane. That is what makes the pair
interesting here: a homography is the *wrong* model for it and explains only ~560–650
of the 844 matches, while the fundamental matrix — the right model — keeps ~825. The
two demos are a direct illustration of model mis-specification showing up as a high
outlier ratio.

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
├── data/                      # EuRoC MH_01_easy pair 000024/000025 (5 s apart)
├── results.csv                # benchmark results from the docker image
└── examples/
    ├── ransac_data.h          # shared ORB pipeline, metrics, synthetic line data, viz
    ├── ransac_homography.cpp  # H: 7 OpenCV RANSAC/USAC variants
    ├── ransac_fundamental.cpp # F: 9 OpenCV methods with full USAC configuration
    ├── ransac_custom.cpp      # Line/H/F RANSAC from scratch (Eigen) vs OpenCV RANSAC
    ├── ransac_ransaclib.cpp   # Line/H/F solvers plugged into RansacLib LO-MSAC
    └── ransac_magsac.cpp      # H/F via MAGSAC++ vs OpenCV RANSAC
```

## Fair comparison rules

- **Same data**: one shared ORB + BF-Hamming + ratio-test pipeline (`ransac_data.h`) feeds every H/F estimator.
- **Same metrics**: mean inlier reprojection error (H, px) and mean squared Sampson distance (F) computed by shared code for every method, whichever library produced the model.
- **Same threshold**: 3 px, 0.99 confidence, everywhere. MAGSAC++ is threshold-free, so its inlier count is derived post-hoc with the same 3 px rule.
- **Same build**: all demos compile at Release; libraries are used with realistic settings. The one deliberate exception is `ransac_custom`, which puts a 50-iteration floor under its adaptive stopping rule — see [Inlier rules are not shared](#inlier-rules-are-not-shared-for-f) and the file's own comments.

### Inlier rules are not shared for F

**H is fine.** Every H estimator here, OpenCV's included, selects inliers by forward reprojection error `|H·x₁ − x₂|²`, so H inlier counts are directly comparable. (Verified against OpenCV's `HomographyEstimatorCallback::computeError`.)

**F is not.** Two different residuals are in play at the same nominal 3 px threshold:

| Rule | Formula | Used by |
|------|---------|---------|
| Sampson | `d²/(A+B)` | `ransac_custom`, all `USAC_*` variants, RansacLib, MAGSAC++ (post-hoc remask) |
| max-form symmetric epipolar | `d²/min(A,B)` | `FM_RANSAC`, `FM_LMEDS`, `FM_7POINT` — OpenCV's classic path |

with `d = x₂ᵀFx₁`, `A = |(Fᵀx₂)_xy|²`, `B = |(Fx₁)_xy|²`. Sources: `modules/calib3d/src/fundam.cpp`, `FMEstimatorCallback::computeError` for the classic path; `modules/calib3d/src/usac/estimator.cpp`, `SampsonErrorImpl` for USAC.

Because `min(A,B) ≤ (A+B)/2`, the max-form residual is **always at least twice** Sampson's, so OpenCV's classic F rule is strictly the tighter one at the same threshold. That alone moves the inlier count by tens of points, which is why `FM_RANSAC` reports 739 while every Sampson-scored method lands at 824–828 on this data. **Do not compare an F inlier count across the two groups** — the difference measures the rule, not the estimator.

`ransac_custom` therefore prints every F model under **both** rules, and re-scores OpenCV's own F under the max-form rule as a check that the replication is exact.

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

All executables run without arguments (H/F demos also accept two image paths).

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
