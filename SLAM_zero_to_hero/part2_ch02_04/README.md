# Homography for Visual SLAM

Code exercise for homography estimation, H/F model selection (ORB-SLAM style), and image stitching — all on **real images**, using OpenCV and optionally PoseLib.

---

## Data

All demos run on real image pairs shipped in `data/`:

| Pair | Source | Motion / scene | Why it matters |
|------|--------|----------------|----------------|
| `wall_img1.png` / `wall_img3.png` | [Oxford VGG affine dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/) ("wall") | Viewpoint change, **planar brick wall** | A homography is exactly valid; ground-truth H provided (`wall_H1to3p.txt`) |
| `kitti00_fwd_000024.png` / `..._000025.png` | [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) seq 00 | **Forward motion** (~0.9 m, ~0° rotation), 3D street | The classic case where a homography is *invalid*: depth + translation |
| `kitti00_turn_003677.png` / `..._003682.png` | KITTI odometry seq 00 | **Turning** (~21°, ~2.4 m) | Rotation-dominant with a distant scene: stitches into a panorama; ground-truth relative pose provided (`kitti00_turn_poses.txt`, KITTI pose format) |

KITTI seq 00 camera 0 intrinsics (`P0` in `calib.txt`) are hardcoded where needed: `fx = fy = 718.856`, `cx = 607.1928`, `cy = 185.2157`.

---

## Project Structure

```
part2_ch02_04/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/                              # real image pairs (see table above)
└── examples/
    ├── homography_demo.cpp            # Estimation vs GT (wall) + decomposition vs GT poses (KITTI turn)
    ├── hf_model_selection.cpp         # H/F model selection on 3 real pairs (OpenCV)
    ├── hf_model_selection_poselib.cpp # Same experiment with PoseLib minimal solvers
    ├── image_stitching.cpp            # Panorama from the KITTI turn pair (OpenCV)
    └── image_stitching_poselib.cpp    # Same panorama with PoseLib homography_4pt
```

---

## What each demo shows

### `homography_demo` — estimation and decomposition
- **Part 1 (wall, planar):** ORB matches with a deliberately loose ratio test (~10% natural outliers), then H by plain DLT vs RANSAC, both compared against the dataset's ground-truth homography. Expect DLT to be pulled tens of pixels off by the outliers while RANSAC stays sub-pixel. The comparison is visualized two ways: warp blends (`wall_blend_dlt.png` ghosts, `wall_blend.png` is sharp) and a grid-transfer drawing (`wall_grid_error.png`: green = GT, cyan = RANSAC on top of it, red = DLT with its error vectors). Also saves `wall_warped.png`.
- **Part 2 (KITTI turn):** H estimated between two frames of a ~21° turn, then `cv::decomposeHomographyMat` with the real KITTI intrinsics. The recovered rotation is compared against the KITTI ground-truth poses (expect ~19–23° for the plausible solutions vs 21.3° GT). Visualized as the two labeled input frames (`kitti_inputs.png`), inlier/outlier matches (`kitti_matches.png`), and a warp blend (`kitti_blend.png`) — the distant buildings align but the near ground ghosts, showing why H on this non-planar scene is only good for the rotation.

### `hf_model_selection` — ORB-SLAM initialization logic
Fits H and F to the same matches and computes the ORB-SLAM score ratio `R_H = S_H / (S_H + S_F)` with truncated chi-square scores (Mur-Artal et al., TRO 2015, Sec. IV.A):
- **wall** (planar) → `R_H ≈ 0.50` → **H** selected
- **KITTI forward** (3D + translation) → `R_H ≈ 0.42` → **F** selected
- **KITTI turn** → `R_H ≈ 0.33` → **F** selected — rotation alone doesn't make H win; 2.4 m of travel still creates parallax

Note: naive mean-error ratios do **not** work for this decision — symmetric transfer error and epipolar/Sampson error live on different scales. That is why ORB-SLAM uses truncated inlier-based scores; this demo implements them.

### `image_stitching` — panorama and its failure mode
Default input is the KITTI **turn** pair: rotation-dominant motion over a distant scene, so one homography aligns the views into a proper panorama (~40% extra field of view). Then try the **forward** pair to see the failure case — no new field of view (the second frame is a "zoom" of the first) and parallax ghosting on nearby structure:

```bash
./build/image_stitching data/kitti00_fwd_000024.png data/kitti00_fwd_000025.png
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

Run from the chapter root so the default `data/` paths resolve:

```bash
# Homography estimation (vs GT) and decomposition (vs GT poses)
./build/homography_demo

# H/F model selection on the 3 real pairs (OpenCV / PoseLib)
./build/hf_model_selection
./build/hf_model_selection_poselib

# Panorama from the KITTI turn pair (OpenCV / PoseLib)
./build/image_stitching
./build/image_stitching_poselib

# Panorama failure case: forward motion
./build/image_stitching data/kitti00_fwd_000024.png data/kitti00_fwd_000025.png

# Model selection on any pair of your own images
./build/hf_model_selection my_img1.png my_img2.png
```

`homography_demo` and the stitching demos open visualization windows only when `$DISPLAY` is set (press any key in the window to continue); results are always saved as image files regardless (`panorama_result.jpg`, `matches_visualization.jpg`, `panorama_poselib.jpg`, `wall_*.png`, `kitti_*.png`).

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch02_04 \
    ./build/image_stitching
```

---

## References

- [OpenCV `calib3d` module](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) (`findHomography`, `decomposeHomographyMat`, `findFundamentalMat`)
- [PoseLib](https://github.com/PoseLib/PoseLib)
- Mur-Artal, Montiel, Tardós, "ORB-SLAM: A Versatile and Accurate Monocular SLAM System", IEEE TRO 2015 — Sec. IV.A (map initialization / model selection)
- [Oxford VGG affine covariant features dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/) (wall sequence, ground-truth homographies)
- [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (sequence 00, grayscale camera 0)
