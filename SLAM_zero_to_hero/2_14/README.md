# Homography and Bird's Eye View (BEV) Projection using OpenCV

This tutorial covers homography estimation and its application to Bird's Eye View (BEV) projection - fundamental techniques in autonomous driving, augmented reality, and image stitching.

---

## Overview

A **homography** is a projective transformation that maps points from one plane to another. It's represented by a 3x3 matrix and is essential for:

- **Image Stitching**: Aligning overlapping images for panoramas
- **Augmented Reality**: Projecting virtual content onto planar surfaces
- **Bird's Eye View**: Transforming road scenes for lane detection
- **Document Scanning**: Rectifying perspective distortion
- **Planar Object Tracking**: Following markers and patterns

---

## Mathematical Foundation

### The Homography Matrix

A homography H maps points between two images of the same plane:

```
[x']   [h11 h12 h13] [x]
[y'] = [h21 h22 h23] [y]
[w']   [h31 h32 h33] [1]

Final coordinates: (x'/w', y'/w')
```

### Properties

| Property | Value |
|----------|-------|
| Size | 3x3 matrix |
| DoF | 8 (9 elements - 1 scale) |
| Minimum points | 4 (each gives 2 equations) |
| Preserves | Lines (collinearity) |
| Does not preserve | Angles, lengths, ratios |

### Homography Decomposition

A homography between calibrated cameras can be decomposed into rotation, translation, and plane normal:

```
H = K * (R - t*n^T/d) * K^(-1)
```

Where:
- **K**: Camera intrinsic matrix
- **R**: Rotation matrix
- **t**: Translation vector
- **n**: Plane normal
- **d**: Distance to plane

---

## Types of Planar Transformations

```
Euclidean ⊂ Similarity ⊂ Affine ⊂ Projective (Homography)
```

| Transform | DoF | Preserves | Matrix Form |
|-----------|-----|-----------|-------------|
| Euclidean | 3 | Distances, angles | [R, t; 0, 1] |
| Similarity | 4 | Angles, ratios | [sR, t; 0, 1] |
| Affine | 6 | Parallelism | [A, t; 0, 1] |
| Projective | 8 | Lines | Full 3x3 |

---

## Bird's Eye View (BEV) Projection

### What is BEV?

BEV transforms a camera's perspective view to a top-down view, as if looking from directly above.

```
Perspective View (Camera)          Bird's Eye View (Top-down)
         _____                           _______
        /     \                         |       |
       /       \                        |       |
      /    |    \                       |   |   |
     /     |     \                      |   |   |
    /______|______\                     |___|___|
    (road appears                       (road appears
     to converge)                        parallel)
```

### Inverse Perspective Mapping (IPM)

IPM "inverts" the perspective projection by assuming the road is a flat plane:

```
Ground Plane Assumption:
- Road is planar (Z = 0)
- Camera height and pitch are known
- Ground plane normal: n = [0, 1, 0]^T
```

---

## Project Structure

```
2_14/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── homography_demo.cpp            # Homography estimation and decomposition
    ├── homography_poselib.cpp         # 4-point homography using PoseLib
    ├── bev_projection.cpp             # Bird's eye view transform
    └── bev_lane_detection.cpp         # Lane detection from BEV
```

---

## PoseLib Homography

[PoseLib](https://github.com/PoseLib/PoseLib) provides a minimal 4-point homography solver:

```cpp
#include <PoseLib/PoseLib.h>

// 4-point homography solver (minimal)
std::vector<Eigen::Vector3d> x1(pts1.begin(), pts1.begin() + 4);
std::vector<Eigen::Vector3d> x2(pts2.begin(), pts2.begin() + 4);

Eigen::Matrix3d H;
int num_solutions = poselib::homography_4pt(x1, x2, &H);
```

### Key Differences

| Feature | OpenCV | PoseLib |
|---------|--------|---------|
| Min points | 4 | 4 (minimal) |
| RANSAC | Built-in | Separate |
| Decomposition | decomposeHomographyMat | Manual |
| Input | Pixel coords | Homogeneous (normalized) |

---

## How to Build

### Dependencies
- OpenCV 4.x

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:2_14
```

---

## How to Run

### Local

```bash
# Homography estimation and decomposition (OpenCV)
./build/homography_demo

# 4-point homography using PoseLib
./build/homography_poselib

# BEV projection
./build/bev_projection [driving_image.jpg]

# Lane detection from BEV
./build/bev_lane_detection [driving_image.jpg]
```

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:2_14
```

---

## Code Examples

### 1. Homography Estimation

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

// Method 1: From 4+ point correspondences
std::vector<cv::Point2f> src_points = {
    {100, 100}, {400, 100}, {400, 300}, {100, 300}
};
std::vector<cv::Point2f> dst_points = {
    {120, 80}, {380, 120}, {420, 280}, {80, 320}
};

cv::Mat H = cv::findHomography(src_points, dst_points);
std::cout << "Homography:\n" << H << std::endl;

// Method 2: With RANSAC for robustness
cv::Mat mask;
cv::Mat H_ransac = cv::findHomography(src_points, dst_points, cv::RANSAC, 3.0, mask);

// Method 3: Exact 4-point solution
cv::Mat H_exact = cv::getPerspectiveTransform(src_points, dst_points);
```

### 2. Applying Homography

```cpp
// Warp entire image
cv::Mat warped;
cv::warpPerspective(src_image, warped, H, cv::Size(800, 600));

// Transform individual points
std::vector<cv::Point2f> points_in = {{100, 100}, {200, 150}};
std::vector<cv::Point2f> points_out;
cv::perspectiveTransform(points_in, points_out, H);
```

### 3. Bird's Eye View Projection

```cpp
// Define trapezoidal ROI on road (perspective view)
std::vector<cv::Point2f> src_pts = {
    {200, 720},   // bottom-left
    {595, 450},   // top-left
    {685, 450},   // top-right
    {1080, 720}   // bottom-right
};

// Define rectangular destination (BEV)
std::vector<cv::Point2f> dst_pts = {
    {300, 720},   // bottom-left
    {300, 0},     // top-left
    {980, 0},     // top-right
    {980, 720}    // bottom-right
};

// Compute homography
cv::Mat H = cv::getPerspectiveTransform(src_pts, dst_pts);

// Apply transformation
cv::Mat bev_image;
cv::warpPerspective(input_image, bev_image, H, input_image.size());

// Compute inverse for mapping back
cv::Mat H_inv = cv::getPerspectiveTransform(dst_pts, src_pts);
```

### 4. Homography from Features

```cpp
// Detect and match features
cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
std::vector<cv::KeyPoint> kp1, kp2;
cv::Mat desc1, desc2;
orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

cv::BFMatcher matcher(cv::NORM_HAMMING);
std::vector<cv::DMatch> matches;
matcher.match(desc1, desc2, matches);

// Extract points
std::vector<cv::Point2f> pts1, pts2;
for (const auto& m : matches) {
    pts1.push_back(kp1[m.queryIdx].pt);
    pts2.push_back(kp2[m.trainIdx].pt);
}

// Estimate homography with RANSAC
cv::Mat mask;
cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);

// Count inliers
int inliers = cv::countNonZero(mask);
std::cout << "Inliers: " << inliers << "/" << pts1.size() << std::endl;
```

### 5. Homography Decomposition

```cpp
// Camera intrinsics
cv::Mat K = (cv::Mat_<double>(3,3) <<
    718.856, 0, 607.19,
    0, 718.856, 185.22,
    0, 0, 1);

// Decompose homography
std::vector<cv::Mat> rotations, translations, normals;
int solutions = cv::decomposeHomographyMat(H, K, rotations, translations, normals);

std::cout << "Number of solutions: " << solutions << std::endl;
for (int i = 0; i < solutions; i++) {
    std::cout << "Solution " << i << ":\n";
    std::cout << "  R:\n" << rotations[i] << std::endl;
    std::cout << "  t: " << translations[i].t() << std::endl;
    std::cout << "  n: " << normals[i].t() << std::endl;
}
```

### 6. Image Stitching (Panorama)

```cpp
// Compute homography from image2 to image1
cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, 3.0);

// Determine output size
std::vector<cv::Point2f> corners = {
    {0, 0}, {(float)img2.cols, 0},
    {(float)img2.cols, (float)img2.rows}, {0, (float)img2.rows}
};
std::vector<cv::Point2f> corners_warped;
cv::perspectiveTransform(corners, corners_warped, H);

// Find bounding box
cv::Rect2f bounds = cv::boundingRect(corners_warped);
int width = std::max(img1.cols, (int)(bounds.x + bounds.width));
int height = std::max(img1.rows, (int)(bounds.y + bounds.height));

// Warp image2 and blend
cv::Mat result = cv::Mat::zeros(height, width, img1.type());
img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));

cv::Mat warped;
cv::warpPerspective(img2, warped, H, cv::Size(width, height));

// Simple blending (can use more sophisticated methods)
cv::Mat mask = (warped != 0);
warped.copyTo(result, mask);
```

---

## Homography vs Fundamental Matrix

When should you use homography vs fundamental matrix?

| Scenario | Use | Reason |
|----------|-----|--------|
| Planar scene | Homography | Exact transformation |
| 3D scene | Fundamental Matrix | General epipolar geometry |
| Pure rotation | Homography | F is degenerate |
| AR on markers | Homography | Markers are planar |
| SLAM initialization | Both | Compare models |

### Model Selection (like ORB-SLAM)

```cpp
// Compute both models in parallel
cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask_H);
cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, mask_F);

// Compute reprojection errors
double score_H = computeSymmetricTransferError(pts1, pts2, H, mask_H);
double score_F = computeEpipolarError(pts1, pts2, F, mask_F);

// Select model
double ratio = score_H / (score_H + score_F);
if (ratio > 0.45) {
    // Planar scene - use homography
    // Initialize from homography decomposition
} else {
    // 3D scene - use essential matrix
    // Initialize from E decomposition
}
```

---

## BEV Applications in Autonomous Driving

### 1. Lane Detection

```
Perspective View         BEV
    \_____/         |_____|_____|
     \   /          |     |     |
      \_/           |     |     |

Curved lanes appear   Lanes are parallel
to converge           and easier to detect
```

### 2. Path Planning

- Distances are metric in BEV (if calibrated)
- Easier to plan trajectories
- Direct interface with LiDAR-based maps

### 3. Parking Assistance

- Accurate space measurement
- Easy visualization for drivers
- Simple geometric calculations

---

## BEV in Modern SLAM Systems

Many modern perception systems use BEV representations:

| System | Description |
|--------|-------------|
| **Tesla FSD** | Multi-camera BEV fusion |
| **BEVFormer** | Transformer-based BEV generation |
| **HDMapNet** | BEV for HD map construction |
| **LSS (Lift-Splat-Shoot)** | Learned depth + BEV projection |

---

## Calibration for Accurate BEV

For accurate metric BEV, you need:

1. **Camera intrinsics** (fx, fy, cx, cy)
2. **Camera extrinsics** (height h, pitch θ, roll φ, yaw ψ)
3. **Ground plane model** (assumed flat or measured)

```cpp
// Compute homography from camera parameters
cv::Mat computeIPM_Homography(double h, double theta,
                               double fx, double fy,
                               double cx, double cy,
                               double output_scale) {
    // Rotation matrix for camera pitch
    cv::Mat R = (cv::Mat_<double>(3,3) <<
        1, 0, 0,
        0, cos(theta), -sin(theta),
        0, sin(theta), cos(theta));

    // Homography for ground plane at height h
    // ... (full derivation in code)

    return H;
}
```

---

## Common Issues

### 1. Inaccurate BEV near Image Edges
- **Cause**: Large perspective distortion
- **Solution**: Use narrower FOV or crop edges

### 2. Objects Appear Stretched
- **Cause**: Non-planar objects (cars, pedestrians)
- **Solution**: Only use BEV for ground plane features

### 3. Wrong Scale
- **Cause**: Incorrect camera height calibration
- **Solution**: Verify extrinsic calibration

---

## References

- Hartley & Zisserman, "Multiple View Geometry", Chapter 4
- [OpenCV getPerspectiveTransform](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
- [OpenCV findHomography](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Bertozzi & Broggi, "GOLD: A Parallel Real-Time Stereo Vision System for Generic Obstacle and Lane Detection"
