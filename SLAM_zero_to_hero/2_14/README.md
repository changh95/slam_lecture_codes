# Homography for Visual SLAM

This tutorial covers homography estimation and its applications in Visual SLAM - focusing on H/F model selection for SLAM initialization and image stitching for panorama creation.

---

## Overview

**Homography** is a projective transformation that maps points from one plane to another. In Visual SLAM, it's essential for:

- **SLAM Initialization**: Deciding between planar (H) vs 3D (F) scene structure
- **Image Stitching**: Creating panoramas from overlapping images
- **AR Marker Tracking**: Estimating pose from planar fiducials
- **Planar Surface Reconstruction**: Detecting and mapping planar structures

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

## Homography vs Essential/Fundamental Matrix

| Scenario | Use | Reason |
|----------|-----|--------|
| Planar scene | Homography | Exact transformation |
| 3D scene | Essential/Fundamental | General epipolar geometry |
| Pure rotation | Homography | E/F degenerate |
| AR markers | Homography | Markers are planar |
| SLAM initialization | Both | Compare models |

---

## H/F Model Selection (ORB-SLAM Style)

Modern SLAM systems compute both H and F in parallel and select based on score ratio:

```cpp
// Compute both models
cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask_H);
cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, mask_F);

// Compute errors
double score_H = computeSymmetricTransferError(pts1, pts2, H, mask_H);
double score_F = computeSampsonError(pts1, pts2, F, mask_F);

// Select model
double ratio = score_H / (score_H + score_F);
if (ratio > 0.45) {
    // Planar scene - use homography initialization
} else {
    // 3D scene - use essential matrix initialization
}
```

---

## Project Structure

```
2_14/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── homography_demo.cpp           # Homography estimation and decomposition
    ├── hf_model_selection.cpp        # H/F model selection (OpenCV)
    ├── hf_model_selection_poselib.cpp # H/F model selection (PoseLib)
    ├── image_stitching.cpp           # Image stitching (OpenCV)
    └── image_stitching_poselib.cpp   # Image stitching (PoseLib)
```

---

## Alternative Libraries: PoseLib

In addition to OpenCV, this exercise includes examples using **PoseLib** for minimal solvers:

### PoseLib Homography

```cpp
#include <PoseLib/PoseLib.h>

// 4-point homography solver (minimal)
std::vector<Eigen::Vector3d> x1(bearings1.begin(), bearings1.begin() + 4);
std::vector<Eigen::Vector3d> x2(bearings2.begin(), bearings2.begin() + 4);

Eigen::Matrix3d H;
int num_solutions = poselib::homography_4pt(x1, x2, &H);
```

### Key Differences

| Feature | OpenCV | PoseLib |
|---------|--------|---------|
| Min points | 4 | 4 (minimal) |
| RANSAC | Built-in | Separate |
| Input | Pixel coords | Bearing vectors |
| Output | cv::Mat | Eigen::Matrix3d |

---

## How to Build

### Dependencies
- OpenCV 4.x
- Eigen3
- PoseLib (optional)

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

// From point correspondences
std::vector<cv::Point2f> pts1 = { ... };
std::vector<cv::Point2f> pts2 = { ... };

// DLT (no RANSAC)
cv::Mat H_dlt = cv::findHomography(pts1, pts2, 0);

// With RANSAC
cv::Mat mask;
cv::Mat H_ransac = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);

// Exact 4-point solution
cv::Mat H_exact = cv::getPerspectiveTransform(pts1, pts2);
```

### 2. Symmetric Transfer Error

```cpp
double computeSymmetricTransferError(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& H) {

    cv::Mat H_inv = H.inv();
    double total_error = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        // Forward: pts1 -> pts2
        cv::Mat p1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_proj = H * p1;
        p2_proj /= p2_proj.at<double>(2);

        double e_fwd = std::pow(p2_proj.at<double>(0) - pts2[i].x, 2) +
                       std::pow(p2_proj.at<double>(1) - pts2[i].y, 2);

        // Backward: pts2 -> pts1
        cv::Mat p2 = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1_proj = H_inv * p2;
        p1_proj /= p1_proj.at<double>(2);

        double e_bwd = std::pow(p1_proj.at<double>(0) - pts1[i].x, 2) +
                       std::pow(p1_proj.at<double>(1) - pts1[i].y, 2);

        total_error += e_fwd + e_bwd;
    }

    return total_error / pts1.size();
}
```

### 3. Sampson Error (for Fundamental Matrix)

```cpp
double computeSampsonError(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& F) {

    double total_error = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat x1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);

        cv::Mat Fx1 = F * x1;
        cv::Mat Ftx2 = F.t() * x2;
        double x2tFx1 = x2.dot(Fx1);

        double error = (x2tFx1 * x2tFx1) /
                       (Fx1.at<double>(0)*Fx1.at<double>(0) +
                        Fx1.at<double>(1)*Fx1.at<double>(1) +
                        Ftx2.at<double>(0)*Ftx2.at<double>(0) +
                        Ftx2.at<double>(1)*Ftx2.at<double>(1));

        total_error += error;
    }

    return total_error / pts1.size();
}
```

### 4. Homography Decomposition

```cpp
// Camera intrinsics
cv::Mat K = (cv::Mat_<double>(3,3) <<
    fx, 0, cx,
    0, fy, cy,
    0, 0, 1);

// Decompose homography
std::vector<cv::Mat> Rs, ts, normals;
int num_solutions = cv::decomposeHomographyMat(H, K, Rs, ts, normals);

// Select valid solution (normal pointing toward camera)
for (int i = 0; i < num_solutions; ++i) {
    if (normals[i].at<double>(2) > 0) {
        // This solution is physically valid
        cv::Mat R = Rs[i];
        cv::Mat t = ts[i];
        cv::Mat n = normals[i];
    }
}
```

### 5. Image Stitching

```cpp
// Detect features
cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
std::vector<cv::KeyPoint> kp1, kp2;
cv::Mat desc1, desc2;
orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

// Match features
cv::BFMatcher matcher(cv::NORM_HAMMING);
std::vector<std::vector<cv::DMatch>> knn_matches;
matcher.knnMatch(desc1, desc2, knn_matches, 2);

// Ratio test
std::vector<cv::Point2f> pts1, pts2;
for (const auto& m : knn_matches) {
    if (m[0].distance < 0.75f * m[1].distance) {
        pts1.push_back(kp1[m[0].queryIdx].pt);
        pts2.push_back(kp2[m[0].trainIdx].pt);
    }
}

// Compute homography (img2 -> img1)
cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, 3.0);

// Warp and blend
cv::Mat panorama;
cv::warpPerspective(img2, panorama, H, cv::Size(img1.cols + img2.cols, img1.rows));
img1.copyTo(panorama(cv::Rect(0, 0, img1.cols, img1.rows)));
```

---

## References

- Hartley & Zisserman, "Multiple View Geometry", Chapter 4
- Mur-Artal et al., "ORB-SLAM: A Versatile and Accurate Monocular SLAM System"
- [OpenCV findHomography](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [PoseLib](https://github.com/PoseLib/PoseLib)
