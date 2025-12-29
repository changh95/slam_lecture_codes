# Epipolar Geometry: Essential and Fundamental Matrix Estimation

This tutorial covers the mathematical foundations of epipolar geometry and demonstrates Essential and Fundamental matrix estimation using OpenCV - fundamental concepts for Visual SLAM initialization and two-view geometry.

---

## Overview

Epipolar geometry describes the geometric relationship between two views of a scene. It is the foundation for:

- **Visual Odometry**: Estimating camera motion from image correspondences
- **SLAM Initialization**: Building the initial map from two views
- **Stereo Vision**: Rectification and depth estimation
- **Structure from Motion**: 3D reconstruction from multiple images

---

## Theoretical Background

### The Epipolar Constraint

When a 3D point **X** is observed by two cameras, its projections **x** and **x'** in the two images satisfy a fundamental geometric constraint.

```
         X (3D point)
        /|\
       / | \
      /  |  \
     /   |   \
    /    |    \
   O-----+-----O'
 Camera1  |  Camera2
    \     |     /
     \    |    /
      \   |   /
       \  |  /
        \ | /
    x ----e---- x'
   (img1) | (img2)
      epipolar lines
```

**Key Concepts:**
- **Epipole (e, e')**: The projection of one camera center onto the other camera's image plane
- **Epipolar Line**: For a point in one image, the corresponding point in the other image lies on this line
- **Epipolar Plane**: The plane containing the baseline and the 3D point

---

## The Fundamental Matrix (F)

The **Fundamental Matrix F** is a 3x3 matrix that encapsulates the epipolar geometry between two **uncalibrated** cameras.

### Epipolar Constraint

```
x'^T * F * x = 0
```

Where:
- **x** = (u, v, 1)^T is a point in image 1 (homogeneous coordinates)
- **x'** = (u', v', 1)^T is the corresponding point in image 2
- **F** is a 3x3 rank-2 matrix

### Properties of F

| Property | Description |
|----------|-------------|
| Rank | 2 (det(F) = 0) |
| DoF | 7 (9 elements - 1 scale - 1 rank constraint) |
| Epipolar line | l' = F * x (line in image 2 for point x in image 1) |
| Epipoles | F * e = 0, F^T * e' = 0 |

### Geometric Interpretation

```
l' = F * x    // Epipolar line in image 2 corresponding to point x in image 1
l  = F^T * x' // Epipolar line in image 1 corresponding to point x' in image 2
```

---

## The Essential Matrix (E)

The **Essential Matrix E** is a specialized form of F for **calibrated** cameras (known intrinsic parameters).

### Relationship with F

```
E = K'^T * F * K
```

Where **K** and **K'** are the camera intrinsic matrices.

### Epipolar Constraint (Normalized Coordinates)

```
x_n'^T * E * x_n = 0
```

Where x_n and x_n' are normalized image coordinates:
```
x_n = K^(-1) * x
```

### Properties of E

| Property | Description |
|----------|-------------|
| Rank | 2 |
| DoF | 5 (3 rotation + 2 translation direction) |
| Singular Values | Two equal non-zero values, one zero |
| Decomposition | E = [t]_x * R (skew-symmetric of t times R) |

### The 5-DoF Explanation

- **3 DoF**: Rotation (roll, pitch, yaw)
- **2 DoF**: Translation direction (scale is unrecoverable in monocular vision)

---

## Decomposing E into R and t

The Essential matrix can be decomposed using SVD to recover the relative pose:

```
E = U * diag(1, 1, 0) * V^T
```

This gives **four possible solutions**:

```
W = [0 -1 0]     W^T = [0  1 0]
    [1  0 0]           [-1 0 0]
    [0  0 1]           [0  0 1]

Solution 1: R = U * W   * V^T,  t = +U_3
Solution 2: R = U * W   * V^T,  t = -U_3
Solution 3: R = U * W^T * V^T,  t = +U_3
Solution 4: R = U * W^T * V^T,  t = -U_3
```

**Selecting the correct solution**: Only one solution places triangulated points in front of both cameras (positive depth). This is the **cheirality check**.

---

## Estimation Algorithms

### 8-Point Algorithm (for F)

The simplest method, requiring at least 8 point correspondences:

1. Set up linear system from epipolar constraint: `A * vec(F) = 0`
2. Solve using SVD
3. Enforce rank-2 constraint by zeroing smallest singular value

**Normalization is crucial!** Use Hartley normalization:
- Translate points so centroid is at origin
- Scale so average distance from origin is sqrt(2)

### 7-Point Algorithm (for F)

Minimal solver using exactly 7 points:
- Gives up to 3 solutions
- Used in RANSAC for efficiency

### 5-Point Algorithm (for E)

Minimal solver for calibrated cameras:
- Uses polynomial solver
- Gives up to 10 solutions
- Most efficient for RANSAC with calibrated cameras

---

## Project Structure

```
2_12/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    ├── fundamental_estimation.cpp     # F matrix estimation
    ├── essential_estimation.cpp       # E matrix and pose recovery
    ├── epipolar_visualization.cpp     # Visualize epipolar lines
    └── two_view_reconstruction.cpp    # Full pipeline: E -> R,t -> triangulation
```

---

## How to Build

### Dependencies
- OpenCV 4.x
- Eigen3

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:2_12
```

---

## How to Run

### Local

```bash
# Fundamental matrix estimation
./build/fundamental_estimation image1.jpg image2.jpg

# Essential matrix and pose recovery
./build/essential_estimation image1.jpg image2.jpg --focal 718.856 --cx 607.19 --cy 185.22

# Epipolar line visualization
./build/epipolar_visualization image1.jpg image2.jpg

# Two-view reconstruction
./build/two_view_reconstruction image1.jpg image2.jpg -o points3d.ply
```

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:2_12
```

---

## Code Examples

### 1. Fundamental Matrix Estimation

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

// Detect and match features
cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
std::vector<cv::KeyPoint> kp1, kp2;
cv::Mat desc1, desc2;
orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

cv::BFMatcher matcher(cv::NORM_HAMMING);
std::vector<cv::DMatch> matches;
matcher.match(desc1, desc2, matches);

// Extract matched points
std::vector<cv::Point2f> pts1, pts2;
for (const auto& m : matches) {
    pts1.push_back(kp1[m.queryIdx].pt);
    pts2.push_back(kp2[m.trainIdx].pt);
}

// Estimate Fundamental matrix with RANSAC
cv::Mat mask;
cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, mask);

std::cout << "Fundamental Matrix:\n" << F << std::endl;
std::cout << "Inliers: " << cv::countNonZero(mask) << "/" << pts1.size() << std::endl;

// Verify epipolar constraint
for (int i = 0; i < pts1.size(); i++) {
    if (mask.at<uchar>(i)) {
        cv::Mat x1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat error = x2.t() * F * x1;
        // error should be close to 0
    }
}
```

### 2. Essential Matrix and Pose Recovery

```cpp
// Camera intrinsics
double focal = 718.856;
cv::Point2d pp(607.1928, 185.2157);

// Estimate Essential matrix
cv::Mat E, mask;
E = cv::findEssentialMat(pts1, pts2, focal, pp, cv::RANSAC, 0.999, 1.0, mask);

std::cout << "Essential Matrix:\n" << E << std::endl;

// Recover pose (R, t)
cv::Mat R, t;
int inliers = cv::recoverPose(E, pts1, pts2, R, t, focal, pp, mask);

std::cout << "Rotation:\n" << R << std::endl;
std::cout << "Translation:\n" << t << std::endl;
std::cout << "Inliers: " << inliers << std::endl;

// Note: t is a unit vector (scale is unknown in monocular vision)
```

### 3. Computing Epipolar Lines

```cpp
// Compute epipolar lines in image 2 for points in image 1
std::vector<cv::Vec3f> lines2;
cv::computeCorrespondEpilines(pts1, 1, F, lines2);

// Draw epipolar lines
cv::Mat vis2 = img2.clone();
for (size_t i = 0; i < lines2.size(); i++) {
    // Line equation: ax + by + c = 0
    float a = lines2[i][0];
    float b = lines2[i][1];
    float c = lines2[i][2];

    // Draw line from left to right edge
    cv::Point pt1(0, -c / b);
    cv::Point pt2(img2.cols, -(c + a * img2.cols) / b);
    cv::line(vis2, pt1, pt2, cv::Scalar(0, 255, 0), 1);

    // Draw corresponding point
    cv::circle(vis2, pts2[i], 5, cv::Scalar(0, 0, 255), -1);
}
```

### 4. Two-View Triangulation

```cpp
// Build projection matrices
cv::Mat K = (cv::Mat_<double>(3,3) <<
    focal, 0, pp.x,
    0, focal, pp.y,
    0, 0, 1);

// P1 = K * [I | 0]
cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);

// P2 = K * [R | t]
cv::Mat Rt;
cv::hconcat(R, t, Rt);
cv::Mat P2 = K * Rt;

// Triangulate points
cv::Mat points4D;
cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

// Convert from homogeneous coordinates
std::vector<cv::Point3d> points3D;
for (int i = 0; i < points4D.cols; i++) {
    cv::Mat x = points4D.col(i);
    x /= x.at<float>(3);
    points3D.emplace_back(
        x.at<float>(0),
        x.at<float>(1),
        x.at<float>(2)
    );
}
```

---

## RANSAC Methods in OpenCV

| Flag | Description | Use Case |
|------|-------------|----------|
| `FM_RANSAC` | Standard RANSAC | General purpose |
| `FM_LMEDS` | Least Median of Squares | High outlier ratio |
| `FM_8POINT` | 8-point algorithm | Clean data, no outliers |
| `USAC_MAGSAC` | Modern USAC | Best overall |

```cpp
// Using different RANSAC methods
cv::Mat F1 = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.99);
cv::Mat F2 = cv::findFundamentalMat(pts1, pts2, cv::FM_LMEDS);
cv::Mat F3 = cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, 3.0, 0.99);
```

---

## Visual SLAM Initialization Pipeline

```
1. Detect features in two frames (ORB, SIFT, etc.)
            |
            v
2. Match features between frames
            |
            v
3. Estimate Essential Matrix with RANSAC
   - Use cv::findEssentialMat()
   - Filter outliers with mask
            |
            v
4. Recover Pose (R, t)
   - Use cv::recoverPose()
   - Cheirality check built-in
            |
            v
5. Triangulate initial 3D points
   - Use cv::triangulatePoints()
            |
            v
6. Bundle Adjustment
   - Refine R, t, and 3D points
            |
            v
7. Continue SLAM
```

---

## Degenerate Cases

### 1. Coplanar Points (Homography Dominates)

When all points lie on a plane, the fundamental matrix is degenerate:
- Use **homography** instead: `H = K' * (R - t*n^T/d) * K^(-1)`
- ORB-SLAM uses parallel estimation of H and F

### 2. Pure Rotation (No Translation)

When there's no translation between views:
- Essential matrix is undefined (all epipolar lines pass through the same point)
- Cannot triangulate 3D points
- Solution: Ensure sufficient baseline

### 3. Forward Motion

When moving straight forward:
- Epipole is in the center of the image
- Poor triangulation accuracy for central points
- Solution: Combine with rotation or use lateral points

---

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| High reprojection error | Poor matches | Use ratio test, geometric verification |
| Wrong scale | Monocular ambiguity | Use known object size or IMU |
| Multiple solutions | Degenerate geometry | Check scene structure |
| Poor triangulation | Small baseline | Increase camera motion |

---

## Further Reading

1. Hartley & Zisserman, "Multiple View Geometry in Computer Vision", Chapters 9-11
2. Szeliski, "Computer Vision: Algorithms and Applications", Chapter 7
3. Ma et al., "An Invitation to 3-D Vision"
4. Nister, "An Efficient Solution to the Five-Point Relative Pose Problem"

---

## References

- [OpenCV Epipolar Geometry Tutorial](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)
- [OpenCV findFundamentalMat](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Five-Point Algorithm Paper](https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf)
