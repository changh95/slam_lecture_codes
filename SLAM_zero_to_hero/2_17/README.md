# Triangulation using OpenCV Library

This tutorial demonstrates triangulation methods for 3D point reconstruction from stereo/multi-view images - a fundamental technique in Visual SLAM and 3D reconstruction.

## Theory

### What is Triangulation?

Triangulation is the process of determining a 3D point's location by finding the intersection of two or more rays from different camera viewpoints. Given:
- Two camera poses (projection matrices P1 and P2)
- Corresponding 2D point observations (x1, x2)

We want to find the 3D point X such that:
```
x1 = P1 * X
x2 = P2 * X
```

### Why Triangulation Matters in SLAM

1. **Map Building**: Create 3D landmarks from 2D feature correspondences
2. **Depth Estimation**: Recover depth information from stereo cameras
3. **Structure from Motion**: Reconstruct 3D scenes from image sequences
4. **Loop Closure Verification**: Validate loop closures using 3D geometry

---

## Triangulation Methods

### 1. Direct Linear Transform (DLT)

The DLT method reformulates triangulation as a linear system. For each camera i with projection matrix Pi and observation xi = (ui, vi):

```
xi x (Pi * X) = 0
```

This cross-product constraint gives us two independent equations per view:
```
ui * (P3^T * X) - (P1^T * X) = 0
vi * (P3^T * X) - (P2^T * X) = 0
```

Where P1^T, P2^T, P3^T are the rows of the projection matrix.

Stacking equations from multiple views:
```
A * X = 0
```

Solved via SVD: X is the right singular vector corresponding to the smallest singular value.

**Pros:**
- Simple and fast
- Closed-form solution
- Works with any number of views

**Cons:**
- Does not minimize geometric error
- Sensitive to noise
- No statistical optimality

### 2. Mid-Point Method

The mid-point method finds the point that minimizes the sum of squared distances to all rays.

For two rays:
- Ray 1: origin O1, direction d1
- Ray 2: origin O2, direction d2

Find the closest point of approach between the two rays. The 3D point is the midpoint of the shortest line segment connecting the rays.

**Algorithm:**
1. Parameterize rays: r1(t) = O1 + t*d1, r2(s) = O2 + s*d2
2. Find t, s that minimize ||r1(t) - r2(s)||^2
3. Return midpoint: (r1(t) + r2(s)) / 2

**Pros:**
- Geometrically intuitive
- Handles parallel rays (returns point at infinity indicator)
- Fast computation

**Cons:**
- Not optimal in image space
- Assumes equal uncertainty in both views

### 3. Optimal Triangulation (Iterative)

Minimizes the reprojection error - the sum of squared distances between:
- Observed 2D points
- Reprojected 3D point

```
min_X sum_i ||xi - project(Pi, X)||^2
```

This is typically solved using:
- Levenberg-Marquardt optimization
- Or the polynomial method by Hartley-Sturm

**Pros:**
- Statistically optimal (MLE under Gaussian noise)
- Minimizes meaningful geometric error

**Cons:**
- Iterative (slower)
- May not converge

---

## Depth from Stereo

For a calibrated stereo pair with known baseline b and focal length f:

```
depth = (f * b) / disparity
```

Where `disparity = x_left - x_right` is the horizontal pixel difference.

This is a special case of triangulation where cameras are rectified (parallel optical axes).

---

## How to Build

### Dependencies
- OpenCV 4.x
- Eigen 3.x

### Local Build
```bash
mkdir build
cd build
cmake ..
make -j
```

### Docker Build
```bash
docker build . -t slam_zero_to_hero:2_17
```

## How to Run

### Local
```bash
# DLT, midpoint, stereo depth methods
./build/triangulation_demo

# OpenGV triangulation methods
./build/triangulation_opengv
```

### Docker
```bash
docker run -it --rm slam_zero_to_hero:2_17
```

---

## OpenGV Triangulation

[OpenGV](https://laurentkneip.github.io/opengv/) provides efficient triangulation methods:

```cpp
#include <opengv/triangulation/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

// Create adapter with bearing vectors and relative pose
opengv::relative_pose::CentralRelativeAdapter adapter(bearings1, bearings2, t12, R12);

// Linear triangulation
opengv::point_t pt = opengv::triangulation::triangulate(adapter, index);

// Optimal L2 triangulation (Lee & Civera)
opengv::point_t pt2 = opengv::triangulation::triangulate2(adapter, index);
```

### Triangulation Methods Comparison

| Method | Library | Description |
|--------|---------|-------------|
| DLT | OpenCV/Custom | Linear, fast, not optimal |
| Mid-point | Custom | Geometrically intuitive |
| triangulate | OpenGV | Linear method |
| triangulate2 | OpenGV | Optimal L2 (Lee & Civera) |

---

## Code Examples

### 1. OpenCV cv::triangulatePoints

OpenCV's built-in DLT triangulation:

```cpp
cv::Mat pts_4d;
cv::triangulatePoints(P0, P1, points_left, points_right, pts_4d);

// Convert from homogeneous coordinates
for (int i = 0; i < pts_4d.cols; i++) {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0);  // Normalize by w
    // x now contains [X, Y, Z, 1]
}
```

### 2. Custom DLT Implementation

Building the linear system from scratch:

```cpp
Eigen::Vector3d triangulate_dlt(
    const Eigen::Matrix<double, 3, 4>& P1,
    const Eigen::Matrix<double, 3, 4>& P2,
    const Eigen::Vector2d& x1,
    const Eigen::Vector2d& x2)
{
    Eigen::Matrix4d A;
    A.row(0) = x1(0) * P1.row(2) - P1.row(0);
    A.row(1) = x1(1) * P1.row(2) - P1.row(1);
    A.row(2) = x2(0) * P2.row(2) - P2.row(0);
    A.row(3) = x2(1) * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    return X.head<3>() / X(3);  // Dehomogenize
}
```

### 3. Mid-Point Method

Geometric ray intersection:

```cpp
Eigen::Vector3d triangulate_midpoint(
    const Eigen::Vector3d& O1, const Eigen::Vector3d& d1,
    const Eigen::Vector3d& O2, const Eigen::Vector3d& d2)
{
    Eigen::Vector3d w0 = O1 - O2;
    double a = d1.dot(d1);
    double b = d1.dot(d2);
    double c = d2.dot(d2);
    double d = d1.dot(w0);
    double e = d2.dot(w0);

    double denom = a * c - b * b;
    double t = (b * e - c * d) / denom;
    double s = (a * e - b * d) / denom;

    Eigen::Vector3d p1 = O1 + t * d1;
    Eigen::Vector3d p2 = O2 + s * d2;

    return (p1 + p2) / 2.0;  // Midpoint
}
```

---

## Topics Covered

1. **Triangulation Theory**
   - Epipolar geometry basics
   - Projection matrices
   - Homogeneous coordinates

2. **Implementation Methods**
   - DLT (Direct Linear Transform)
   - Mid-point method
   - OpenCV cv::triangulatePoints

3. **Stereo Depth Estimation**
   - Disparity computation
   - Depth from baseline and focal length
   - Rectified stereo geometry

4. **Practical Considerations**
   - Handling degenerate cases (parallel rays)
   - Numerical stability
   - Error metrics (reprojection error)

---

## Further Reading

- Hartley & Zisserman, "Multiple View Geometry in Computer Vision", Chapter 12
- OpenCV documentation: [triangulatePoints](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Szeliski, "Computer Vision: Algorithms and Applications", Chapter 7
