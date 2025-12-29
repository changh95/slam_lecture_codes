# Optical Flow using OpenCV

This tutorial demonstrates optical flow computation using OpenCV, covering both **sparse** (Lucas-Kanade) and **dense** (Farneback) methods. Optical flow is a fundamental technique in visual odometry and SLAM for tracking features across frames and estimating camera motion.

## What is Optical Flow?

Optical flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by relative motion between the observer (camera) and the scene. It represents the displacement of each pixel (or selected features) between consecutive frames.

### Mathematical Foundation

The optical flow constraint equation is derived from the **brightness constancy assumption**:

```
I(x, y, t) = I(x + dx, y + dy, t + dt)
```

Where:
- `I(x, y, t)` is the pixel intensity at position (x, y) at time t
- `dx, dy` are the displacements in x and y directions
- `dt` is the time interval between frames

Using Taylor series expansion and assuming small motion:

```
Ix * u + Iy * v + It = 0
```

Where:
- `Ix, Iy` are spatial derivatives (gradients)
- `It` is the temporal derivative
- `u = dx/dt` and `v = dy/dt` are the optical flow velocities

This is one equation with two unknowns (u, v), making it **under-determined**. Different optical flow methods resolve this differently.

---

## Sparse Optical Flow: Lucas-Kanade Method

### Overview

The Lucas-Kanade method assumes that:
1. **Brightness constancy**: Pixel intensities don't change between frames
2. **Spatial coherence**: Neighboring pixels have similar motion (flow is locally constant)
3. **Small motion**: Movement between frames is small

### Algorithm

For a window of `n` pixels around each feature point, we create an over-determined system:

```
| Ix1  Iy1 |       | -It1 |
| Ix2  Iy2 |       | -It2 |
| ...  ... | [u] = | ...  |
| Ixn  Iyn | [v]   | -Itn |
```

This is solved using least squares:

```
[u, v]^T = (A^T * A)^-1 * A^T * b
```

### Pyramidal Lucas-Kanade

To handle larger motions, OpenCV uses a **pyramidal** (multi-scale) approach:

1. Build image pyramids (downsampled versions)
2. Compute optical flow at the coarsest level
3. Propagate and refine at finer levels

This allows tracking of large displacements while maintaining accuracy.

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `winSize` | Window size for local neighborhood | 21x21 or 31x31 |
| `maxLevel` | Number of pyramid levels (0 = no pyramid) | 3 |
| `criteria` | Termination criteria (iterations/epsilon) | 30 iterations, epsilon=0.01 |
| `flags` | Algorithm flags (use previous flow, etc.) | 0 |
| `minEigThreshold` | Minimum eigenvalue for valid tracking | 0.0001 |

### Advantages
- Computationally efficient (only computes flow at feature points)
- Well-suited for feature tracking in SLAM/VO
- Provides status flags for tracking success/failure

### Disadvantages
- Only sparse flow (at selected points)
- Sensitive to large motions (mitigated by pyramids)
- Can lose track of features over time

---

## Dense Optical Flow: Farneback Method

### Overview

The Farneback method computes optical flow for **every pixel** in the image. It uses polynomial expansion to approximate each pixel neighborhood.

### Algorithm

1. **Polynomial Expansion**: Approximate the neighborhood of each pixel as a quadratic polynomial:
   ```
   f(x) ~ x^T * A * x + b^T * x + c
   ```

2. **Displacement Estimation**: If the signal undergoes a translation `d`, the new polynomial coefficients can be related to the original, and `d` can be estimated.

3. **Multi-scale**: Use Gaussian pyramids for coarse-to-fine estimation.

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `pyr_scale` | Scale between pyramid levels | 0.5 |
| `levels` | Number of pyramid levels | 3 |
| `winsize` | Averaging window size | 15 |
| `iterations` | Iterations at each pyramid level | 3 |
| `poly_n` | Size of pixel neighborhood for polynomial | 5 or 7 |
| `poly_sigma` | Gaussian sigma for polynomial smoothing | 1.1 or 1.5 |
| `flags` | Optional flags (e.g., use Gaussian blur) | 0 |

### Advantages
- Dense flow field (motion for every pixel)
- Good for understanding scene motion
- Can detect moving objects

### Disadvantages
- Computationally expensive
- May not be as accurate at feature points as Lucas-Kanade
- Produces noisy results in textureless regions

---

## Applications in SLAM and Visual Odometry

### 1. Feature Tracking for Visual Odometry
```
Frame N -> Detect features (FAST, Shi-Tomasi corners)
           |
           v
Frame N+1 -> Track features using Lucas-Kanade
           |
           v
Compute Essential Matrix from correspondences
           |
           v
Recover camera pose (R, t)
```

### 2. Key Benefits for SLAM
- **Real-time tracking**: Lucas-Kanade is fast enough for real-time VO
- **Motion estimation**: Direct pixel correspondences for pose estimation
- **Feature lifetime**: Track features across multiple frames
- **Loop closure**: Verify candidate matches using optical flow

### 3. Handling Tracking Failures
- **Re-detection**: When tracked features fall below threshold, detect new ones
- **Track quality**: Monitor backward-forward error (track point forward then backward)
- **Geometric constraints**: Use epipolar geometry to filter outliers

---

## How to Build

Dependencies: OpenCV 4.x

### Local build:
```bash
mkdir build && cd build
cmake ..
make -j
```

### Docker build:
```bash
docker build . -t slam_zero_to_hero:2_7
```

## How to Run

### Local:
```bash
# Sparse optical flow (Lucas-Kanade feature tracking)
./build/sparse_optical_flow

# Dense optical flow (Farneback)
./build/dense_optical_flow
```

### Docker:
```bash
docker run -it --rm slam_zero_to_hero:2_7
```

---

## Topics Covered

### 1. Sparse Optical Flow (`sparse_optical_flow.cpp`)
- Shi-Tomasi corner detection for initial features
- Pyramidal Lucas-Kanade tracking
- Track quality assessment (forward-backward error)
- Feature re-detection when tracking quality degrades
- Visual odometry preprocessing pipeline

### 2. Dense Optical Flow (`dense_optical_flow.cpp`)
- Farneback algorithm implementation
- Flow visualization (HSV color coding)
- Motion magnitude and direction analysis
- Performance comparison with sparse methods

---

## Code Examples

### Feature Detection + Lucas-Kanade Tracking
```cpp
// Detect initial features
cv::goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 10);

// Track features to next frame
cv::calcOpticalFlowPyrLK(prev_gray, curr_gray,
                          prev_pts, curr_pts,
                          status, error,
                          cv::Size(21, 21), 3);
```

### Dense Flow Computation
```cpp
cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow,
                              0.5, 3, 15, 3, 5, 1.2, 0);
```

---

## Performance Considerations

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Lucas-Kanade | Fast | Good at features | Visual Odometry, Feature Tracking |
| Farneback | Slow | Dense coverage | Motion Analysis, Object Detection |

For real-time SLAM applications, Lucas-Kanade is typically preferred due to its efficiency when tracking hundreds of features per frame.

---

## References

- Lucas, B.D., Kanade, T. (1981). "An Iterative Image Registration Technique with an Application to Stereo Vision"
- Farneback, G. (2003). "Two-Frame Motion Estimation Based on Polynomial Expansion"
- Bouguet, J.Y. (2000). "Pyramidal Implementation of the Lucas Kanade Feature Tracker"
- Scaramuzza, D., Fraundorfer, F. (2011). "Visual Odometry: Part I" - IEEE Robotics & Automation Magazine
