# Perspective-n-Points (PnP) and Fiducial Marker Tracking using OpenCV

This tutorial covers camera pose estimation from 2D-3D point correspondences using PnP algorithms, as well as fiducial marker tracking for robot localization and ground truth collection.

---

## Overview

The **Perspective-n-Point (PnP)** problem estimates the camera pose (position and orientation) given:
- A set of **n** 3D points in world coordinates
- Their corresponding **2D projections** in the image
- Camera intrinsic matrix **K**

This is fundamental to Visual SLAM for:
- **Relocalization**: Finding pose when tracking is lost
- **Loop closure verification**: Validating geometric consistency
- **Marker-based tracking**: Using fiducial markers for ground truth

---

## Part 1: PnP Algorithms

### Mathematical Formulation

Given:
- 3D points: `P_i = (X_i, Y_i, Z_i)` in world frame
- 2D points: `p_i = (u_i, v_i)` in image frame
- Camera intrinsic matrix **K**

We want to find the rotation **R** and translation **t** such that:

```
s * [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T
```

### PnP Methods in OpenCV

| Method | Min Points | Complexity | Description |
|--------|-----------|------------|-------------|
| `SOLVEPNP_ITERATIVE` | 4 | O(n*k) | Levenberg-Marquardt refinement (default) |
| `SOLVEPNP_P3P` | 3 | O(1) | 3-point minimal solver |
| `SOLVEPNP_AP3P` | 3 | O(1) | Algebraic P3P (faster) |
| `SOLVEPNP_EPNP` | 4 | O(n) | Efficient PnP |
| `SOLVEPNP_DLS` | 4 | O(n) | Direct Least Squares |
| `SOLVEPNP_SQPNP` | 3 | O(n) | Sequential Quadratic Programming |
| `SOLVEPNP_IPPE` | 4 | O(1) | For coplanar points |

### Direct Linear Transform (DLT)

DLT is an algebraic solution requiring at least 6 points:

```
For each point correspondence:
[X Y Z 1 0 0 0 0 -uX -uY -uZ -u] [p1]     [0]
[0 0 0 0 X Y Z 1 -vX -vY -vZ -v] [p2]  =  [0]
                                  [...]
                                  [p12]
```

Solved via SVD, then R and t are extracted from the 3x4 projection matrix.

---

## Part 2: Fiducial Markers

### What are Fiducial Markers?

Fiducial markers are visual patterns designed for easy detection and identification:
- **Unique ID**: Each marker has a distinct binary code
- **6-DoF pose**: Position (x, y, z) and orientation (roll, pitch, yaw)
- **High accuracy**: Sub-centimeter precision when calibrated

### Marker Types

| Type | Description | Library |
|------|-------------|---------|
| **ArUco** | Square binary markers | OpenCV |
| **AprilTag** | Similar to ArUco, better detection | apriltag |
| **ChArUco** | ArUco + Chessboard hybrid | OpenCV |
| **QR Code** | Data encoding (not for pose) | - |

---

## Project Structure

```
2_19/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── markers/                    # Generated marker images
└── examples/
    ├── pnp_demo.cpp                # PnP algorithms comparison
    ├── pnp_ransac_demo.cpp         # Robust PnP with outliers
    ├── marker_detection.cpp        # ArUco marker detection
    ├── marker_pose_estimation.cpp  # 6-DoF pose from markers
    ├── charuco_calibration.cpp     # Camera calibration with ChArUco
    └── ground_truth_collection.cpp # Collect GT for SLAM evaluation
```

---

## How to Build

### Dependencies
- OpenCV 4.x with aruco module (contrib)
- Eigen3

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:2_19
```

---

## How to Run

### PnP Examples

```bash
# Compare PnP methods
./build/pnp_demo

# Robust PnP with RANSAC
./build/pnp_ransac_demo
```

### Marker Examples

```bash
# Generate markers
./build/marker_detection --generate --id 42 --output marker_42.png

# Detect markers from camera
./build/marker_detection --camera 0

# Pose estimation
./build/marker_pose_estimation --camera 0 --calib camera_calib.yaml --size 0.05

# Camera calibration
./build/charuco_calibration --camera 0 --output calib.yaml
```

### Docker

```bash
docker run -it --rm \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:2_19
```

---

## Code Examples

### 1. Basic PnP

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

// 3D points in world frame
std::vector<cv::Point3f> object_points = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}
};

// 2D projections in image
std::vector<cv::Point2f> image_points = {
    {320, 240}, {420, 240}, {420, 340}, {320, 340},
    {330, 230}, {410, 230}
};

// Camera intrinsics
cv::Mat K = (cv::Mat_<double>(3,3) <<
    500, 0, 320,
    0, 500, 240,
    0, 0, 1);
cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

// Solve PnP
cv::Mat rvec, tvec;
bool success = cv::solvePnP(
    object_points, image_points,
    K, dist_coeffs,
    rvec, tvec,
    false,              // Use extrinsic guess
    cv::SOLVEPNP_EPNP   // Method
);

// Convert rotation vector to matrix
cv::Mat R;
cv::Rodrigues(rvec, R);

std::cout << "Rotation:\n" << R << std::endl;
std::cout << "Translation: " << tvec.t() << std::endl;
```

### 2. PnP with RANSAC

```cpp
// Robust estimation with outliers
cv::Mat rvec, tvec, inliers;
bool success = cv::solvePnPRansac(
    object_points, image_points,
    K, dist_coeffs,
    rvec, tvec,
    false,              // Use extrinsic guess
    100,                // Max iterations
    8.0,                // Reprojection error threshold
    0.99,               // Confidence
    inliers,            // Output inlier indices
    cv::SOLVEPNP_EPNP   // Method
);

std::cout << "Inliers: " << inliers.rows << "/" << object_points.size() << std::endl;
```

### 3. ArUco Marker Detection

```cpp
#include <opencv2/aruco.hpp>

// Create detector
cv::aruco::Dictionary dictionary =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::aruco::DetectorParameters params;
cv::aruco::ArucoDetector detector(dictionary, params);

// Detect markers
std::vector<std::vector<cv::Point2f>> corners;
std::vector<int> ids;
detector.detectMarkers(image, corners, ids);

// Draw detected markers
cv::aruco::drawDetectedMarkers(image, corners, ids);
```

### 4. Marker Pose Estimation

```cpp
// Load camera calibration
cv::Mat camera_matrix, dist_coeffs;
cv::FileStorage fs("camera_calib.yaml", cv::FileStorage::READ);
fs["camera_matrix"] >> camera_matrix;
fs["dist_coeffs"] >> dist_coeffs;
fs.release();

// Marker size in meters
float marker_size = 0.05f;

// Estimate poses
std::vector<cv::Vec3d> rvecs, tvecs;
cv::aruco::estimatePoseSingleMarkers(
    corners, marker_size, camera_matrix, dist_coeffs, rvecs, tvecs);

// Draw axes for each marker
for (size_t i = 0; i < ids.size(); i++) {
    cv::drawFrameAxes(image, camera_matrix, dist_coeffs,
                      rvecs[i], tvecs[i], marker_size * 0.5f);

    // Print pose
    std::cout << "Marker " << ids[i] << ":" << std::endl;
    std::cout << "  Position: " << tvecs[i] << std::endl;
    std::cout << "  Rotation: " << rvecs[i] << std::endl;
}
```

### 5. ChArUco Board Calibration

```cpp
// Create ChArUco board
cv::aruco::CharucoBoard board(
    cv::Size(5, 7),     // 5x7 squares
    0.04f,              // Square size (meters)
    0.03f,              // Marker size (meters)
    dictionary
);

// Collect corners from multiple images
std::vector<std::vector<cv::Point2f>> all_corners;
std::vector<std::vector<int>> all_ids;
std::vector<cv::Mat> all_images;

for (const auto& image : calibration_images) {
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int> marker_ids;
    detector.detectMarkers(image, marker_corners, marker_ids);

    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    cv::aruco::interpolateCornersCharuco(
        marker_corners, marker_ids, image, &board,
        charuco_corners, charuco_ids);

    all_corners.push_back(charuco_corners);
    all_ids.push_back(charuco_ids);
    all_images.push_back(image);
}

// Calibrate camera
cv::Mat camera_matrix, dist_coeffs;
double rms = cv::aruco::calibrateCameraCharuco(
    all_corners, all_ids, &board,
    image_size, camera_matrix, dist_coeffs
);

std::cout << "Calibration RMS error: " << rms << std::endl;
```

### 6. Ground Truth Collection

```cpp
struct GroundTruthPose {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    int marker_id;
};

class GroundTruthCollector {
public:
    void processFrame(const cv::Mat& image, double timestamp) {
        // Detect markers
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        detector_.detectMarkers(image, corners, ids);

        if (ids.empty()) return;

        // Estimate poses
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(
            corners, marker_size_, camera_matrix_, dist_coeffs_,
            rvecs, tvecs);

        // Convert to world frame using known marker positions
        for (size_t i = 0; i < ids.size(); i++) {
            if (marker_world_poses_.count(ids[i])) {
                GroundTruthPose gt;
                gt.timestamp = timestamp;
                gt.marker_id = ids[i];

                // Transform: camera pose in world frame
                Eigen::Matrix3d R_marker_cam;
                cv::Mat R_cv;
                cv::Rodrigues(rvecs[i], R_cv);
                cv::cv2eigen(R_cv, R_marker_cam);

                Eigen::Vector3d t_marker_cam(
                    tvecs[i][0], tvecs[i][1], tvecs[i][2]);

                // Camera in marker frame
                Eigen::Matrix3d R_cam_marker = R_marker_cam.transpose();
                Eigen::Vector3d t_cam_marker = -R_cam_marker * t_marker_cam;

                // Camera in world frame
                const auto& marker_pose = marker_world_poses_[ids[i]];
                gt.position = marker_pose.position +
                              marker_pose.orientation * t_cam_marker;
                gt.orientation = marker_pose.orientation *
                                 Eigen::Quaterniond(R_cam_marker);

                ground_truth_.push_back(gt);
            }
        }
    }

    void saveTUM(const std::string& filename) {
        std::ofstream f(filename);
        f << "# timestamp tx ty tz qx qy qz qw\n";
        for (const auto& gt : ground_truth_) {
            f << std::fixed << std::setprecision(6)
              << gt.timestamp << " "
              << gt.position.x() << " "
              << gt.position.y() << " "
              << gt.position.z() << " "
              << gt.orientation.x() << " "
              << gt.orientation.y() << " "
              << gt.orientation.z() << " "
              << gt.orientation.w() << "\n";
        }
    }

private:
    cv::aruco::ArucoDetector detector_;
    cv::Mat camera_matrix_, dist_coeffs_;
    float marker_size_;
    std::map<int, Eigen::Isometry3d> marker_world_poses_;
    std::vector<GroundTruthPose> ground_truth_;
};
```

---

## SLAM Integration

### PnP for Relocalization

```cpp
bool relocalize(const Frame& current_frame,
                const std::vector<MapPoint>& map_points) {
    // Match features to map points
    std::vector<cv::Point3f> object_pts;
    std::vector<cv::Point2f> image_pts;

    for (const auto& match : matches) {
        object_pts.push_back(map_points[match.map_id].position);
        image_pts.push_back(current_frame.keypoints[match.kp_id].pt);
    }

    if (object_pts.size() < 10) return false;

    // Solve PnP with RANSAC
    cv::Mat rvec, tvec, inliers;
    bool success = cv::solvePnPRansac(
        object_pts, image_pts,
        camera_matrix_, dist_coeffs_,
        rvec, tvec,
        false, 100, 8.0, 0.99, inliers,
        cv::SOLVEPNP_EPNP);

    if (!success || inliers.rows < 10) return false;

    // Set camera pose
    current_frame.setPose(rvec, tvec);
    return true;
}
```

---

## PnP Method Comparison

| Method | Accuracy | Speed | Robustness | Use Case |
|--------|----------|-------|------------|----------|
| P3P | Medium | Fast | Low | RANSAC minimal |
| EPnP | High | Fast | Medium | General purpose |
| DLS | High | Medium | Medium | Many points |
| Iterative | Highest | Slow | High | Final refinement |
| SQPnP | High | Fast | High | Best overall |

---

## Tips for Accurate Tracking

1. **Marker Size**: Larger markers = better accuracy at distance
2. **Lighting**: Avoid glare, use matte finish
3. **Calibration**: Use many images (20+) from different angles
4. **Multiple Markers**: Use marker boards for redundancy
5. **Filtering**: Apply temporal filtering for smooth tracking

---

## Output Formats

### TUM Format

```
# timestamp tx ty tz qx qy qz qw
1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
1.033 0.1 0.0 0.0 0.0 0.0 0.0 1.0
```

### KITTI Format

```
r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```

---

## References

- [OpenCV solvePnP Documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV ArUco Tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- Lepetit et al., "EPnP: An Accurate O(n) Solution to the PnP Problem"
- Kneip et al., "A Novel Parametrization of the Perspective-Three-Point Problem"
- [AprilTag](https://april.eecs.umich.edu/software/apriltag)
