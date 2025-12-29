# Simple Monocular Visual Odometry using OpenCV

This tutorial demonstrates a complete monocular visual odometry (VO) pipeline using OpenCV. Visual odometry estimates the camera's motion by analyzing changes in images over time - a fundamental component of Visual SLAM.

---

## What is Visual Odometry?

**Visual Odometry (VO)** is the process of estimating the pose (position and orientation) of a camera by analyzing the sequence of images it captures.

### VO vs SLAM

| Aspect | Visual Odometry | Visual SLAM |
|--------|-----------------|-------------|
| Output | Incremental pose estimates | Global map + trajectory |
| Loop closure | No | Yes |
| Drift | Accumulates | Corrected by loops |
| Complexity | Simpler | More complex |
| Use case | Local navigation | Global localization |

---

## Monocular VO Pipeline Overview

```
+-------------------+     +-------------------+     +-------------------+
|   Image Input     | --> | Feature Detection | --> | Feature Tracking  |
| (Frame t, t+1)    |     | (FAST/ORB/GFTT)   |     |  (KLT Optical     |
+-------------------+     +-------------------+     |      Flow)        |
                                                    +-------------------+
                                                             |
                                                             v
+-------------------+     +-------------------+     +-------------------+
|  Pose Update      | <-- |  Pose Recovery    | <-- | Essential Matrix  |
|  (Accumulate R,t) |     |  (recoverPose)    |     |  (findEssentialMat|
+-------------------+     +-------------------+     |   with RANSAC)    |
                                                    +-------------------+
                                                             ^
                                                             |
                                                    +-------------------+
                                                    | Scale Estimation  |
                                                    | (from GT or other |
                                                    |    sensors)       |
                                                    +-------------------+
```

---

## Pipeline Components

### 1. Feature Detection

Features are distinctive points in an image that can be reliably detected and tracked.

| Detector | Description | Speed | Use Case |
|----------|-------------|-------|----------|
| **FAST** | Corner detector | Very fast | Real-time VO |
| **GFTT** | Shi-Tomasi corners | Fast | General tracking |
| **ORB** | Oriented FAST + BRIEF | Fast | Feature matching |

### 2. Feature Tracking (KLT Optical Flow)

Lucas-Kanade optical flow tracks features from one frame to the next:
- Uses image pyramids for handling large motions
- Iterative refinement for sub-pixel accuracy
- Handles feature loss and outliers

### 3. Essential Matrix Estimation

The essential matrix `E` encodes the relative rotation and translation:
- `x2^T * E * x1 = 0` (epipolar constraint)
- Uses RANSAC for robust estimation against outliers
- Requires camera intrinsics (focal length, principal point)

### 4. Pose Recovery

Decompose the essential matrix to get R and t:
- `E = [t]_x * R` (skew-symmetric matrix of t times R)
- Four possible solutions, only one geometrically valid
- OpenCV's `recoverPose` selects the correct solution via triangulation

### 5. Scale Estimation (The Monocular Challenge)

Monocular VO cannot determine absolute scale from images alone:
- **Ground truth**: Use known trajectory (for evaluation)
- **Known objects**: Use objects with known dimensions
- **IMU fusion**: Accelerometer provides metric scale
- **Ground plane**: Assume known camera height

---

## Project Structure

```
2_15/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── include/
│   └── monocular_vo.hpp          # VO pipeline header
├── src/
│   └── monocular_vo.cpp          # VO pipeline implementation
└── examples/
    ├── feature_tracking_demo.cpp # Feature detection/tracking demo
    ├── monocular_vo.cpp          # Standalone monocular VO pipeline
    └── run_vo_kitti.cpp          # Complete VO on KITTI dataset
```

---

## How to Build

### Dependencies
- OpenCV 4.x (core, features2d, calib3d, video, highgui)

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build -t slam_zero_to_hero:2_15 .
```

---

## How to Run

### Download KITTI Dataset

1. Download a sequence from [KITTI Visual Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
2. Extract to a folder (e.g., `/data/kitti/sequences/00/`)

### Run Feature Tracking Demo

```bash
# Local
./build/feature_tracking_demo /path/to/images 100

# Docker
docker run -it --rm \
    -v /path/to/kitti:/data \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:2_15 ./feature_tracking_demo /data/sequences/00/image_0 100
```

### Run Complete VO Pipeline

```bash
# Local
./build/run_vo_kitti /path/to/kitti/sequences/00/image_0 /path/to/kitti/poses/00.txt

# Docker
docker run -it --rm \
    -v /path/to/kitti:/data \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:2_15 ./run_vo_kitti /data/sequences/00/image_0 /data/poses/00.txt
```

---

## Code Examples

### 1. Complete VO Pipeline Class

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>

class MonocularVO {
public:
    MonocularVO(double focal, cv::Point2d pp)
        : focal_(focal), pp_(pp), frame_id_(0) {
        R_total_ = cv::Mat::eye(3, 3, CV_64F);
        t_total_ = cv::Mat::zeros(3, 1, CV_64F);
    }

    void processFrame(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (frame_id_ == 0) {
            // First frame: detect features
            detectFeatures(gray, prev_pts_);
            prev_gray_ = gray.clone();
            frame_id_++;
            return;
        }

        // Track features
        std::vector<cv::Point2f> curr_pts;
        std::vector<uchar> status;
        trackFeatures(prev_gray_, gray, prev_pts_, curr_pts, status);

        // Filter by status
        std::vector<cv::Point2f> prev_good, curr_good;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                prev_good.push_back(prev_pts_[i]);
                curr_good.push_back(curr_pts[i]);
            }
        }

        // Estimate motion
        if (prev_good.size() >= 8) {
            cv::Mat E, R, t, mask;
            E = cv::findEssentialMat(curr_good, prev_good, focal_, pp_,
                                      cv::RANSAC, 0.999, 1.0, mask);
            cv::recoverPose(E, curr_good, prev_good, R, t, focal_, pp_, mask);

            // Accumulate pose (with scale)
            double scale = getScale();  // From ground truth or other source
            t_total_ = t_total_ + scale * (R_total_ * t);
            R_total_ = R * R_total_;
        }

        // Re-detect if too few features
        if (curr_good.size() < 100) {
            detectFeatures(gray, prev_pts_);
        } else {
            prev_pts_ = curr_good;
        }

        prev_gray_ = gray.clone();
        frame_id_++;
    }

    cv::Mat getPosition() const { return t_total_.clone(); }
    cv::Mat getRotation() const { return R_total_.clone(); }

private:
    void detectFeatures(const cv::Mat& gray, std::vector<cv::Point2f>& pts) {
        cv::goodFeaturesToTrack(gray, pts, 2000, 0.01, 10);
    }

    void trackFeatures(const cv::Mat& prev, const cv::Mat& curr,
                       std::vector<cv::Point2f>& prev_pts,
                       std::vector<cv::Point2f>& curr_pts,
                       std::vector<uchar>& status) {
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev, curr, prev_pts, curr_pts,
                                  status, err, cv::Size(21, 21), 3);
    }

    virtual double getScale() { return 1.0; }  // Override for real scale

    double focal_;
    cv::Point2d pp_;
    int frame_id_;
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_;
    cv::Mat R_total_, t_total_;
};
```

### 2. Feature Detection and Tracking

```cpp
// Detect good features to track
std::vector<cv::Point2f> corners;
cv::goodFeaturesToTrack(
    gray,           // Input image
    corners,        // Output corners
    2000,           // Max corners
    0.01,           // Quality level
    10,             // Min distance between corners
    cv::noArray(),  // Mask (optional)
    3,              // Block size
    false,          // Use Harris detector
    0.04            // Harris parameter
);

// Track features using Lucas-Kanade
std::vector<cv::Point2f> next_pts;
std::vector<uchar> status;
std::vector<float> error;

cv::calcOpticalFlowPyrLK(
    prev_gray,              // Previous image
    curr_gray,              // Current image
    prev_pts,               // Previous points
    next_pts,               // Tracked points (output)
    status,                 // Status (1 = found, 0 = lost)
    error,                  // Tracking error
    cv::Size(21, 21),       // Window size
    3,                      // Pyramid levels
    cv::TermCriteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        30, 0.01)           // Termination criteria
);
```

### 3. Essential Matrix and Pose Recovery

```cpp
// Camera intrinsics (KITTI sequence 00)
double focal = 718.856;
cv::Point2d pp(607.1928, 185.2157);

// Essential matrix from point correspondences
cv::Mat E, mask;
E = cv::findEssentialMat(
    curr_pts,       // Points in current frame
    prev_pts,       // Points in previous frame
    focal,          // Focal length
    pp,             // Principal point
    cv::RANSAC,     // RANSAC method
    0.999,          // Confidence
    1.0,            // Threshold
    mask            // Inlier mask
);

// Decompose to R and t (t is unit vector, scale unknown)
cv::Mat R, t;
int inliers = cv::recoverPose(E, curr_pts, prev_pts, R, t, focal, pp, mask);

std::cout << "Rotation:\n" << R << std::endl;
std::cout << "Translation (unit): " << t.t() << std::endl;
std::cout << "Inliers: " << inliers << std::endl;
```

### 4. Scale Recovery from Ground Truth

```cpp
class MonocularVO_KITTI : public MonocularVO {
public:
    MonocularVO_KITTI(double focal, cv::Point2d pp,
                      const std::string& poses_file)
        : MonocularVO(focal, pp) {
        loadGroundTruth(poses_file);
    }

private:
    void loadGroundTruth(const std::string& file) {
        std::ifstream f(file);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream iss(line);
            std::vector<double> pose(12);
            for (int i = 0; i < 12; i++) iss >> pose[i];
            ground_truth_.push_back(cv::Point3d(pose[3], pose[7], pose[11]));
        }
    }

    double getScale() override {
        if (frame_id_ < 2 || frame_id_ >= ground_truth_.size())
            return 1.0;

        cv::Point3d p1 = ground_truth_[frame_id_ - 1];
        cv::Point3d p2 = ground_truth_[frame_id_];

        return cv::norm(p2 - p1);
    }

    std::vector<cv::Point3d> ground_truth_;
};
```

### 5. Visualization

```cpp
void visualizeTrajectory(const std::vector<cv::Point3d>& trajectory,
                          const std::vector<cv::Point3d>& ground_truth) {
    cv::Mat traj_img(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    double scale = 1.0;  // Adjust based on trajectory extent

    for (size_t i = 1; i < trajectory.size(); i++) {
        // Draw estimated trajectory (blue)
        cv::Point2i p1(400 + scale * trajectory[i-1].x,
                       400 - scale * trajectory[i-1].z);
        cv::Point2i p2(400 + scale * trajectory[i].x,
                       400 - scale * trajectory[i].z);
        cv::line(traj_img, p1, p2, cv::Scalar(255, 0, 0), 2);

        // Draw ground truth (green)
        if (i < ground_truth.size()) {
            cv::Point2i g1(400 + scale * ground_truth[i-1].x,
                           400 - scale * ground_truth[i-1].z);
            cv::Point2i g2(400 + scale * ground_truth[i].x,
                           400 - scale * ground_truth[i].z);
            cv::line(traj_img, g1, g2, cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("Trajectory", traj_img);
}
```

---

## Key Concepts

### Camera Intrinsics

```cpp
// KITTI sequence 00 camera parameters
double focal = 718.856;          // Focal length in pixels
cv::Point2d pp(607.1928, 185.2157);  // Principal point

// Camera matrix K
cv::Mat K = (cv::Mat_<double>(3,3) <<
    focal, 0, pp.x,
    0, focal, pp.y,
    0, 0, 1);
```

### Pose Accumulation

```cpp
// Frame-to-frame motion: R_curr, t_curr
// World pose: R_world, t_world

// Update world pose
t_world = t_world + scale * (R_world * t_curr);
R_world = R_curr * R_world;

// Extract position
double x = t_world.at<double>(0);
double y = t_world.at<double>(1);
double z = t_world.at<double>(2);
```

---

## Limitations of Monocular VO

| Limitation | Cause | Mitigation |
|------------|-------|------------|
| Scale ambiguity | Single camera | IMU fusion, known objects |
| Pure rotation | No baseline | Detect and skip |
| Texture-less regions | No features | Use direct methods |
| Motion blur | Fast motion | Higher frame rate |
| Drift | No loop closure | Add SLAM backend |

---

## Performance Tips

1. **Feature distribution**: Use grid-based detection
2. **Outlier rejection**: Use bidirectional tracking
3. **Re-detection**: Detect new features when count drops
4. **Motion model**: Use constant velocity assumption for prediction

```cpp
// Bidirectional tracking for outlier rejection
std::vector<cv::Point2f> back_pts;
cv::calcOpticalFlowPyrLK(curr_gray, prev_gray, curr_pts, back_pts, status, err);

// Check forward-backward consistency
for (size_t i = 0; i < prev_pts.size(); i++) {
    if (status[i] && cv::norm(prev_pts[i] - back_pts[i]) > 1.0) {
        status[i] = 0;  // Reject inconsistent track
    }
}
```

---

## Evaluation Metrics

### Absolute Trajectory Error (ATE)

```cpp
double computeATE(const std::vector<cv::Point3d>& estimated,
                  const std::vector<cv::Point3d>& ground_truth) {
    double sum_sq = 0.0;
    int n = std::min(estimated.size(), ground_truth.size());

    for (int i = 0; i < n; i++) {
        sum_sq += cv::norm(estimated[i] - ground_truth[i]) *
                  cv::norm(estimated[i] - ground_truth[i]);
    }

    return std::sqrt(sum_sq / n);  // RMSE
}
```

### Relative Pose Error (RPE)

Measures local consistency over fixed distances (e.g., 100m).

---

## References

- Nister, D., Naroditsky, O., & Bergen, J. (2004). "Visual odometry". CVPR 2004.
- Scaramuzza, D., & Fraundorfer, F. (2011). "Visual Odometry: Part I & II". IEEE RAM.
- KITTI Vision Benchmark Suite: https://www.cvlibs.net/datasets/kitti/
- [OpenCV Video Analysis](https://docs.opencv.org/4.x/d7/df3/group__video.html)
