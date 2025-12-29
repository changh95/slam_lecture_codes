/**
 * @file monocular_vo.cpp
 * @brief Complete monocular visual odometry pipeline example
 *
 * This example demonstrates a standalone monocular VO implementation:
 * - Feature detection using GFTT (Good Features to Track)
 * - Feature tracking using Lucas-Kanade optical flow
 * - Essential matrix estimation using cv::findEssentialMat
 * - Pose recovery using cv::recoverPose
 * - Trajectory accumulation and visualization
 *
 * This is a self-contained example that doesn't use the library.
 * For the library version, see run_vo_kitti.cpp.
 *
 * Usage:
 *   ./monocular_vo <image_directory> [focal] [cx] [cy]
 *
 * Example with KITTI sequence 00:
 *   ./monocular_vo /path/to/kitti/sequences/00/image_0 718.856 607.1928 185.2157
 */

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iomanip>

namespace fs = std::filesystem;

// ============================================================================
// Camera Parameters (default: KITTI sequence 00)
// ============================================================================
struct CameraParams {
    double focal = 718.856;
    double cx = 607.1928;
    double cy = 185.2157;

    cv::Point2d pp() const { return cv::Point2d(cx, cy); }
};

// ============================================================================
// VO State
// ============================================================================
struct VOState {
    int frame_id = 0;
    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_pts;

    // Accumulated pose (world frame)
    cv::Mat R_world = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_world = cv::Mat::zeros(3, 1, CV_64F);

    // Trajectory history
    std::vector<cv::Point3d> trajectory;
};

// ============================================================================
// Feature Detection
// ============================================================================
void detectFeatures(const cv::Mat& gray, std::vector<cv::Point2f>& pts,
                    int max_features = 2000) {
    pts.clear();
    cv::goodFeaturesToTrack(
        gray,
        pts,
        max_features,
        0.01,       // Quality level
        10,         // Min distance
        cv::noArray(),
        3,          // Block size
        false,      // Use Harris
        0.04        // Harris param
    );
}

// ============================================================================
// Feature Tracking with Bidirectional Consistency Check
// ============================================================================
void trackFeatures(const cv::Mat& prev, const cv::Mat& curr,
                   std::vector<cv::Point2f>& prev_pts,
                   std::vector<cv::Point2f>& curr_pts,
                   std::vector<uchar>& status) {
    if (prev_pts.empty()) return;

    std::vector<float> err;

    // Forward tracking: prev -> curr
    cv::calcOpticalFlowPyrLK(
        prev, curr,
        prev_pts, curr_pts,
        status, err,
        cv::Size(21, 21),
        3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
    );

    // Backward tracking: curr -> prev
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar> back_status;
    cv::calcOpticalFlowPyrLK(
        curr, prev,
        curr_pts, back_pts,
        back_status, err,
        cv::Size(21, 21),
        3
    );

    // Consistency check
    const double threshold = 1.0;
    for (size_t i = 0; i < prev_pts.size(); i++) {
        if (status[i] && back_status[i]) {
            double dist = cv::norm(prev_pts[i] - back_pts[i]);
            if (dist > threshold) {
                status[i] = 0;
            }
        } else {
            status[i] = 0;
        }
    }
}

// ============================================================================
// Motion Estimation
// ============================================================================
bool estimateMotion(const std::vector<cv::Point2f>& prev_pts,
                    const std::vector<cv::Point2f>& curr_pts,
                    const CameraParams& cam,
                    cv::Mat& R, cv::Mat& t) {
    if (prev_pts.size() < 8 || curr_pts.size() < 8) {
        return false;
    }

    cv::Mat E, mask;

    // Estimate Essential matrix using RANSAC
    E = cv::findEssentialMat(
        curr_pts, prev_pts,
        cam.focal,
        cam.pp(),
        cv::RANSAC,
        0.999,      // Confidence
        1.0,        // Threshold
        mask
    );

    if (E.empty()) {
        return false;
    }

    // Recover pose from Essential matrix
    int inliers = cv::recoverPose(E, curr_pts, prev_pts, R, t,
                                   cam.focal, cam.pp(), mask);

    // Check if we have enough inliers
    return inliers > 10;
}

// ============================================================================
// Process Single Frame
// ============================================================================
bool processFrame(const cv::Mat& frame, VOState& state,
                  const CameraParams& cam, double scale = 1.0) {
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }

    // First frame: detect features only
    if (state.frame_id == 0) {
        detectFeatures(gray, state.prev_pts);
        state.prev_gray = gray.clone();
        state.trajectory.push_back(cv::Point3d(0, 0, 0));
        state.frame_id++;
        return true;
    }

    // Track features
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    trackFeatures(state.prev_gray, gray, state.prev_pts, curr_pts, status);

    // Filter by status
    std::vector<cv::Point2f> prev_good, curr_good;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            prev_good.push_back(state.prev_pts[i]);
            curr_good.push_back(curr_pts[i]);
        }
    }

    bool success = false;

    // Estimate motion
    if (prev_good.size() >= 8) {
        cv::Mat R, t;
        if (estimateMotion(prev_good, curr_good, cam, R, t)) {
            // Accumulate pose
            // t_world = t_world + scale * R_world * t
            // R_world = R * R_world
            state.t_world = state.t_world + scale * (state.R_world * t);
            state.R_world = R * state.R_world;
            success = true;
        }
    }

    // Update trajectory
    cv::Point3d pos(
        state.t_world.at<double>(0, 0),
        state.t_world.at<double>(1, 0),
        state.t_world.at<double>(2, 0)
    );
    state.trajectory.push_back(pos);

    // Re-detect if too few features
    const int min_features = 100;
    if (curr_good.size() < min_features) {
        detectFeatures(gray, state.prev_pts);
    } else {
        state.prev_pts = curr_good;
    }

    state.prev_gray = gray.clone();
    state.frame_id++;

    return success;
}

// ============================================================================
// Visualization
// ============================================================================
cv::Mat drawTrajectory(const std::vector<cv::Point3d>& trajectory,
                       int canvas_size = 800) {
    cv::Mat img(canvas_size, canvas_size, CV_8UC3, cv::Scalar(255, 255, 255));
    int center = canvas_size / 2;

    // Auto-compute scale
    double max_extent = 1.0;
    for (const auto& pt : trajectory) {
        max_extent = std::max(max_extent, std::abs(pt.x));
        max_extent = std::max(max_extent, std::abs(pt.z));
    }
    double scale = (canvas_size / 2 - 50) / max_extent;

    // Draw grid
    for (int i = 0; i <= canvas_size; i += 50) {
        cv::line(img, cv::Point(i, 0), cv::Point(i, canvas_size),
                 cv::Scalar(230, 230, 230), 1);
        cv::line(img, cv::Point(0, i), cv::Point(canvas_size, i),
                 cv::Scalar(230, 230, 230), 1);
    }

    // Draw axes
    cv::line(img, cv::Point(center, 0), cv::Point(center, canvas_size),
             cv::Scalar(200, 200, 200), 1);
    cv::line(img, cv::Point(0, center), cv::Point(canvas_size, center),
             cv::Scalar(200, 200, 200), 1);

    // Draw trajectory
    for (size_t i = 1; i < trajectory.size(); i++) {
        cv::Point2i p1(center + static_cast<int>(scale * trajectory[i-1].x),
                       center - static_cast<int>(scale * trajectory[i-1].z));
        cv::Point2i p2(center + static_cast<int>(scale * trajectory[i].x),
                       center - static_cast<int>(scale * trajectory[i].z));
        cv::line(img, p1, p2, cv::Scalar(255, 0, 0), 2);
    }

    // Draw current position
    if (!trajectory.empty()) {
        cv::Point2i curr(center + static_cast<int>(scale * trajectory.back().x),
                         center - static_cast<int>(scale * trajectory.back().z));
        cv::circle(img, curr, 5, cv::Scalar(0, 0, 255), -1);
    }

    // Legend
    cv::putText(img, "Trajectory (X-Z plane)", cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    return img;
}

cv::Mat drawFeatures(const cv::Mat& frame,
                     const std::vector<cv::Point2f>& pts,
                     int frame_id) {
    cv::Mat vis;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
    } else {
        vis = frame.clone();
    }

    for (const auto& pt : pts) {
        cv::circle(vis, pt, 2, cv::Scalar(0, 255, 0), -1);
    }

    std::string info = "Frame: " + std::to_string(frame_id) +
                       " | Features: " + std::to_string(pts.size());
    cv::putText(vis, info, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

    return vis;
}

// ============================================================================
// Load Image Sequence
// ============================================================================
std::vector<std::string> loadImages(const std::string& dir) {
    std::vector<std::string> paths;

    if (!fs::exists(dir)) {
        return paths;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                paths.push_back(entry.path().string());
            }
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Simple Monocular Visual Odometry" << std::endl;
        std::cout << "\nUsage: " << argv[0]
                  << " <image_directory> [focal] [cx] [cy]" << std::endl;
        std::cout << "\nExample (KITTI sequence 00):" << std::endl;
        std::cout << "  " << argv[0]
                  << " /path/to/kitti/sequences/00/image_0 718.856 607.1928 185.2157"
                  << std::endl;
        std::cout << "\nNote: Without ground truth, scale will be unit scale." << std::endl;
        std::cout << "      For proper scale, use run_vo_kitti with poses file." << std::endl;
        return 1;
    }

    // Parse arguments
    std::string image_dir = argv[1];
    CameraParams cam;
    if (argc > 2) cam.focal = std::stod(argv[2]);
    if (argc > 3) cam.cx = std::stod(argv[3]);
    if (argc > 4) cam.cy = std::stod(argv[4]);

    std::cout << "Camera parameters:" << std::endl;
    std::cout << "  Focal length: " << cam.focal << std::endl;
    std::cout << "  Principal point: (" << cam.cx << ", " << cam.cy << ")" << std::endl;

    // Load images
    std::vector<std::string> image_paths = loadImages(image_dir);
    if (image_paths.empty()) {
        std::cerr << "No images found in: " << image_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << image_paths.size() << " images" << std::endl;

    // Initialize VO state
    VOState state;

    // Create windows
    cv::namedWindow("VO - Features", cv::WINDOW_NORMAL);
    cv::namedWindow("VO - Trajectory", cv::WINDOW_NORMAL);
    cv::resizeWindow("VO - Features", 800, 300);
    cv::resizeWindow("VO - Trajectory", 600, 600);

    // Main loop
    for (size_t i = 0; i < image_paths.size(); i++) {
        cv::Mat frame = cv::imread(image_paths[i]);
        if (frame.empty()) {
            std::cerr << "Failed to load: " << image_paths[i] << std::endl;
            continue;
        }

        // Process frame (unit scale since no ground truth)
        bool success = processFrame(frame, state, cam, 1.0);

        // Visualize
        cv::Mat vis_features = drawFeatures(frame, state.prev_pts, state.frame_id);
        cv::Mat vis_trajectory = drawTrajectory(state.trajectory);

        // Add position info
        cv::Point3d pos = state.trajectory.back();
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2)
           << "Position: (" << pos.x << ", " << pos.y << ", " << pos.z << ")";
        cv::putText(vis_features, ss.str(), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        if (!success && state.frame_id > 1) {
            cv::putText(vis_features, "Motion estimation failed", cv::Point(10, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("VO - Features", vis_features);
        cv::imshow("VO - Trajectory", vis_trajectory);

        // Handle key press
        int key = cv::waitKey(30);
        if (key == 27 || key == 'q') break;
        if (key == ' ') cv::waitKey(0);  // Pause

        // Print progress
        if (i % 50 == 0) {
            std::cout << "Frame " << i << "/" << image_paths.size()
                      << " | Features: " << state.prev_pts.size()
                      << " | Position: " << pos << std::endl;
        }
    }

    // Save final trajectory
    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "Processed " << state.frame_id << " frames" << std::endl;
    std::cout << "Final position: " << state.trajectory.back() << std::endl;

    // Save trajectory image
    cv::Mat final_traj = drawTrajectory(state.trajectory);
    cv::imwrite("trajectory.png", final_traj);
    std::cout << "Trajectory saved to trajectory.png" << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
