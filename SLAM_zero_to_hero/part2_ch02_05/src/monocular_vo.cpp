#include "monocular_vo.hpp"
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================================
// MonocularVO Implementation
// ============================================================================

MonocularVO::MonocularVO(double focal, cv::Point2d pp,
                         int min_features, int max_features)
    : focal_(focal)
    , pp_(pp)
    , min_features_(min_features)
    , max_features_(max_features)
    , frame_id_(0)
    , visualize_(false) {
    // Initialize pose as identity
    R_total_ = cv::Mat::eye(3, 3, CV_64F);
    t_total_ = cv::Mat::zeros(3, 1, CV_64F);

    // Build camera matrix
    K_ = (cv::Mat_<double>(3, 3) <<
        focal_, 0, pp_.x,
        0, focal_, pp_.y,
        0, 0, 1);

    // Add initial position to trajectory
    trajectory_.push_back(cv::Point3d(0, 0, 0));
}

bool MonocularVO::processFrame(const cv::Mat& frame) {
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }

    // First frame: detect features only
    if (frame_id_ == 0) {
        detectFeatures(gray, prev_pts_);
        prev_gray_ = gray.clone();
        frame_id_++;

        if (visualize_) {
            vis_frame_ = frame.clone();
            for (const auto& pt : prev_pts_) {
                cv::circle(vis_frame_, pt, 3, cv::Scalar(0, 255, 0), -1);
            }
        }

        return true;
    }

    // Track features from previous frame
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;

    // Use bidirectional tracking for better outlier rejection
    bidirectionalTracking(prev_gray_, gray, prev_pts_, curr_pts, status);

    // Filter by status
    std::vector<cv::Point2f> prev_good, curr_good;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            prev_good.push_back(prev_pts_[i]);
            curr_good.push_back(curr_pts[i]);
        }
    }

    // Create visualization before modifying points
    if (visualize_) {
        createVisualization(frame, prev_pts_, curr_pts, status);
    }

    // Estimate motion if enough features
    bool pose_estimated = false;
    if (prev_good.size() >= 8) {
        cv::Mat E, R, t, mask;

        // Find essential matrix using RANSAC
        E = cv::findEssentialMat(curr_good, prev_good, focal_, pp_,
                                  cv::RANSAC, 0.999, 1.0, mask);

        if (!E.empty()) {
            // Recover pose from essential matrix
            int inliers = cv::recoverPose(E, curr_good, prev_good, R, t,
                                          focal_, pp_, mask);

            if (inliers > 10) {
                // Get scale (from ground truth or other source)
                double scale = getScale();

                // Check for reasonable motion (reject large jumps)
                double t_norm = cv::norm(t);
                if (t_norm > 0.001 && scale > 0.001 && scale < 100.0) {
                    // Accumulate pose
                    // t_world = t_world + scale * R_world * t_frame
                    // R_world = R_frame * R_world
                    t_total_ = t_total_ + scale * (R_total_ * t);
                    R_total_ = R * R_total_;

                    pose_estimated = true;
                }
            }
        }
    }

    // Update trajectory
    trajectory_.push_back(getPositionPoint());

    // Re-detect features if count is too low
    if (curr_good.size() < static_cast<size_t>(min_features_)) {
        detectFeatures(gray, prev_pts_);
    } else {
        prev_pts_ = curr_good;
    }

    prev_gray_ = gray.clone();
    frame_id_++;

    return pose_estimated;
}

cv::Point3d MonocularVO::getPositionPoint() const {
    return cv::Point3d(
        t_total_.at<double>(0, 0),
        t_total_.at<double>(1, 0),
        t_total_.at<double>(2, 0)
    );
}

void MonocularVO::detectFeatures(const cv::Mat& gray, std::vector<cv::Point2f>& pts) {
    pts.clear();

    // Use Good Features to Track (Shi-Tomasi corners)
    cv::goodFeaturesToTrack(
        gray,           // Input image
        pts,            // Output corners
        max_features_,  // Max corners
        0.01,           // Quality level (minimum eigenvalue ratio)
        10,             // Min distance between corners
        cv::noArray(),  // Mask
        3,              // Block size for corner detection
        false,          // Use Harris detector
        0.04            // Harris parameter (unused if useHarrisDetector=false)
    );
}

void MonocularVO::trackFeatures(const cv::Mat& prev, const cv::Mat& curr,
                                 const std::vector<cv::Point2f>& prev_pts,
                                 std::vector<cv::Point2f>& curr_pts,
                                 std::vector<uchar>& status) {
    if (prev_pts.empty()) {
        return;
    }

    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        prev,                   // Previous image
        curr,                   // Current image
        prev_pts,               // Previous points
        curr_pts,               // Tracked points (output)
        status,                 // Status (1 = found, 0 = lost)
        err,                    // Tracking error
        cv::Size(21, 21),       // Window size
        3,                      // Pyramid levels
        cv::TermCriteria(       // Termination criteria
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            30, 0.01
        )
    );
}

void MonocularVO::bidirectionalTracking(const cv::Mat& prev, const cv::Mat& curr,
                                         std::vector<cv::Point2f>& prev_pts,
                                         std::vector<cv::Point2f>& curr_pts,
                                         std::vector<uchar>& status) {
    if (prev_pts.empty()) {
        return;
    }

    // Forward tracking: prev -> curr
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev, curr, prev_pts, curr_pts, status, err,
                              cv::Size(21, 21), 3);

    // Backward tracking: curr -> prev (for consistency check)
    std::vector<cv::Point2f> back_pts;
    std::vector<uchar> back_status;
    cv::calcOpticalFlowPyrLK(curr, prev, curr_pts, back_pts, back_status, err,
                              cv::Size(21, 21), 3);

    // Check forward-backward consistency
    const double consistency_threshold = 1.0;  // pixels
    for (size_t i = 0; i < prev_pts.size(); i++) {
        if (status[i] && back_status[i]) {
            double dist = cv::norm(prev_pts[i] - back_pts[i]);
            if (dist > consistency_threshold) {
                status[i] = 0;  // Reject inconsistent track
            }
        } else {
            status[i] = 0;
        }
    }
}

double MonocularVO::getScale() {
    // Default: return unit scale
    // Override this in derived class for ground truth or sensor fusion
    return 1.0;
}

void MonocularVO::createVisualization(const cv::Mat& frame,
                                       const std::vector<cv::Point2f>& prev_pts,
                                       const std::vector<cv::Point2f>& curr_pts,
                                       const std::vector<uchar>& status) {
    vis_frame_ = frame.clone();
    if (vis_frame_.channels() == 1) {
        cv::cvtColor(vis_frame_, vis_frame_, cv::COLOR_GRAY2BGR);
    }

    // Draw tracked features
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && i < curr_pts.size()) {
            // Draw current point (green)
            cv::circle(vis_frame_, curr_pts[i], 3, cv::Scalar(0, 255, 0), -1);

            // Draw motion vector (blue line)
            if (i < prev_pts.size()) {
                cv::line(vis_frame_, prev_pts[i], curr_pts[i],
                         cv::Scalar(255, 0, 0), 1);
            }
        }
    }

    // Add text info
    std::string info = "Frame: " + std::to_string(frame_id_) +
                       " | Features: " + std::to_string(prev_pts_.size());
    cv::putText(vis_frame_, info, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

    cv::Point3d pos = getPositionPoint();
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "Pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ")";
    cv::putText(vis_frame_, ss.str(), cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
}

// ============================================================================
// MonocularVO_KITTI Implementation
// ============================================================================

MonocularVO_KITTI::MonocularVO_KITTI(double focal, cv::Point2d pp,
                                       const std::string& poses_file)
    : MonocularVO(focal, pp) {
    loadGroundTruth(poses_file);
}

void MonocularVO_KITTI::loadGroundTruth(const std::string& file) {
    std::ifstream f(file);
    if (!f.is_open()) {
        std::cerr << "Warning: Could not open ground truth file: " << file << std::endl;
        return;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::vector<double> pose(12);
        for (int i = 0; i < 12; i++) {
            iss >> pose[i];
        }

        // KITTI pose format: 3x4 transformation matrix [R|t] stored row-major
        // pose[0-2]: first row of R
        // pose[3]: tx
        // pose[4-6]: second row of R
        // pose[7]: ty
        // pose[8-10]: third row of R
        // pose[11]: tz
        ground_truth_.push_back(cv::Point3d(pose[3], pose[7], pose[11]));
    }

    std::cout << "Loaded " << ground_truth_.size() << " ground truth poses" << std::endl;
}

double MonocularVO_KITTI::getScale() {
    if (frame_id_ < 2 || frame_id_ >= static_cast<int>(ground_truth_.size())) {
        return 1.0;
    }

    // Compute scale from ground truth displacement
    cv::Point3d p1 = ground_truth_[frame_id_ - 1];
    cv::Point3d p2 = ground_truth_[frame_id_];

    double scale = cv::norm(p2 - p1);

    // Sanity check
    if (scale < 0.001 || scale > 10.0) {
        return 1.0;
    }

    return scale;
}

double MonocularVO_KITTI::computeATE() const {
    if (trajectory_.empty() || ground_truth_.empty()) {
        return -1.0;
    }

    double sum_sq = 0.0;
    int n = std::min(trajectory_.size(), ground_truth_.size());

    for (int i = 0; i < n; i++) {
        double dx = trajectory_[i].x - ground_truth_[i].x;
        double dy = trajectory_[i].y - ground_truth_[i].y;
        double dz = trajectory_[i].z - ground_truth_[i].z;
        sum_sq += dx * dx + dy * dy + dz * dz;
    }

    return std::sqrt(sum_sq / n);  // RMSE
}

// ============================================================================
// Utility Functions
// ============================================================================

namespace vo_utils {

cv::Mat drawTrajectory(const std::vector<cv::Point3d>& estimated,
                       const std::vector<cv::Point3d>& ground_truth,
                       double scale) {
    const int canvas_size = 800;
    cv::Mat traj_img(canvas_size, canvas_size, CV_8UC3, cv::Scalar(255, 255, 255));

    int center_x = canvas_size / 2;
    int center_y = canvas_size / 2;

    // Auto-compute scale if not provided
    if (scale <= 0) {
        double max_extent = 1.0;
        for (const auto& pt : estimated) {
            max_extent = std::max(max_extent, std::abs(pt.x));
            max_extent = std::max(max_extent, std::abs(pt.z));
        }
        for (const auto& pt : ground_truth) {
            max_extent = std::max(max_extent, std::abs(pt.x));
            max_extent = std::max(max_extent, std::abs(pt.z));
        }
        scale = (canvas_size / 2 - 50) / max_extent;
    }

    // Draw grid
    cv::Scalar grid_color(230, 230, 230);
    for (int i = 0; i <= canvas_size; i += 50) {
        cv::line(traj_img, cv::Point(i, 0), cv::Point(i, canvas_size), grid_color, 1);
        cv::line(traj_img, cv::Point(0, i), cv::Point(canvas_size, i), grid_color, 1);
    }

    // Draw axes
    cv::line(traj_img, cv::Point(center_x, 0), cv::Point(center_x, canvas_size),
             cv::Scalar(200, 200, 200), 1);
    cv::line(traj_img, cv::Point(0, center_y), cv::Point(canvas_size, center_y),
             cv::Scalar(200, 200, 200), 1);

    // Draw ground truth (green)
    for (size_t i = 1; i < ground_truth.size(); i++) {
        cv::Point2i g1(center_x + static_cast<int>(scale * ground_truth[i - 1].x),
                       center_y - static_cast<int>(scale * ground_truth[i - 1].z));
        cv::Point2i g2(center_x + static_cast<int>(scale * ground_truth[i].x),
                       center_y - static_cast<int>(scale * ground_truth[i].z));
        cv::line(traj_img, g1, g2, cv::Scalar(0, 200, 0), 2);
    }

    // Draw estimated trajectory (blue)
    for (size_t i = 1; i < estimated.size(); i++) {
        cv::Point2i p1(center_x + static_cast<int>(scale * estimated[i - 1].x),
                       center_y - static_cast<int>(scale * estimated[i - 1].z));
        cv::Point2i p2(center_x + static_cast<int>(scale * estimated[i].x),
                       center_y - static_cast<int>(scale * estimated[i].z));
        cv::line(traj_img, p1, p2, cv::Scalar(255, 0, 0), 2);
    }

    // Draw current position marker
    if (!estimated.empty()) {
        cv::Point2i curr(center_x + static_cast<int>(scale * estimated.back().x),
                         center_y - static_cast<int>(scale * estimated.back().z));
        cv::circle(traj_img, curr, 5, cv::Scalar(0, 0, 255), -1);
    }

    // Legend
    cv::putText(traj_img, "Estimated (blue)", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    cv::putText(traj_img, "Ground Truth (green)", cv::Point(10, 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 0), 2);

    return traj_img;
}

cv::Mat drawFeatureTracks(const cv::Mat& frame,
                          const std::vector<cv::Point2f>& prev_pts,
                          const std::vector<cv::Point2f>& curr_pts,
                          const std::vector<uchar>& status) {
    cv::Mat vis = frame.clone();
    if (vis.channels() == 1) {
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < status.size() && i < prev_pts.size() && i < curr_pts.size(); i++) {
        if (status[i]) {
            // Draw previous point (red)
            cv::circle(vis, prev_pts[i], 2, cv::Scalar(0, 0, 255), -1);
            // Draw current point (green)
            cv::circle(vis, curr_pts[i], 3, cv::Scalar(0, 255, 0), -1);
            // Draw motion vector (blue)
            cv::line(vis, prev_pts[i], curr_pts[i], cv::Scalar(255, 0, 0), 1);
        }
    }

    return vis;
}

std::vector<std::string> loadImageSequence(const std::string& dir,
                                            int start_idx,
                                            int end_idx) {
    std::vector<std::string> images;

    // Try to find images with common naming patterns
    std::vector<std::string> patterns = {
        "%06d.png", "%06d.jpg", "%05d.png", "%05d.jpg",
        "%04d.png", "%04d.jpg", "%d.png", "%d.jpg"
    };

    for (const auto& pattern : patterns) {
        char filename[256];
        snprintf(filename, sizeof(filename), pattern.c_str(), start_idx);
        std::string test_path = dir + "/" + filename;

        if (fs::exists(test_path)) {
            // Found the pattern, load all images
            int idx = start_idx;
            while (true) {
                snprintf(filename, sizeof(filename), pattern.c_str(), idx);
                std::string path = dir + "/" + filename;

                if (!fs::exists(path)) break;
                if (end_idx >= 0 && idx > end_idx) break;

                images.push_back(path);
                idx++;
            }
            break;
        }
    }

    // If no pattern matched, try listing directory
    if (images.empty() && fs::exists(dir)) {
        std::vector<std::string> all_files;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                    all_files.push_back(entry.path().string());
                }
            }
        }
        std::sort(all_files.begin(), all_files.end());

        int start = std::max(0, start_idx);
        int end = (end_idx < 0) ? static_cast<int>(all_files.size()) : std::min(end_idx + 1, static_cast<int>(all_files.size()));
        for (int i = start; i < end; i++) {
            images.push_back(all_files[i]);
        }
    }

    return images;
}

}  // namespace vo_utils
