/**
 * @file feature_tracking_demo.cpp
 * @brief Demonstrates feature detection and KLT optical flow tracking
 *
 * This example shows:
 * - Feature detection using FAST, ORB, and Good Features to Track (GFTT)
 * - Feature tracking using Lucas-Kanade optical flow
 * - Bidirectional tracking for outlier rejection
 *
 * Usage:
 *   ./feature_tracking_demo <image_directory> [num_frames]
 *
 * Example:
 *   ./feature_tracking_demo /path/to/kitti/sequences/00/image_0 100
 */

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

/**
 * @brief Load image sequence from directory
 */
std::vector<std::string> loadImagePaths(const std::string& dir, int max_images = -1) {
    std::vector<std::string> paths;

    if (!fs::exists(dir)) {
        std::cerr << "Directory does not exist: " << dir << std::endl;
        return paths;
    }

    // Collect all image files
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                paths.push_back(entry.path().string());
            }
        }
    }

    // Sort alphabetically (KITTI uses numbered filenames)
    std::sort(paths.begin(), paths.end());

    // Limit number of images
    if (max_images > 0 && paths.size() > static_cast<size_t>(max_images)) {
        paths.resize(max_images);
    }

    return paths;
}

/**
 * @brief Detect features using different methods
 */
enum class DetectorType { FAST, ORB, GFTT };

std::vector<cv::Point2f> detectFeatures(const cv::Mat& gray, DetectorType type,
                                         int max_features = 2000) {
    std::vector<cv::Point2f> points;

    switch (type) {
        case DetectorType::FAST: {
            std::vector<cv::KeyPoint> keypoints;
            cv::FAST(gray, keypoints, 20, true);

            // Sort by response and keep top features
            std::sort(keypoints.begin(), keypoints.end(),
                      [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                          return a.response > b.response;
                      });

            int n = std::min(max_features, static_cast<int>(keypoints.size()));
            for (int i = 0; i < n; i++) {
                points.push_back(keypoints[i].pt);
            }
            break;
        }

        case DetectorType::ORB: {
            auto orb = cv::ORB::create(max_features);
            std::vector<cv::KeyPoint> keypoints;
            orb->detect(gray, keypoints);

            for (const auto& kp : keypoints) {
                points.push_back(kp.pt);
            }
            break;
        }

        case DetectorType::GFTT: {
            cv::goodFeaturesToTrack(
                gray,
                points,
                max_features,
                0.01,      // Quality level
                10,        // Min distance
                cv::noArray(),
                3,         // Block size
                false,     // Use Harris
                0.04       // Harris parameter
            );
            break;
        }
    }

    return points;
}

/**
 * @brief Track features using Lucas-Kanade optical flow
 */
void trackFeatures(const cv::Mat& prev_gray, const cv::Mat& curr_gray,
                   std::vector<cv::Point2f>& prev_pts,
                   std::vector<cv::Point2f>& curr_pts,
                   std::vector<uchar>& status,
                   bool bidirectional = true) {
    if (prev_pts.empty()) {
        return;
    }

    std::vector<float> err;

    // Forward tracking
    cv::calcOpticalFlowPyrLK(
        prev_gray, curr_gray,
        prev_pts, curr_pts,
        status, err,
        cv::Size(21, 21),  // Window size
        3,                  // Pyramid levels
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
    );

    if (bidirectional) {
        // Backward tracking for consistency check
        std::vector<cv::Point2f> back_pts;
        std::vector<uchar> back_status;
        cv::calcOpticalFlowPyrLK(
            curr_gray, prev_gray,
            curr_pts, back_pts,
            back_status, err,
            cv::Size(21, 21), 3
        );

        // Check forward-backward consistency
        const double threshold = 1.0;  // pixels
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
}

/**
 * @brief Draw tracking visualization
 */
cv::Mat drawTracking(const cv::Mat& frame,
                     const std::vector<cv::Point2f>& prev_pts,
                     const std::vector<cv::Point2f>& curr_pts,
                     const std::vector<uchar>& status) {
    cv::Mat vis;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
    } else {
        vis = frame.clone();
    }

    int tracked_count = 0;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && i < curr_pts.size() && i < prev_pts.size()) {
            // Draw motion vector
            cv::line(vis, prev_pts[i], curr_pts[i], cv::Scalar(0, 255, 255), 1);
            // Draw current point
            cv::circle(vis, curr_pts[i], 3, cv::Scalar(0, 255, 0), -1);
            tracked_count++;
        }
    }

    // Add info text
    std::string info = "Tracked: " + std::to_string(tracked_count) + "/" +
                       std::to_string(prev_pts.size());
    cv::putText(vis, info, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

    return vis;
}

/**
 * @brief Compare different detectors
 */
void compareDetectors(const cv::Mat& gray) {
    std::cout << "\n=== Feature Detector Comparison ===" << std::endl;

    // FAST
    auto t1 = std::chrono::high_resolution_clock::now();
    auto fast_pts = detectFeatures(gray, DetectorType::FAST);
    auto t2 = std::chrono::high_resolution_clock::now();
    double fast_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ORB
    t1 = std::chrono::high_resolution_clock::now();
    auto orb_pts = detectFeatures(gray, DetectorType::ORB);
    t2 = std::chrono::high_resolution_clock::now();
    double orb_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // GFTT
    t1 = std::chrono::high_resolution_clock::now();
    auto gftt_pts = detectFeatures(gray, DetectorType::GFTT);
    t2 = std::chrono::high_resolution_clock::now();
    double gftt_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "FAST: " << fast_pts.size() << " features in " << fast_ms << " ms" << std::endl;
    std::cout << "ORB:  " << orb_pts.size() << " features in " << orb_ms << " ms" << std::endl;
    std::cout << "GFTT: " << gftt_pts.size() << " features in " << gftt_ms << " ms" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_directory> [num_frames]" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " /path/to/kitti/sequences/00/image_0 100" << std::endl;
        return 1;
    }

    std::string image_dir = argv[1];
    int max_frames = (argc > 2) ? std::stoi(argv[2]) : -1;

    // Load image paths
    std::vector<std::string> image_paths = loadImagePaths(image_dir, max_frames);
    if (image_paths.empty()) {
        std::cerr << "No images found in: " << image_dir << std::endl;
        return 1;
    }

    std::cout << "Found " << image_paths.size() << " images" << std::endl;

    // Load first image
    cv::Mat prev_frame = cv::imread(image_paths[0], cv::IMREAD_GRAYSCALE);
    if (prev_frame.empty()) {
        std::cerr << "Failed to load: " << image_paths[0] << std::endl;
        return 1;
    }

    // Compare detectors on first frame
    compareDetectors(prev_frame);

    // Detect initial features using GFTT
    std::vector<cv::Point2f> prev_pts = detectFeatures(prev_frame, DetectorType::GFTT);
    std::cout << "\nInitial features: " << prev_pts.size() << std::endl;

    // Create windows
    cv::namedWindow("Feature Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("Feature Tracking", 1200, 400);

    // Processing loop
    int frame_idx = 1;
    double total_tracking_time = 0;

    while (frame_idx < static_cast<int>(image_paths.size())) {
        // Load current frame
        cv::Mat curr_frame = cv::imread(image_paths[frame_idx], cv::IMREAD_GRAYSCALE);
        if (curr_frame.empty()) {
            std::cerr << "Failed to load: " << image_paths[frame_idx] << std::endl;
            break;
        }

        // Track features
        std::vector<cv::Point2f> curr_pts;
        std::vector<uchar> status;

        auto t1 = std::chrono::high_resolution_clock::now();
        trackFeatures(prev_frame, curr_frame, prev_pts, curr_pts, status, true);
        auto t2 = std::chrono::high_resolution_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_tracking_time += track_ms;

        // Count tracked features
        int tracked = 0;
        for (const auto& s : status) {
            if (s) tracked++;
        }

        // Create visualization
        cv::Mat vis = drawTracking(curr_frame, prev_pts, curr_pts, status);

        // Add frame info
        std::string frame_info = "Frame: " + std::to_string(frame_idx) +
                                 " | Time: " + std::to_string(track_ms).substr(0, 5) + " ms";
        cv::putText(vis, frame_info, cv::Point(10, vis.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);

        cv::imshow("Feature Tracking", vis);

        // Filter points by status for next iteration
        std::vector<cv::Point2f> good_pts;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                good_pts.push_back(curr_pts[i]);
            }
        }

        // Re-detect if too few features
        const int min_features = 100;
        if (good_pts.size() < min_features) {
            std::cout << "Re-detecting features at frame " << frame_idx
                      << " (had " << good_pts.size() << ")" << std::endl;
            prev_pts = detectFeatures(curr_frame, DetectorType::GFTT);
        } else {
            prev_pts = good_pts;
        }

        prev_frame = curr_frame.clone();
        frame_idx++;

        // Handle key press
        int key = cv::waitKey(30);
        if (key == 27 || key == 'q') {  // ESC or 'q' to quit
            break;
        } else if (key == ' ') {  // Space to pause
            std::cout << "Paused. Press any key to continue..." << std::endl;
            cv::waitKey(0);
        }
    }

    // Print statistics
    std::cout << "\n=== Statistics ===" << std::endl;
    std::cout << "Processed frames: " << frame_idx << std::endl;
    std::cout << "Average tracking time: "
              << (total_tracking_time / frame_idx) << " ms" << std::endl;

    cv::destroyAllWindows();
    return 0;
}
