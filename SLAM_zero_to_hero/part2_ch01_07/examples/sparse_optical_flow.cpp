/**
 * Sparse Optical Flow using Lucas-Kanade Method
 *
 * This example demonstrates feature tracking using pyramidal Lucas-Kanade
 * optical flow, which is a fundamental technique for visual odometry and SLAM.
 *
 * Key concepts:
 * - Shi-Tomasi corner detection for initial features
 * - Pyramidal Lucas-Kanade tracking
 * - Forward-backward error checking for track validation
 * - Feature re-detection when tracking quality degrades
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Configuration parameters
struct OpticalFlowConfig {
    // Feature detection parameters
    int max_features = 200;           // Maximum number of features to track
    double quality_level = 0.01;      // Quality level for corner detection
    double min_distance = 10.0;       // Minimum distance between features
    int block_size = 3;               // Block size for corner detection

    // Lucas-Kanade parameters
    cv::Size win_size{21, 21};        // Search window size
    int max_pyramid_level = 3;        // Number of pyramid levels
    int max_iterations = 30;          // Maximum iterations per level
    double epsilon = 0.01;            // Convergence threshold

    // Track quality parameters
    double max_error = 12.0;          // Maximum tracking error threshold
    double fb_threshold = 1.0;        // Forward-backward error threshold
    int min_features = 50;            // Minimum features before re-detection
};

/**
 * Generate synthetic test frames that simulate camera motion
 * This creates a sequence of frames with known motion for testing
 */
class SyntheticSequenceGenerator {
public:
    SyntheticSequenceGenerator(int width, int height, int seed = 42)
        : width_(width), height_(height), rng_(seed) {
        generateBaseImage();
    }

    cv::Mat generateBaseImage() {
        base_image_ = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(128));

        // Add random features (corners, edges)
        for (int i = 0; i < 100; i++) {
            cv::Point center(rng_.uniform(50, width_ - 50),
                           rng_.uniform(50, height_ - 50));
            int size = rng_.uniform(10, 40);
            int intensity = rng_.uniform(0, 255);

            // Draw rectangles (create corner features)
            cv::rectangle(base_image_,
                         center - cv::Point(size/2, size/2),
                         center + cv::Point(size/2, size/2),
                         cv::Scalar(intensity), -1);
        }

        // Add some circles
        for (int i = 0; i < 50; i++) {
            cv::Point center(rng_.uniform(30, width_ - 30),
                           rng_.uniform(30, height_ - 30));
            int radius = rng_.uniform(5, 20);
            int intensity = rng_.uniform(0, 255);
            cv::circle(base_image_, center, radius, cv::Scalar(intensity), -1);
        }

        // Add texture (gradient and noise)
        cv::Mat noise(height_, width_, CV_8UC1);
        rng_.fill(noise, cv::RNG::NORMAL, 0, 15);
        base_image_ += noise;

        // Add a checkerboard pattern in corner for strong features
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 100; x++) {
                if ((x / 10 + y / 10) % 2 == 0) {
                    base_image_.at<uchar>(y, x) = 255;
                } else {
                    base_image_.at<uchar>(y, x) = 0;
                }
            }
        }

        return base_image_;
    }

    /**
     * Get a frame with simulated camera translation
     */
    cv::Mat getTranslatedFrame(double dx, double dy) {
        cv::Mat result;
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
        cv::warpAffine(base_image_, result, M, base_image_.size(),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return result;
    }

    /**
     * Get a frame with simulated camera rotation
     */
    cv::Mat getRotatedFrame(double angle_deg, cv::Point2f center = cv::Point2f(-1, -1)) {
        if (center.x < 0) {
            center = cv::Point2f(width_ / 2.0f, height_ / 2.0f);
        }
        cv::Mat result;
        cv::Mat M = cv::getRotationMatrix2D(center, angle_deg, 1.0);
        cv::warpAffine(base_image_, result, M, base_image_.size(),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return result;
    }

    /**
     * Get a frame with combined motion (translation + rotation)
     */
    cv::Mat getTransformedFrame(double dx, double dy, double angle_deg) {
        cv::Point2f center(width_ / 2.0f, height_ / 2.0f);
        cv::Mat R = cv::getRotationMatrix2D(center, angle_deg, 1.0);
        R.at<double>(0, 2) += dx;
        R.at<double>(1, 2) += dy;

        cv::Mat result;
        cv::warpAffine(base_image_, result, R, base_image_.size(),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return result;
    }

private:
    int width_, height_;
    cv::RNG rng_;
    cv::Mat base_image_;
};

/**
 * Feature tracker using Lucas-Kanade optical flow
 * Implements a robust tracking pipeline suitable for visual odometry
 */
class LKFeatureTracker {
public:
    explicit LKFeatureTracker(const OpticalFlowConfig& config = OpticalFlowConfig())
        : config_(config) {
        // Set up termination criteria
        term_criteria_ = cv::TermCriteria(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            config_.max_iterations,
            config_.epsilon
        );
    }

    /**
     * Detect initial features using Shi-Tomasi corner detector
     */
    void detectFeatures(const cv::Mat& frame) {
        cv::goodFeaturesToTrack(
            frame,
            prev_points_,
            config_.max_features,
            config_.quality_level,
            config_.min_distance,
            cv::Mat(),
            config_.block_size,
            false,
            0.04
        );

        frame.copyTo(prev_frame_);
        track_ids_.clear();
        for (size_t i = 0; i < prev_points_.size(); i++) {
            track_ids_.push_back(next_track_id_++);
        }

        std::cout << "Detected " << prev_points_.size() << " initial features" << std::endl;
    }

    /**
     * Track features to the next frame using pyramidal Lucas-Kanade
     * Returns the number of successfully tracked features
     */
    int trackFeatures(const cv::Mat& curr_frame) {
        if (prev_points_.empty()) {
            std::cerr << "No features to track. Call detectFeatures first." << std::endl;
            return 0;
        }

        std::vector<cv::Point2f> curr_points;
        std::vector<uchar> status;
        std::vector<float> error;

        // Forward tracking: prev -> curr
        cv::calcOpticalFlowPyrLK(
            prev_frame_, curr_frame,
            prev_points_, curr_points,
            status, error,
            config_.win_size,
            config_.max_pyramid_level,
            term_criteria_,
            0,
            0.0001
        );

        // Backward tracking for validation: curr -> prev
        std::vector<cv::Point2f> back_points;
        std::vector<uchar> back_status;
        std::vector<float> back_error;

        cv::calcOpticalFlowPyrLK(
            curr_frame, prev_frame_,
            curr_points, back_points,
            back_status, back_error,
            config_.win_size,
            config_.max_pyramid_level,
            term_criteria_,
            0,
            0.0001
        );

        // Filter tracks based on status, error, and forward-backward consistency
        std::vector<cv::Point2f> good_prev, good_curr;
        std::vector<int> good_ids;
        std::vector<cv::Point2f> flow_vectors;

        for (size_t i = 0; i < curr_points.size(); i++) {
            // Check forward tracking status
            if (status[i] == 0) continue;

            // Check backward tracking status
            if (back_status[i] == 0) continue;

            // Check tracking error
            if (error[i] > config_.max_error) continue;

            // Forward-backward consistency check
            double fb_error = cv::norm(prev_points_[i] - back_points[i]);
            if (fb_error > config_.fb_threshold) continue;

            // Check if point is within image bounds
            if (curr_points[i].x < 0 || curr_points[i].x >= curr_frame.cols ||
                curr_points[i].y < 0 || curr_points[i].y >= curr_frame.rows) {
                continue;
            }

            good_prev.push_back(prev_points_[i]);
            good_curr.push_back(curr_points[i]);
            good_ids.push_back(track_ids_[i]);
            flow_vectors.push_back(curr_points[i] - prev_points_[i]);
        }

        // Store results
        prev_points_ = good_prev;
        curr_points_ = good_curr;
        track_ids_ = good_ids;
        flow_vectors_ = flow_vectors;

        // Update previous frame
        curr_frame.copyTo(prev_frame_);
        prev_points_ = curr_points_;

        int tracked = static_cast<int>(curr_points_.size());

        // Re-detect features if too few remain
        if (tracked < config_.min_features) {
            std::cout << "Low feature count (" << tracked << "), re-detecting..." << std::endl;
            detectAdditionalFeatures(curr_frame);
        }

        return tracked;
    }

    /**
     * Detect additional features while avoiding existing ones
     */
    void detectAdditionalFeatures(const cv::Mat& frame) {
        // Create mask to avoid detecting near existing features
        cv::Mat mask = cv::Mat::ones(frame.size(), CV_8UC1) * 255;
        for (const auto& pt : prev_points_) {
            cv::circle(mask, pt, static_cast<int>(config_.min_distance), cv::Scalar(0), -1);
        }

        // Detect new features
        std::vector<cv::Point2f> new_points;
        int num_to_detect = config_.max_features - static_cast<int>(prev_points_.size());
        if (num_to_detect <= 0) return;

        cv::goodFeaturesToTrack(
            frame,
            new_points,
            num_to_detect,
            config_.quality_level,
            config_.min_distance,
            mask,
            config_.block_size,
            false,
            0.04
        );

        // Add new features
        for (const auto& pt : new_points) {
            prev_points_.push_back(pt);
            track_ids_.push_back(next_track_id_++);
        }

        std::cout << "Added " << new_points.size() << " new features. "
                  << "Total: " << prev_points_.size() << std::endl;
    }

    /**
     * Compute average flow for motion estimation
     */
    cv::Point2f computeAverageFlow() const {
        if (flow_vectors_.empty()) return cv::Point2f(0, 0);

        cv::Point2f avg(0, 0);
        for (const auto& flow : flow_vectors_) {
            avg += flow;
        }
        avg.x /= flow_vectors_.size();
        avg.y /= flow_vectors_.size();
        return avg;
    }

    /**
     * Get current tracked points
     */
    const std::vector<cv::Point2f>& getPreviousPoints() const { return prev_points_; }
    const std::vector<cv::Point2f>& getCurrentPoints() const { return curr_points_; }
    const std::vector<int>& getTrackIds() const { return track_ids_; }
    const std::vector<cv::Point2f>& getFlowVectors() const { return flow_vectors_; }

    /**
     * Visualize tracking results
     */
    cv::Mat visualize(const cv::Mat& frame) const {
        cv::Mat vis;
        if (frame.channels() == 1) {
            cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
        } else {
            frame.copyTo(vis);
        }

        // Draw tracks
        for (size_t i = 0; i < curr_points_.size() && i < prev_points_.size(); i++) {
            // Color based on track ID for consistency
            int hue = (track_ids_[i] * 37) % 180;
            cv::Scalar color = cv::Scalar(hue * 1.4, 255 - hue, 100 + hue % 155);

            // Draw flow line
            cv::line(vis, prev_points_[i], curr_points_[i], color, 2);

            // Draw current point
            cv::circle(vis, curr_points_[i], 4, color, -1);

            // Draw previous point (smaller)
            cv::circle(vis, prev_points_[i], 2, cv::Scalar(100, 100, 100), -1);
        }

        // Add info text
        std::string info = "Tracking " + std::to_string(curr_points_.size()) + " features";
        cv::putText(vis, info, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::Point2f avg_flow = computeAverageFlow();
        std::string flow_info = "Avg flow: (" +
            std::to_string(static_cast<int>(avg_flow.x)) + ", " +
            std::to_string(static_cast<int>(avg_flow.y)) + ")";
        cv::putText(vis, flow_info, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        return vis;
    }

private:
    OpticalFlowConfig config_;
    cv::TermCriteria term_criteria_;

    cv::Mat prev_frame_;
    std::vector<cv::Point2f> prev_points_;
    std::vector<cv::Point2f> curr_points_;
    std::vector<int> track_ids_;
    std::vector<cv::Point2f> flow_vectors_;
    int next_track_id_ = 0;
};

/**
 * Demonstration of visual odometry preprocessing using optical flow
 */
void demonstrateVOPreprocessing() {
    std::cout << "\n=== Visual Odometry Preprocessing Demo ===" << std::endl;

    // Create synthetic sequence
    const int width = 640;
    const int height = 480;
    SyntheticSequenceGenerator generator(width, height);

    // Initialize tracker
    OpticalFlowConfig config;
    config.max_features = 150;
    config.fb_threshold = 1.0;
    LKFeatureTracker tracker(config);

    // Simulate camera motion sequence
    std::vector<std::pair<double, double>> motions = {
        {5.0, 0.0},    // Move right
        {5.0, 0.0},    // Move right
        {0.0, 5.0},    // Move down
        {0.0, 5.0},    // Move down
        {-3.0, 2.0},   // Move left and down
        {-3.0, 2.0},   // Move left and down
        {0.0, -5.0},   // Move up
        {3.0, -3.0},   // Move right and up
    };

    // Get initial frame and detect features
    cv::Mat prev_frame = generator.getTranslatedFrame(0, 0);
    tracker.detectFeatures(prev_frame);

    std::cout << "\nSimulating " << motions.size() << " frames of camera motion...\n" << std::endl;

    double total_dx = 0, total_dy = 0;
    double estimated_dx = 0, estimated_dy = 0;

    for (size_t i = 0; i < motions.size(); i++) {
        double dx = motions[i].first;
        double dy = motions[i].second;
        total_dx += dx;
        total_dy += dy;

        // Generate next frame with motion
        cv::Mat curr_frame = generator.getTranslatedFrame(total_dx, total_dy);

        // Track features
        auto start = std::chrono::high_resolution_clock::now();
        int tracked = tracker.trackFeatures(curr_frame);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Estimate motion from optical flow
        cv::Point2f avg_flow = tracker.computeAverageFlow();
        estimated_dx += avg_flow.x;
        estimated_dy += avg_flow.y;

        std::cout << "Frame " << (i + 1) << ": "
                  << "Tracked=" << tracked << " features, "
                  << "GT motion=(" << dx << ", " << dy << "), "
                  << "Est motion=(" << std::fixed << std::setprecision(2)
                  << avg_flow.x << ", " << avg_flow.y << "), "
                  << "Time=" << std::fixed << std::setprecision(2) << elapsed_ms << "ms"
                  << std::endl;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total ground truth motion: (" << total_dx << ", " << total_dy << ")" << std::endl;
    std::cout << "Total estimated motion: (" << std::fixed << std::setprecision(2)
              << estimated_dx << ", " << estimated_dy << ")" << std::endl;
    std::cout << "Estimation error: ("
              << std::abs(total_dx - estimated_dx) << ", "
              << std::abs(total_dy - estimated_dy) << ")" << std::endl;
}

/**
 * Benchmark Lucas-Kanade performance with different parameters
 */
void benchmarkParameters() {
    std::cout << "\n=== Lucas-Kanade Parameter Benchmark ===" << std::endl;

    const int width = 640;
    const int height = 480;
    SyntheticSequenceGenerator generator(width, height);

    cv::Mat frame1 = generator.getTranslatedFrame(0, 0);
    cv::Mat frame2 = generator.getTranslatedFrame(10, 5);

    // Detect features
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(frame1, points, 200, 0.01, 10);

    std::cout << "Testing with " << points.size() << " features\n" << std::endl;

    // Test different window sizes
    std::vector<int> window_sizes = {11, 21, 31, 41};
    std::vector<int> pyramid_levels = {0, 1, 2, 3, 4};

    std::cout << "Window Size | Pyramid Levels | Time (ms) | Success Rate" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    for (int win_size : window_sizes) {
        for (int levels : pyramid_levels) {
            std::vector<cv::Point2f> next_points;
            std::vector<uchar> status;
            std::vector<float> error;

            cv::TermCriteria criteria(
                cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

            auto start = std::chrono::high_resolution_clock::now();

            cv::calcOpticalFlowPyrLK(
                frame1, frame2,
                points, next_points,
                status, error,
                cv::Size(win_size, win_size),
                levels,
                criteria
            );

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

            int success_count = 0;
            for (uchar s : status) {
                if (s) success_count++;
            }
            double success_rate = 100.0 * success_count / points.size();

            std::cout << std::setw(11) << win_size << " | "
                      << std::setw(14) << levels << " | "
                      << std::setw(9) << std::fixed << std::setprecision(2) << elapsed << " | "
                      << std::setw(11) << std::fixed << std::setprecision(1) << success_rate << "%"
                      << std::endl;
        }
    }
}

/**
 * Demonstrate feature lifetime tracking
 */
void demonstrateFeatureLifetime() {
    std::cout << "\n=== Feature Lifetime Tracking Demo ===" << std::endl;

    const int width = 640;
    const int height = 480;
    const int num_frames = 20;
    SyntheticSequenceGenerator generator(width, height);

    OpticalFlowConfig config;
    config.max_features = 100;
    config.fb_threshold = 0.5;  // Stricter for lifetime demo
    LKFeatureTracker tracker(config);

    // Initialize
    cv::Mat prev_frame = generator.getTranslatedFrame(0, 0);
    tracker.detectFeatures(prev_frame);

    // Track through sequence
    std::map<int, int> feature_lifetimes;  // track_id -> lifetime
    std::vector<int> initial_ids = tracker.getTrackIds();
    for (int id : initial_ids) {
        feature_lifetimes[id] = 1;
    }

    double cumulative_x = 0, cumulative_y = 0;

    for (int i = 1; i < num_frames; i++) {
        // Simulate gradual motion
        double dx = 3.0 * std::sin(i * 0.3);
        double dy = 2.0 * std::cos(i * 0.4);
        cumulative_x += dx;
        cumulative_y += dy;

        cv::Mat curr_frame = generator.getTranslatedFrame(cumulative_x, cumulative_y);

        int tracked = tracker.trackFeatures(curr_frame);

        // Update lifetimes
        std::vector<int> current_ids = tracker.getTrackIds();
        for (int id : current_ids) {
            if (feature_lifetimes.find(id) != feature_lifetimes.end()) {
                feature_lifetimes[id]++;
            } else {
                feature_lifetimes[id] = 1;  // Newly detected
            }
        }

        std::cout << "Frame " << i << ": " << tracked << " features tracked" << std::endl;
    }

    // Compute lifetime statistics
    std::vector<int> lifetimes;
    for (const auto& pair : feature_lifetimes) {
        lifetimes.push_back(pair.second);
    }

    std::sort(lifetimes.begin(), lifetimes.end());

    int total = 0;
    for (int l : lifetimes) total += l;
    double avg = static_cast<double>(total) / lifetimes.size();

    std::cout << "\n=== Feature Lifetime Statistics ===" << std::endl;
    std::cout << "Total features tracked: " << lifetimes.size() << std::endl;
    std::cout << "Average lifetime: " << std::fixed << std::setprecision(1) << avg << " frames" << std::endl;
    std::cout << "Min lifetime: " << lifetimes.front() << " frames" << std::endl;
    std::cout << "Max lifetime: " << lifetimes.back() << " frames" << std::endl;
    std::cout << "Median lifetime: " << lifetimes[lifetimes.size() / 2] << " frames" << std::endl;

    // Count features that survived all frames
    int survivors = 0;
    for (int l : lifetimes) {
        if (l == num_frames) survivors++;
    }
    std::cout << "Features surviving all " << num_frames << " frames: " << survivors << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "======================================================" << std::endl;
    std::cout << "  Sparse Optical Flow - Lucas-Kanade Feature Tracker  " << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // Run demonstrations
    demonstrateVOPreprocessing();
    benchmarkParameters();
    demonstrateFeatureLifetime();

    std::cout << "\n=== All demonstrations complete ===" << std::endl;

    return 0;
}
