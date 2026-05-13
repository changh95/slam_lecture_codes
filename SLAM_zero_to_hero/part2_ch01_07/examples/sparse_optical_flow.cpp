/**
 * Sparse Optical Flow using Lucas-Kanade Method
 *
 * Live demo: tracks Shi-Tomasi corners frame-by-frame through a TUM RGB-D
 * sequence using pyramidal Lucas-Kanade with forward-backward consistency
 * checking. Renders each tracked frame to a cv::imshow window.
 *
 * Usage:
 *   sparse_optical_flow [seq_dir] [num_frames]
 *
 *   seq_dir     - path to a TUM RGB-D sequence containing rgb/*.png + rgb.txt
 *                 (default: /data/tum_rgbd/rgbd_dataset_freiburg1_desk)
 *   num_frames  - number of frames to stream through (default: 500)
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

struct OpticalFlowConfig {
    int max_features = 200;
    double quality_level = 0.01;
    double min_distance = 10.0;
    int block_size = 3;

    cv::Size win_size{21, 21};
    int max_pyramid_level = 3;
    int max_iterations = 30;
    double epsilon = 0.01;

    double max_error = 12.0;
    double fb_threshold = 1.0;
    int min_features = 50;
};

static std::vector<std::string> loadTumRgbList(const std::string& seq_dir, size_t limit) {
    std::vector<std::string> frames;
    fs::path rgb_txt = fs::path(seq_dir) / "rgb.txt";
    if (fs::exists(rgb_txt)) {
        std::ifstream in(rgb_txt);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string ts, rel;
            if (!(iss >> ts >> rel)) continue;
            frames.push_back((fs::path(seq_dir) / rel).string());
            if (frames.size() >= limit) break;
        }
    } else {
        fs::path rgb_dir = fs::path(seq_dir) / "rgb";
        for (const auto& entry : fs::directory_iterator(rgb_dir)) {
            if (entry.path().extension() == ".png") {
                frames.push_back(entry.path().string());
            }
        }
        std::sort(frames.begin(), frames.end());
        if (frames.size() > limit) frames.resize(limit);
    }
    return frames;
}

class LKFeatureTracker {
public:
    explicit LKFeatureTracker(const OpticalFlowConfig& config = OpticalFlowConfig())
        : config_(config),
          term_criteria_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                         config.max_iterations, config.epsilon) {}

    void detectFeatures(const cv::Mat& frame) {
        cv::goodFeaturesToTrack(frame, prev_points_, config_.max_features,
                                config_.quality_level, config_.min_distance,
                                cv::Mat(), config_.block_size, false, 0.04);
        frame.copyTo(prev_frame_);
        track_ids_.clear();
        for (size_t i = 0; i < prev_points_.size(); ++i) {
            track_ids_.push_back(next_track_id_++);
        }
        last_prev_points_.clear();
        std::cout << "Detected " << prev_points_.size() << " initial features" << std::endl;
    }

    int trackFeatures(const cv::Mat& curr_frame) {
        if (prev_points_.empty()) return 0;

        std::vector<cv::Point2f> curr_points, back_points;
        std::vector<uchar> status, back_status;
        std::vector<float> err, back_err;

        cv::calcOpticalFlowPyrLK(prev_frame_, curr_frame, prev_points_, curr_points,
                                 status, err, config_.win_size,
                                 config_.max_pyramid_level, term_criteria_, 0, 1e-4);
        cv::calcOpticalFlowPyrLK(curr_frame, prev_frame_, curr_points, back_points,
                                 back_status, back_err, config_.win_size,
                                 config_.max_pyramid_level, term_criteria_, 0, 1e-4);

        std::vector<cv::Point2f> good_prev, good_curr, flow_vec;
        std::vector<int> good_ids;
        for (size_t i = 0; i < curr_points.size(); ++i) {
            if (!status[i] || !back_status[i]) continue;
            if (err[i] > config_.max_error) continue;
            if (cv::norm(prev_points_[i] - back_points[i]) > config_.fb_threshold) continue;
            if (curr_points[i].x < 0 || curr_points[i].x >= curr_frame.cols ||
                curr_points[i].y < 0 || curr_points[i].y >= curr_frame.rows) continue;

            good_prev.push_back(prev_points_[i]);
            good_curr.push_back(curr_points[i]);
            good_ids.push_back(track_ids_[i]);
            flow_vec.push_back(curr_points[i] - prev_points_[i]);
        }

        last_prev_points_ = good_prev;
        prev_points_ = good_curr;
        curr_points_ = good_curr;
        track_ids_ = good_ids;
        flow_vectors_ = flow_vec;
        curr_frame.copyTo(prev_frame_);

        int tracked = static_cast<int>(curr_points_.size());
        if (tracked < config_.min_features) {
            detectAdditionalFeatures(curr_frame);
        }
        return tracked;
    }

    void detectAdditionalFeatures(const cv::Mat& frame) {
        cv::Mat mask = cv::Mat::ones(frame.size(), CV_8UC1) * 255;
        for (const auto& pt : prev_points_) {
            cv::circle(mask, pt, static_cast<int>(config_.min_distance), cv::Scalar(0), -1);
        }
        int num_to_detect = config_.max_features - static_cast<int>(prev_points_.size());
        if (num_to_detect <= 0) return;

        std::vector<cv::Point2f> new_points;
        cv::goodFeaturesToTrack(frame, new_points, num_to_detect, config_.quality_level,
                                config_.min_distance, mask, config_.block_size, false, 0.04);
        for (const auto& pt : new_points) {
            prev_points_.push_back(pt);
            track_ids_.push_back(next_track_id_++);
        }
    }

    cv::Point2f computeAverageFlow() const {
        if (flow_vectors_.empty()) return {0, 0};
        cv::Point2f avg(0, 0);
        for (const auto& f : flow_vectors_) avg += f;
        avg.x /= flow_vectors_.size();
        avg.y /= flow_vectors_.size();
        return avg;
    }

    cv::Mat visualize(const cv::Mat& bgr_frame) const {
        cv::Mat vis = bgr_frame.clone();
        for (size_t i = 0; i < curr_points_.size() && i < last_prev_points_.size(); ++i) {
            int hue = (track_ids_[i] * 37) % 180;
            cv::Scalar color(hue * 1.4, 255 - hue, 100 + hue % 155);
            cv::line(vis, last_prev_points_[i], curr_points_[i], color, 2);
            cv::circle(vis, curr_points_[i], 4, color, -1);
            cv::circle(vis, last_prev_points_[i], 2, cv::Scalar(100, 100, 100), -1);
        }
        std::string info = "Tracking " + std::to_string(curr_points_.size()) + " features";
        cv::putText(vis, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);
        cv::Point2f avg = computeAverageFlow();
        std::ostringstream oss;
        oss << "Avg flow: (" << std::fixed << std::setprecision(1) << avg.x
            << ", " << avg.y << ")";
        cv::putText(vis, oss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);
        return vis;
    }

private:
    OpticalFlowConfig config_;
    cv::TermCriteria term_criteria_;
    cv::Mat prev_frame_;
    std::vector<cv::Point2f> prev_points_;
    std::vector<cv::Point2f> last_prev_points_;
    std::vector<cv::Point2f> curr_points_;
    std::vector<int> track_ids_;
    std::vector<cv::Point2f> flow_vectors_;
    int next_track_id_ = 0;
};

int main(int argc, char** argv) {
    std::string seq_dir = (argc > 1) ? argv[1]
        : "/data/tum_rgbd/rgbd_dataset_freiburg1_desk";
    int num_frames = (argc > 2) ? std::stoi(argv[2]) : 500;

    std::cout << "Sparse LK live demo - OpenCV " << CV_VERSION << std::endl;
    std::cout << "Sequence:  " << seq_dir << std::endl;
    std::cout << "Frames:    " << num_frames << std::endl;

    auto frames = loadTumRgbList(seq_dir, static_cast<size_t>(num_frames));
    if (frames.size() < 2) {
        std::cerr << "Need at least 2 frames in " << seq_dir << "/rgb" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << frames.size() << " frames" << std::endl;

    OpticalFlowConfig config;
    config.max_features = 200;
    LKFeatureTracker tracker(config);

    cv::Mat bgr_prev = cv::imread(frames[0], cv::IMREAD_COLOR);
    cv::Mat gray_prev;
    cv::cvtColor(bgr_prev, gray_prev, cv::COLOR_BGR2GRAY);
    tracker.detectFeatures(gray_prev);

    cv::namedWindow("Sparse LK Tracking", cv::WINDOW_AUTOSIZE);

    for (size_t i = 1; i < frames.size(); ++i) {
        cv::Mat bgr_curr = cv::imread(frames[i], cv::IMREAD_COLOR);
        cv::Mat gray_curr;
        cv::cvtColor(bgr_curr, gray_curr, cv::COLOR_BGR2GRAY);

        auto t0 = std::chrono::high_resolution_clock::now();
        int tracked = tracker.trackFeatures(gray_curr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        cv::Mat vis = tracker.visualize(bgr_curr);
        std::ostringstream tag;
        tag << "frame " << std::setw(4) << std::setfill('0') << i
            << "  " << std::fixed << std::setprecision(1) << ms << " ms";
        cv::putText(vis, tag.str(), cv::Point(10, vis.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Sparse LK Tracking", vis);
        if ((cv::waitKey(30) & 0xff) == 27) break;

        if (i % 25 == 0 || i == frames.size() - 1) {
            cv::Point2f f = tracker.computeAverageFlow();
            std::cout << "Frame " << std::setw(4) << i
                      << ": tracked=" << std::setw(3) << tracked
                      << ", avg flow=(" << std::fixed << std::setprecision(2)
                      << f.x << ", " << f.y << "), " << ms << " ms" << std::endl;
        }
    }

    std::cout << "Press any key in the window to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
