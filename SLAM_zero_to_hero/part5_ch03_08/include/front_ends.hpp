/**
 * @file front_ends.hpp
 * @brief The two KLT front ends compared by tracker_benchmark and frontend_rerun
 *
 * - OpenCvFrontEnd: cv::buildOpticalFlowPyramid + cv::calcOpticalFlowPyrLK, with vilib's
 *   CPU FAST (Rosten) in the empty grid cells
 * - VilibFrontEnd: vilib's FeatureTrackerGPU with its CUDA FAST
 *
 * Both run the same front end once per frame: track every feature into the new frame,
 * drop the ones that fail (KLT does not converge on some pyramid level, or the feature
 * leaves the image), and when fewer than kMinTracks survive, detect FAST corners in the
 * cells that have no feature and add the strongest until there are kMaxFeatures.
 */

#pragma once

#include <vilib/common/frame.h>
#include <vilib/common/framebundle.h>
#include <vilib/feature_detection/fast/fast_gpu.h>
#include <vilib/feature_detection/fast/rosten/fast_cpu.h>
#include <vilib/feature_tracker/feature_tracker_gpu.h>
#include <vilib/storage/pyramid_pool.h>
#include <vilib/storage/subframe.h>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

// KLT
constexpr int kPyramidLevels = 5;              // 752x480 down to 47x30
constexpr int kPatchSize = 16;                 // template patch, on every level
constexpr int kMaxIterations = 30;             // per level; vilib's is the compile-time FEATURE_TRACKER_MAX_ITERATION_COUNT
constexpr double kMinUpdateSquared = 0.0005;   // stop when |update|^2 < this (px^2)
// Detection (the same FAST as fast_benchmark)
constexpr float kThreshold = 10.f;
constexpr int kArcLength = 9;
constexpr int kCellSize = 32;
constexpr int kBorder = kPatchSize / 2;        // a patch around the feature must fit in the image
// Feature budget
constexpr size_t kMaxFeatures = 200;           // detect up to this many features in total
constexpr size_t kMinTracks = 100;             // detect only when fewer tracks than this survive

struct FrameStats {
    size_t tracked = 0;   // features carried over from the previous frame
    size_t detected = 0;  // new features
};

// A feature in the current frame, for drawing: its track id and its position in pixels
struct Feature {
    int id;
    float x, y;
};

/**
 * @brief The CPU front end: OpenCV's pyramidal KLT plus vilib's CPU FAST in the empty cells
 *
 * Every feature keeps the pyramid of the frame it was detected in as its template.
 * Features detected in the same frame are tracked together, from their first position,
 * starting at their position in the previous frame, one pyramid level at a time so that
 * a feature that does not converge on a level can be dropped there, as vilib does.
 */
class OpenCvFrontEnd {
public:
    OpenCvFrontEnd(int width, int height)
        : width_(width),
          height_(height),
          detector_(width, height, kCellSize, kCellSize, 0, 1, kBorder, kBorder, kThreshold, kArcLength,
                    vilib::SUM_OF_ABS_DIFF_ON_ARC) {}

    FrameStats track(const cv::Mat& image) {
        FrameStats stats;
        // Only template frames need image derivatives; the new frame only needs its pyramid
        cv::buildOpticalFlowPyramid(image, cur_pyramid_, window(), kPyramidLevels - 1, false);
        if (!tracks_.empty()) {
            for (auto it = templates_.begin(); it != templates_.end();) {
                trackFrom(it->first, it->second);
                // a template frame with no feature left is no longer needed
                const bool used = std::any_of(tracks_.begin(), tracks_.end(),
                                              [&](const Track& t) { return t.template_frame == it->first; });
                it = used ? std::next(it) : templates_.erase(it);
            }
            stats.tracked = tracks_.size();
        }
        if (tracks_.size() < kMinTracks) {
            stats.detected = detect(image);
            if (stats.detected > 0)
                cv::buildOpticalFlowPyramid(image, templates_[frame_index_], window(), kPyramidLevels - 1, true);
        }
        ++frame_index_;
        return stats;
    }

    void reset() {
        tracks_.clear();
        templates_.clear();
        frame_index_ = 0;
        next_id_ = 0;
    }

    std::vector<Feature> features() const {
        std::vector<Feature> out;
        out.reserve(tracks_.size());
        for (const Track& t : tracks_) out.push_back({t.id, t.current.x, t.current.y});
        return out;
    }

    static cv::Size window() { return cv::Size(kPatchSize, kPatchSize); }

    // OpenCV squares epsilon internally and compares it with |update|^2
    static cv::TermCriteria criteria() {
        return cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, kMaxIterations,
                                std::sqrt(kMinUpdateSquared));
    }

private:
    struct Track {
        cv::Point2f first;    // where it was detected, in its template frame
        cv::Point2f current;  // where it is now
        int template_frame;
        int id;
    };

    // Track every feature detected in template_frame into the current frame, coarse to fine.
    // calcOpticalFlowPyrLK does not say whether a feature converged: its status only flags one
    // that leaves the image or has too little texture, and a feature that runs out of iterations
    // keeps its last position, often a wrong one. Tracked with one call and no other test, 12% of
    // the steps on MH_01 landed more than 3 px from a frame-to-frame check (vilib: 1.2%; README,
    // "What the numbers say", point 5). So each level gets its own call, then one more
    // Gauss-Newton step from the result: if that step still moves the feature by
    // kMinUpdateSquared or more, the level has not converged and the track ends, vilib's rule.
    void trackFrom(int template_frame, const std::vector<cv::Mat>& template_pyramid) {
        active_.clear();
        for (size_t i = 0; i < tracks_.size(); ++i)
            if (tracks_[i].template_frame == template_frame) active_.push_back(i);
        if (active_.empty()) return;
        lost_.assign(tracks_.size(), false);

        const cv::TermCriteria one_step(cv::TermCriteria::COUNT, 1, 0);
        for (int level = kPyramidLevels - 1; level >= 0 && !active_.empty(); --level) {
            // Built with derivatives, the template pyramid holds two Mats per level
            const std::vector<cv::Mat> templ(template_pyramid.begin() + 2 * level,
                                             template_pyramid.begin() + 2 * level + 2);
            const std::vector<cv::Mat> cur{cur_pyramid_[level]};
            const float scale = static_cast<float>(1 << level);
            from_.clear();
            to_.clear();
            for (size_t i : active_) {
                from_.push_back(tracks_[i].first / scale);
                to_.push_back(tracks_[i].current / scale);  // the level above's result, or the previous frame's
            }
            cv::calcOpticalFlowPyrLK(templ, cur, from_, to_, status_, error_, window(), 0, criteria(),
                                     cv::OPTFLOW_USE_INITIAL_FLOW);
            check_ = to_;
            cv::calcOpticalFlowPyrLK(templ, cur, from_, check_, check_status_, error_, window(), 0, one_step,
                                     cv::OPTFLOW_USE_INITIAL_FLOW);
            size_t still = 0;
            for (size_t j = 0; j < active_.size(); ++j) {
                tracks_[active_[j]].current = to_[j] * scale;
                const cv::Point2f step = check_[j] - to_[j];
                if (status_[j] && check_status_[j] && step.dot(step) < kMinUpdateSquared)
                    active_[still++] = active_[j];
                else
                    lost_[active_[j]] = true;
            }
            active_.resize(still);
        }

        size_t kept = 0;
        for (size_t i = 0; i < tracks_.size(); ++i) {
            const Track& t = tracks_[i];
            if (t.template_frame == template_frame && (lost_[i] || !insideBorder(t.current))) continue;
            tracks_[kept++] = t;
        }
        tracks_.resize(kept);
    }

    bool insideBorder(const cv::Point2f& p) const {
        return p.x >= kBorder && p.y >= kBorder && p.x < width_ - kBorder && p.y < height_ - kBorder;
    }

    // Same rule as vilib's tracker: strongest corners first, only in cells without a feature
    size_t detect(const cv::Mat& image) {
        detector_.reset();
        detector_.detect({image});
        auto& grid = detector_.getGrid();
        const auto& corners = detector_.getPoints();

        std::vector<bool> taken(grid.size(), false);
        for (const Track& t : tracks_)
            taken[grid.getCellIndex(static_cast<int>(t.current.x), static_cast<int>(t.current.y))] = true;
        std::vector<size_t> cells;
        for (size_t i = 0; i < grid.size(); ++i)
            if (grid.isOccupied(i) && !taken[i]) cells.push_back(i);
        std::sort(cells.begin(), cells.end(),
                  [&](size_t a, size_t b) { return corners[a].score_ > corners[b].score_; });

        const size_t room = kMaxFeatures > tracks_.size() ? kMaxFeatures - tracks_.size() : 0;
        const size_t added = std::min(room, cells.size());
        for (size_t i = 0; i < added; ++i) {
            const cv::Point2f p(static_cast<float>(corners[cells[i]].x_), static_cast<float>(corners[cells[i]].y_));
            tracks_.push_back({p, p, frame_index_, next_id_++});
        }
        return added;
    }

    int width_, height_;
    vilib::rosten::FASTCPU<true> detector_;
    std::vector<Track> tracks_;
    std::map<int, std::vector<cv::Mat>> templates_;  // frame index -> its pyramid, with derivatives
    std::vector<cv::Mat> cur_pyramid_;
    int frame_index_ = 0;
    int next_id_ = 0;
    std::vector<size_t> active_;  // tracks of the template frame still converging, as indices into tracks_
    std::vector<bool> lost_;
    std::vector<cv::Point2f> from_, to_, check_;
    std::vector<uchar> status_, check_status_;
    std::vector<float> error_;
};

/**
 * @brief The CUDA front end: vilib's FeatureTrackerGPU with its CUDA FAST
 */
class VilibFrontEnd {
public:
    VilibFrontEnd(int width, int height) {
        vilib::FeatureTrackerOptions options;
        options.klt_max_level = kPyramidLevels - 1;
        options.klt_min_level = 0;
        options.klt_patch_sizes = std::vector<int>(kPyramidLevels, kPatchSize);
        options.klt_min_update_squared = kMinUpdateSquared;
        options.klt_template_is_first_observation = true;  // vilib's default; see the README on false
        options.min_tracks_to_detect_new_features = kMinTracks;
        options.use_best_n_features = static_cast<int>(kMaxFeatures);
        options.reset_before_detection = false;
        options.affine_est_offset = false;
        options.affine_est_gain = false;

        // Frames stay alive while their tracks do, so several pyramids are in use at once.
        // 10 is vilib's own IMAGE_PYRAMID_PREALLOCATION_ITEM_NUM; the pool allocates more if needed.
        vilib::PyramidPool::init(10, width, height, 1, kPyramidLevels,
                                 vilib::Subframe::MemoryType::PITCHED_DEVICE_MEMORY);
        detector_ = std::make_shared<vilib::FASTGPU>(width, height, kCellSize, kCellSize, 0, 1, kBorder, kBorder,
                                                     kThreshold, kArcLength, vilib::SUM_OF_ABS_DIFF_ON_ARC);
        tracker_ = std::make_unique<vilib::FeatureTrackerGPU>(options, 1);
        tracker_->setDetectorGPU(detector_, 0);
    }

    ~VilibFrontEnd() {
        tracker_.reset();
        detector_.reset();
        vilib::PyramidPool::deinit();
    }

    // Upload + pyramid. The pyramid buffers come from vilib's PyramidPool; a frame
    // returns them when the last track that started on it ends
    std::shared_ptr<vilib::Frame> makeFrame(const cv::Mat& image) const {
        return std::make_shared<vilib::Frame>(image, 0, kPyramidLevels);
    }

    FrameStats track(const std::shared_ptr<vilib::Frame>& frame) {
        FrameStats stats;
        const auto bundle = std::make_shared<vilib::FrameBundle>(std::vector<std::shared_ptr<vilib::Frame>>{frame});
        tracker_->track(bundle, stats.tracked, stats.detected);
        return stats;
    }

    void reset() { tracker_->reset(); }

    // The frame's features after track(): vilib keeps them in the frame
    static std::vector<Feature> features(const vilib::Frame& frame) {
        std::vector<Feature> out;
        out.reserve(frame.num_features_);
        for (size_t i = 0; i < frame.num_features_; ++i)
            out.push_back({frame.track_id_vec_[i], static_cast<float>(frame.px_vec_(0, i)),
                           static_cast<float>(frame.px_vec_(1, i))});
        return out;
    }

private:
    std::shared_ptr<vilib::DetectorBaseGPU> detector_;
    std::unique_ptr<vilib::FeatureTrackerGPU> tracker_;
};
