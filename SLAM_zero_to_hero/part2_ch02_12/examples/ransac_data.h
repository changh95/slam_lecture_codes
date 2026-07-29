/**
 * Shared data pipeline, metrics, and visualization for the RANSAC/USAC demos.
 *
 * Every demo in this chapter estimates on the SAME inputs so that the
 * numbers in results.csv are comparable across estimators:
 *
 *  - H / F: ORB correspondences from the real EuRoC MAV pair
 *    data/1403636579763555584.png -> data/1403636584763555584.png (the EuRoC
 *    nanosecond-timestamp filenames, kept verbatim; MH_01_easy cam0, frames 0 and 100
 *    of a 20 Hz stream, i.e. 5 s apart -- a wide baseline, not consecutive
 *    frames). Detector settings, matcher, and ratio test are fixed here, in
 *    one place.
 *  - Line fitting: a fixed-seed synthetic point set (a 2D line has no
 *    real-image equivalent in this dataset).
 *
 * Quality metrics are shared too: every estimator's model is scored by the
 * same code (mean inlier reprojection error for H, mean squared Sampson
 * distance for F), regardless of which library produced it.
 */
#ifndef PART2_CH02_12_RANSAC_DATA_H_
#define PART2_CH02_12_RANSAC_DATA_H_

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

inline std::string resolveDataPath(const std::string& name) {
    for (const std::string& base : {"../data/", "data/", "./data/"}) {
        if (std::filesystem::exists(base + name)) return base + name;
    }
    return "../data/" + name;
}

// ORB + BF-Hamming + Lowe ratio test, emitted best-match-first. The single
// source of correspondences for every H/F estimator in this chapter.
//
// The ordering is load-bearing, not cosmetic. PROSAC samples the front of the
// list first and only widens toward uniform sampling as it fails, so it requires
// the correspondences sorted by DESCENDING quality:
//   - opencv2/calib3d.hpp: "USAC_PROSAC = 37, //!< USAC, sorted points, runs
//     PROSAC"
//   - usac/sampler.cpp, ProsacSamplerImpl: "The data points in U_N are sorted in
//     descending order w.r.t. the quality function q", which is what its growth
//     function is derived from.
// knnMatch returns matches in queryIdx order, which carries no quality
// information, so feeding that straight in leaves PROSAC drawing from an
// arbitrary prefix and unable to show what it does. The Lowe ratio is the
// quality score here: a lower ratio means the best match beat the runner-up by
// more, i.e. a more distinctive correspondence, so ascending ratio is descending
// quality. Every estimator receives this same order, so the comparison across
// methods is unaffected.
inline void detectAndMatch(const cv::Mat& img1, const cv::Mat& img2,
                           std::vector<cv::Point2f>& pts1,
                           std::vector<cv::Point2f>& pts2) {
    auto orb = cv::ORB::create(3000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, des1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, des2);
    std::cout << "ORB keypoints: " << kp1.size() << " in img1, " << kp2.size() << " in img2\n";

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(des1, des2, knn, 2);

    struct ScoredMatch {
        float ratio;        // Lowe ratio: lower is a stronger match
        cv::Point2f p1, p2;
    };
    std::vector<ScoredMatch> scored;
    scored.reserve(knn.size());
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
            const float denom = std::max(m[1].distance, 1e-6f);
            scored.push_back({m[0].distance / denom,
                              kp1[m[0].queryIdx].pt,
                              kp2[m[0].trainIdx].pt});
        }
    }

    // stable_sort so equal ratios keep a deterministic order run to run, which
    // results.csv depends on.
    std::stable_sort(scored.begin(), scored.end(),
                     [](const ScoredMatch& a, const ScoredMatch& b) {
                         return a.ratio < b.ratio;
                     });

    pts1.clear(); pts2.clear();
    pts1.reserve(scored.size());
    pts2.reserve(scored.size());
    for (const auto& s : scored) {
        pts1.push_back(s.p1);
        pts2.push_back(s.p2);
    }
    std::cout << "Ratio-test matches: " << pts1.size()
              << " (sorted best-first by Lowe ratio, as PROSAC requires)\n";
}

// Load the real image pair (or the two paths given on the command line) and
// produce the shared ORB correspondences. Returns false on failure.
inline bool loadRealPair(int argc, char* argv[],
                         cv::Mat& img1, cv::Mat& img2,
                         std::vector<cv::Point2f>& pts1,
                         std::vector<cv::Point2f>& pts2,
                         size_t minMatches) {
    std::string left_path  = (argc >= 3) ? argv[1]
                                         : resolveDataPath("1403636579763555584.png");
    std::string right_path = (argc >= 3) ? argv[2]
                                         : resolveDataPath("1403636584763555584.png");

    img1 = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: failed to load images:\n  " << left_path << "\n  " << right_path
                  << "\nRun from build/ so ../data resolves, or pass two image paths.\n";
        return false;
    }
    std::cout << "Image 1: " << left_path  << "  (" << img1.cols << "x" << img1.rows << ")\n";
    std::cout << "Image 2: " << right_path << "  (" << img2.cols << "x" << img2.rows << ")\n\n";

    detectAndMatch(img1, img2, pts1, pts2);
    if (pts1.size() < minMatches) {
        std::cerr << "Not enough matches (" << pts1.size() << " < " << minMatches << ").\n";
        return false;
    }
    return true;
}

// Inlier mask as Nx1 uchar Mat (the OpenCV convention). Helper for demos
// whose own RANSAC tracks inliers as std::vector<bool>.
inline cv::Mat toMask(const std::vector<bool>& flags) {
    cv::Mat mask(static_cast<int>(flags.size()), 1, CV_8U);
    for (size_t i = 0; i < flags.size(); ++i) {
        mask.at<uchar>(static_cast<int>(i)) = flags[i] ? 1 : 0;
    }
    return mask;
}

// Mean reprojection error |H * x1 - x2| in pixels over inliers selected by
// mask (all points if mask is empty).
inline double meanInlierReproj(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const cv::Mat& H, const cv::Mat& mask) {
    if (H.empty() || pts1.empty()) return -1.0;
    double total = 0.0;
    int n = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!mask.empty() && !mask.at<uchar>(static_cast<int>(i))) continue;
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2p = H * x1;
        double w = x2p.at<double>(2);
        if (std::abs(w) < 1e-12) continue;
        double u = x2p.at<double>(0) / w;
        double v = x2p.at<double>(1) / w;
        double dx = u - pts2[i].x;
        double dy = v - pts2[i].y;
        total += std::sqrt(dx * dx + dy * dy);
        ++n;
    }
    return n ? total / n : -1.0;
}

// Mean squared Sampson distance over inliers selected by mask (all points if
// mask is empty).
inline double meanSampson(const cv::Mat& F,
                          const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2,
                          const cv::Mat& mask = cv::Mat()) {
    if (F.empty() || F.rows < 3) return -1.0;
    double total = 0.0;
    int n = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!mask.empty() && !mask.at<uchar>(static_cast<int>(i))) continue;
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat Fp1 = F * p1;
        cv::Mat Ftp2 = F.t() * p2;
        double num = p2.dot(Fp1);
        double denom = Fp1.at<double>(0) * Fp1.at<double>(0)
                     + Fp1.at<double>(1) * Fp1.at<double>(1)
                     + Ftp2.at<double>(0) * Ftp2.at<double>(0)
                     + Ftp2.at<double>(1) * Ftp2.at<double>(1);
        if (denom > 1e-10) {
            total += (num * num) / denom;
            ++n;
        }
    }
    return n ? total / n : -1.0;
}

// Fixed-seed synthetic line dataset shared by the line-fitting demos:
// 45 inliers on y = 0.5x + 120 with sigma=2 noise, plus 30 uniform outliers.
//
// Two properties are deliberate, because the point of the picture is to look
// like a line fitted through *data points*:
//  - Inliers are spaced ~11.5 px apart. The earlier version stepped x by 5 px,
//    which at a 3 px dot radius fused all of them into a single green
//    caterpillar -- you could see a line but not the samples forming it.
//  - Outliers are rejected within kOutlierClearance of the true line, so no
//    point sits ambiguously on the boundary and the expected inlier count is
//    the construction count rather than "about" it.
inline std::vector<cv::Point2f> generateLinePoints() {
    constexpr int kInliers = 45;
    constexpr int kOutliers = 30;
    constexpr float kSlope = 0.5f;
    constexpr float kIntercept = 120.0f;
    constexpr float kStepX = 11.5f;
    constexpr float kFirstX = 40.0f;
    constexpr float kOutlierClearance = 30.0f;  // px from the true line

    std::vector<cv::Point2f> points;
    points.reserve(kInliers + kOutliers);
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 2.0f);

    for (int i = 0; i < kInliers; ++i) {
        const float x = kFirstX + kStepX * static_cast<float>(i);
        const float y = kSlope * x + kIntercept + noise(rng);
        points.emplace_back(x, y);
    }

    // Perpendicular distance to kSlope*x - y + kIntercept = 0.
    const float lineNorm = std::hypot(kSlope, 1.0f);
    std::uniform_real_distribution<float> outlierX(30.0f, 530.0f);
    std::uniform_real_distribution<float> outlierY(30.0f, 530.0f);
    for (int placed = 0; placed < kOutliers;) {
        const float x = outlierX(rng);
        const float y = outlierY(rng);
        if (std::abs(kSlope * x - y + kIntercept) / lineNorm < kOutlierClearance) {
            continue;  // too close to the line to be an unambiguous outlier
        }
        points.emplace_back(x, y);
        ++placed;
    }

    std::shuffle(points.begin(), points.end(), rng);
    return points;
}

// Stack the two frames vertically and draw match segments colored by the
// inlier mask (green = inlier, red = outlier).
inline cv::Mat drawMatchesVis(const cv::Mat& img1, const cv::Mat& img2,
                              const std::vector<cv::Point2f>& pts1,
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& mask) {
    cv::Mat top, bottom, vis;
    cv::cvtColor(img1, top, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, bottom, cv::COLOR_GRAY2BGR);
    cv::vconcat(top, bottom, vis);
    const cv::Point2f yoff(0.0f, static_cast<float>(img1.rows));
    for (size_t i = 0; i < pts1.size(); ++i) {
        bool inlier = !mask.empty() && mask.at<uchar>(static_cast<int>(i));
        cv::Scalar color = inlier ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::circle(vis, pts1[i], 3, color, -1);
        cv::circle(vis, pts2[i] + yoff, 3, color, -1);
        cv::line(vis, pts1[i], pts2[i] + yoff, color, 1);
    }
    return vis;
}

#endif  // PART2_CH02_12_RANSAC_DATA_H_
