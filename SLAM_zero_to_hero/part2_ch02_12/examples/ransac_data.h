/**
 * Shared data pipeline, metrics, and visualization for the RANSAC/USAC demos.
 *
 * Every demo in this chapter estimates on the SAME inputs so that the
 * numbers in results.csv are comparable across estimators:
 *
 *  - H / F: ORB correspondences from the real EuRoC MAV pair
 *    data/000024.png -> data/000025.png (MH_01_easy cam0, frames 0 and 100
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

// ORB + BF-Hamming + Lowe ratio test. The single source of correspondences
// for every H/F estimator in this chapter.
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

    pts1.clear(); pts2.clear();
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
    std::cout << "Ratio-test matches: " << pts1.size() << "\n";
}

// Load the real image pair (or the two paths given on the command line) and
// produce the shared ORB correspondences. Returns false on failure.
inline bool loadRealPair(int argc, char* argv[],
                         cv::Mat& img1, cv::Mat& img2,
                         std::vector<cv::Point2f>& pts1,
                         std::vector<cv::Point2f>& pts2,
                         size_t minMatches) {
    std::string left_path  = (argc >= 3) ? argv[1] : resolveDataPath("000024.png");
    std::string right_path = (argc >= 3) ? argv[2] : resolveDataPath("000025.png");

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
// y = 0.5x + 100 with sigma=2 noise (70 inliers) plus 30 uniform outliers.
inline std::vector<cv::Point2f> generateLinePoints() {
    std::vector<cv::Point2f> points;
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 2.0f);

    for (int i = 0; i < 70; ++i) {
        float x = static_cast<float>(i * 5);
        float y = 0.5f * x + 100.0f + noise(rng);
        points.emplace_back(x, y);
    }
    std::uniform_real_distribution<float> outlierX(0.0f, 350.0f);
    std::uniform_real_distribution<float> outlierY(0.0f, 400.0f);
    for (int i = 0; i < 30; ++i) {
        points.emplace_back(outlierX(rng), outlierY(rng));
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
