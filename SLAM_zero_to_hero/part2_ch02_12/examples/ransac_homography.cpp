/**
 * RANSAC Homography Estimation
 *
 * Run all seven RANSAC / USAC variants on real ORB correspondences from a
 * KITTI consecutive frame pair (forward motion). The dominant ground plane
 * and distant facades fit a homography reasonably while the rest of the
 * scene has parallax outliers -- exactly the regime that exercises RANSAC.
 *
 * Pipeline:
 *   1. Load left/right images (data/000024.png, data/000025.png by default).
 *   2. ORB detect + BF-Hamming + Lowe ratio test (0.75) gives raw matches.
 *   3. Run RANSAC, LMEDS, four USAC flags, and a Custom UsacParams config.
 *   4. Per method, report mean inlier reprojection error, inlier count, time.
 *
 * No synthetic ground-truth H here: real images don't admit a single GT
 * homography. The error column is the *mean inlier reprojection error*
 * |H * x1 - x2| in pixels, which is the metric a robust estimator actually
 * minimizes.
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
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

static std::string resolveDataPath(const std::string& name) {
    for (const std::string& base : {"../data/", "data/", "./data/"}) {
        if (std::filesystem::exists(base + name)) return base + name;
    }
    return "../data/" + name;
}

static void detectAndMatch(const cv::Mat& img1, const cv::Mat& img2,
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

// Mean reprojection error |H * x1 - x2| over inliers selected by mask.
static double meanInlierReproj(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const cv::Mat& H, const cv::Mat& mask) {
    if (H.empty() || pts1.empty()) return -1.0;
    double total = 0.0;
    int n = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!mask.empty() && !mask.at<uchar>(i)) continue;
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

struct MethodResult {
    std::string name;
    cv::Mat H;
    cv::Mat mask;
    double err;
    int inliers;
    double time_ms;
};

int main(int argc, char* argv[]) {
    std::cout << "=== RANSAC Homography Estimation (KITTI consecutive frames) ===\n";
    std::cout << std::fixed << std::setprecision(4);

    std::string left_path  = (argc >= 3) ? argv[1] : resolveDataPath("000024.png");
    std::string right_path = (argc >= 3) ? argv[2] : resolveDataPath("000025.png");

    cv::Mat img1 = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: failed to load images:\n  " << left_path << "\n  " << right_path
                  << "\nRun from build/ so ../data resolves, or pass two image paths.\n";
        return 1;
    }
    std::cout << "Image 1: " << left_path  << "  (" << img1.cols << "x" << img1.rows << ")\n";
    std::cout << "Image 2: " << right_path << "  (" << img2.cols << "x" << img2.rows << ")\n\n";

    std::vector<cv::Point2f> pts1, pts2;
    detectAndMatch(img1, img2, pts1, pts2);
    if (pts1.size() < 4) {
        std::cerr << "Not enough matches for homography.\n";
        return 1;
    }

    const double threshold = 3.0;
    Timer timer;

    auto run = [&](const std::string& name, auto fn) -> MethodResult {
        MethodResult r{name, {}, {}, -1.0, 0, 0.0};
        timer.start();
        r.H = fn(r.mask);
        r.time_ms = timer.elapsedMs();
        if (!r.H.empty()) {
            r.inliers = cv::countNonZero(r.mask);
            r.err = meanInlierReproj(pts1, pts2, r.H, r.mask);
        }
        std::cout << "\n--- " << name << " ---\n";
        if (r.H.empty()) {
            std::cout << "  (no solution)  Time: " << r.time_ms << " ms\n";
        } else {
            std::cout << "  Mean inlier reproj err: " << r.err
                      << " px, Inliers: " << r.inliers << "/" << pts1.size()
                      << ", Time: " << r.time_ms << " ms\n";
        }
        return r;
    };

    std::vector<MethodResult> results;

    results.push_back(run("RANSAC", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::RANSAC, threshold, m, 2000, 0.99);
    }));
    results.push_back(run("LMEDS", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::LMEDS, threshold, m);
    }));
    results.push_back(run("USAC_DEFAULT", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_DEFAULT, threshold, m);
    }));
    results.push_back(run("USAC_MAGSAC", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, threshold, m);
    }));
    results.push_back(run("USAC_PROSAC", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_PROSAC, threshold, m);
    }));
    results.push_back(run("USAC_ACCURATE", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_ACCURATE, threshold, m);
    }));

    cv::UsacParams usacParams;
    usacParams.sampler = cv::SAMPLING_PROSAC;
    usacParams.score = cv::SCORE_METHOD_MAGSAC;
    usacParams.loMethod = cv::LOCAL_OPTIM_INNER_AND_ITER_LO;
    usacParams.loIterations = 15;
    usacParams.final_polisher = cv::MAGSAC;
    usacParams.final_polisher_iterations = 10;
    usacParams.threshold = threshold;
    usacParams.confidence = 0.999;
    usacParams.maxIterations = 5000;
    usacParams.isParallel = true;
    results.push_back(run("Custom UsacParams", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, m, usacParams);
    }));

    std::cout << "\n=== Summary ===\n";
    std::cout << std::left << std::setw(20) << "Method"
              << std::right << std::setw(12) << "Reproj(px)"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(56, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.name
                  << std::right << std::setw(12) << r.err
                  << std::setw(12) << r.inliers
                  << std::setw(12) << r.time_ms << "\n";
    }
    return 0;
}
