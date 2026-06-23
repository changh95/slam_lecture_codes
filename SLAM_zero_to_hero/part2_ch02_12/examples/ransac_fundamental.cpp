/**
 * RANSAC Fundamental Matrix Estimation with USAC
 *
 * Run nine F-estimation methods on real ORB correspondences from a KITTI
 * consecutive frame pair (data/000024.png, data/000025.png by default).
 * No synthetic data: there is no closed-form GT F for arbitrary scenes, so
 * each method is judged on (a) Sampson error against the matches, and
 * (b) inlier count -- the same criteria the estimators themselves optimize.
 *
 * Methods:
 *   1. FM_7POINT     (7-point, single shot)
 *   2. FM_8POINT     (8-point, single shot)
 *   3. FM_RANSAC     (8-point + RANSAC)
 *   4. FM_LMEDS      (Least Median of Squares)
 *   5. USAC_DEFAULT  (PROSAC + MSAC + Inner-LO)
 *   6. USAC_FM_8PTS  (Uniform + RANSAC scoring, 8-point)
 *   7. USAC_MAGSAC   (MAGSAC++ marginalized scoring)
 *   8. Custom USAC   (PROSAC + MAGSAC + Inner&Iter-LO + MAGSAC polisher)
 *   9. USAC_ACCURATE (Graph-Cut local optim)
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

// Sampson distance over inliers selected by mask (or all points if mask empty).
static double meanSampson(const cv::Mat& F,
                          const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2,
                          const cv::Mat& mask = cv::Mat()) {
    if (F.empty() || F.rows < 3) return -1.0;
    double total = 0.0;
    int n = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!mask.empty() && !mask.at<uchar>(i)) continue;
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

struct MethodResult {
    std::string name;
    cv::Mat F;
    cv::Mat mask;
    double sampson;
    int inliers;
    double time_ms;
};

int main(int argc, char* argv[]) {
    std::cout << "=== RANSAC Fundamental Matrix Estimation (KITTI consecutive frames) ===\n";
    std::cout << std::fixed << std::setprecision(6);

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
    if (pts1.size() < 8) {
        std::cerr << "Not enough matches for F estimation.\n";
        return 1;
    }

    // KITTI seq 00-02 rectified intrinsics (for reference; F is uncalibrated).
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        718.856, 0.0, 607.1928,
        0.0, 718.856, 185.2157,
        0.0, 0.0, 1.0);
    std::cout << "\nReference K (KITTI seq 00-02):\n" << K << "\n\n";

    const double threshold = 3.0;
    Timer timer;

    auto run_masked = [&](const std::string& name, auto fn) -> MethodResult {
        MethodResult r{name, {}, {}, -1.0, 0, 0.0};
        timer.start();
        r.F = fn(r.mask);
        r.time_ms = timer.elapsedMs();
        if (!r.F.empty() && r.F.rows == 3) {
            r.sampson = meanSampson(r.F, pts1, pts2, r.mask);
            r.inliers = r.mask.empty() ? 0 : cv::countNonZero(r.mask);
        }
        std::cout << "\n--- " << name << " ---\n";
        if (r.F.empty()) {
            std::cout << "  (no solution)  Time: " << r.time_ms << " ms\n";
        } else if (r.F.rows != 3) {
            std::cout << "  Multiple solutions (" << r.F.rows / 3 << "), Time: "
                      << r.time_ms << " ms\n";
        } else {
            std::cout << "  Sampson err (inliers): " << r.sampson
                      << ", Inliers: " << r.inliers << "/" << pts1.size()
                      << ", Time: " << r.time_ms << " ms\n";
        }
        return r;
    };

    std::vector<MethodResult> results;

    results.push_back(run_masked("FM_7POINT", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_7POINT, threshold, 0.99, m);
    }));
    results.push_back(run_masked("FM_8POINT", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT, threshold, 0.99, m);
    }));
    results.push_back(run_masked("FM_RANSAC", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, threshold, 0.99, m);
    }));
    results.push_back(run_masked("FM_LMEDS", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_LMEDS, threshold, 0.99, m);
    }));
    results.push_back(run_masked("USAC_DEFAULT", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_DEFAULT, threshold, 0.99, m);
    }));
    results.push_back(run_masked("USAC_FM_8PTS", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_FM_8PTS, threshold, 0.99, m);
    }));
    results.push_back(run_masked("USAC_MAGSAC", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, threshold, 0.99, m);
    }));

    cv::UsacParams usacParams;
    usacParams.sampler = cv::SAMPLING_PROSAC;
    usacParams.score = cv::SCORE_METHOD_MAGSAC;
    usacParams.loMethod = cv::LOCAL_OPTIM_INNER_AND_ITER_LO;
    usacParams.loIterations = 15;
    usacParams.neighborsSearch = cv::NEIGH_FLANN_KNN;
    usacParams.final_polisher = cv::MAGSAC;
    usacParams.final_polisher_iterations = 10;
    usacParams.threshold = threshold;
    usacParams.confidence = 0.999;
    usacParams.maxIterations = 10000;
    usacParams.isParallel = true;
    results.push_back(run_masked("Custom USAC", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, m, usacParams);
    }));

    results.push_back(run_masked("USAC_ACCURATE", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_ACCURATE, threshold, 0.99, m);
    }));

    std::cout << "\n=== Summary ===\n";
    std::cout << std::left << std::setw(18) << "Method"
              << std::right << std::setw(14) << "Sampson"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(56, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(18) << r.name
                  << std::right << std::setw(14) << r.sampson
                  << std::setw(12) << r.inliers
                  << std::setw(12) << r.time_ms << "\n";
    }

    // Sanity check: epipolar constraint on USAC_MAGSAC inliers.
    const auto& magsac = results[6];
    if (!magsac.F.empty() && magsac.F.rows == 3 && !magsac.mask.empty()) {
        double total = 0.0;
        int n = 0;
        for (size_t i = 0; i < pts1.size(); ++i) {
            if (magsac.mask.at<uchar>(i)) {
                cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
                cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
                total += std::abs(p2.dot(magsac.F * p1));
                ++n;
            }
        }
        std::cout << "\nUSAC_MAGSAC mean |x2^T F x1| on inliers: "
                  << (n ? total / n : 0.0) << " (expected ~0)\n";
    }
    return 0;
}
