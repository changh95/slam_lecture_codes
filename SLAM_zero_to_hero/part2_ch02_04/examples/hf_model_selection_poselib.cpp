/**
 * @file hf_model_selection_poselib.cpp
 * @brief H/F Model Selection on Real Images using PoseLib minimal solvers
 *
 * Same experiment as hf_model_selection.cpp (ORB-SLAM style selection on
 * real image pairs), but the models are estimated with PoseLib minimal
 * solvers wrapped in a hand-rolled RANSAC:
 *   - poselib::homography_4pt for the homography
 *   - poselib::relpose_5pt for the essential matrix
 *
 * PoseLib works on bearing vectors (normalized image coordinates), so
 * pixels are first mapped through K^-1. The estimated models are mapped
 * back to the pixel domain (H_px = K * H_n * K^-1, F = K^-T * E * K^-1)
 * and scored with the same truncated chi-square scores as the OpenCV
 * version, so the two demos are directly comparable.
 *
 * Real image pairs:
 *   - wall  (Oxford VGG): planar brick wall            -> expect H
 *   - KITTI seq 00, 24->25: forward motion, 3D street  -> expect F
 *   - KITTI seq 00, 3677->3682: turning with parallax  -> expect F
 *
 * Note: KITTI pairs use the true calibration; the wall camera is
 * unknown, so an approximate K is used there (fine for H, which K only
 * normalizes; the essential matrix is then also approximate, but the
 * planar wall is a degenerate scene for E anyway).
 *
 * PoseLib: https://github.com/PoseLib/PoseLib
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <PoseLib/poselib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// Chi-square 95% thresholds and common truncation (ORB-SLAM values)
static constexpr double TH_H = 5.991;
static constexpr double TH_F = 3.841;
static constexpr double GAMMA = 5.991;

/**
 * @brief Detect ORB features and match with ratio test
 */
static void detectAndMatch(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<Eigen::Vector2d>& pts1,
    std::vector<Eigen::Vector2d>& pts2) {

    cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    pts1.clear();
    pts2.clear();
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 && m[0].distance < 0.7f * m[1].distance) {
            const auto& p1 = kp1[m[0].queryIdx].pt;
            const auto& p2 = kp2[m[0].trainIdx].pt;
            pts1.emplace_back(p1.x, p1.y);
            pts2.emplace_back(p2.x, p2.y);
        }
    }
}

/**
 * @brief Map pixels to bearing vectors through K^-1
 */
static std::vector<Eigen::Vector3d> toBearings(
    const std::vector<Eigen::Vector2d>& pts, const Eigen::Matrix3d& K_inv) {

    std::vector<Eigen::Vector3d> bearings;
    bearings.reserve(pts.size());
    for (const auto& p : pts) {
        bearings.push_back((K_inv * p.homogeneous()).normalized());
    }
    return bearings;
}

/**
 * @brief Skew-symmetric matrix [v]_x such that [v]_x * w = v x w
 */
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return S;
}

/**
 * @brief Random sample of k distinct indices in [0, n)
 */
static std::vector<int> randomSample(int n, int k, std::mt19937& gen) {
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    indices.resize(k);
    return indices;
}

/**
 * @brief Squared symmetric transfer error of a pixel-domain homography
 */
static double transferError2(
    const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,
    const Eigen::Matrix3d& H, const Eigen::Matrix3d& H_inv,
    double& e12, double& e21) {

    Eigen::Vector3d q2 = H * p1.homogeneous();
    e12 = (q2.hnormalized() - p2).squaredNorm();
    Eigen::Vector3d q1 = H_inv * p2.homogeneous();
    e21 = (q1.hnormalized() - p1).squaredNorm();
    return e12 + e21;
}

/**
 * @brief RANSAC homography via poselib::homography_4pt, returned in pixels
 */
static Eigen::Matrix3d ransacHomography(
    const std::vector<Eigen::Vector2d>& pts1,
    const std::vector<Eigen::Vector2d>& pts2,
    const std::vector<Eigen::Vector3d>& b1,
    const std::vector<Eigen::Vector3d>& b2,
    const Eigen::Matrix3d& K,
    double threshold_px,
    int max_iterations,
    int& best_inliers) {

    const Eigen::Matrix3d K_inv = K.inverse();
    const double th2 = threshold_px * threshold_px;

    std::mt19937 gen(42);
    best_inliers = 0;
    Eigen::Matrix3d best_H = Eigen::Matrix3d::Identity();

    for (int iter = 0; iter < max_iterations; ++iter) {
        auto idx = randomSample(static_cast<int>(b1.size()), 4, gen);
        std::vector<Eigen::Vector3d> x1, x2;
        for (int i : idx) {
            x1.push_back(b1[i]);
            x2.push_back(b2[i]);
        }

        Eigen::Matrix3d H_norm;
        if (poselib::homography_4pt(x1, x2, &H_norm) == 0) continue;

        // Back to pixel domain
        Eigen::Matrix3d H = K * H_norm * K_inv;
        Eigen::Matrix3d H_inv = H.inverse();

        int inliers = 0;
        for (size_t i = 0; i < pts1.size(); ++i) {
            double e12, e21;
            transferError2(pts1[i], pts2[i], H, H_inv, e12, e21);
            if (e12 < th2 && e21 < th2) inliers++;
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_H = H;
        }
    }
    return best_H;
}

/**
 * @brief RANSAC essential matrix via poselib::relpose_5pt, returned as
 *        a pixel-domain fundamental matrix F = K^-T * E * K^-1
 */
static Eigen::Matrix3d ransacFundamental(
    const std::vector<Eigen::Vector2d>& pts1,
    const std::vector<Eigen::Vector2d>& pts2,
    const std::vector<Eigen::Vector3d>& b1,
    const std::vector<Eigen::Vector3d>& b2,
    const Eigen::Matrix3d& K,
    double threshold_px,
    int max_iterations,
    int& best_inliers) {

    const Eigen::Matrix3d K_inv = K.inverse();
    const double th2 = threshold_px * threshold_px;

    std::mt19937 gen(42);
    best_inliers = 0;
    Eigen::Matrix3d best_F = Eigen::Matrix3d::Identity();

    for (int iter = 0; iter < max_iterations; ++iter) {
        auto idx = randomSample(static_cast<int>(b1.size()), 5, gen);
        std::vector<Eigen::Vector3d> x1, x2;
        for (int i : idx) {
            x1.push_back(b1[i]);
            x2.push_back(b2[i]);
        }

        std::vector<poselib::CameraPose> solutions;
        poselib::relpose_5pt(x1, x2, &solutions);

        for (const auto& pose : solutions) {
            // E = [t]_x * R, then F in pixels
            Eigen::Matrix3d E = skew(pose.t) * pose.R();
            Eigen::Matrix3d F = K_inv.transpose() * E * K_inv;

            int inliers = 0;
            for (size_t i = 0; i < pts1.size(); ++i) {
                // Squared point-to-epipolar-line distance in image 2
                Eigen::Vector3d l2 = F * pts1[i].homogeneous();
                double num = l2.dot(pts2[i].homogeneous());
                double e = num * num / l2.head<2>().squaredNorm();
                if (e < th2) inliers++;
            }
            if (inliers > best_inliers) {
                best_inliers = inliers;
                best_F = F;
            }
        }
    }
    return best_F;
}

/**
 * @brief ORB-SLAM truncated score for a pixel-domain homography
 */
static double scoreHomography(
    const std::vector<Eigen::Vector2d>& pts1,
    const std::vector<Eigen::Vector2d>& pts2,
    const Eigen::Matrix3d& H) {

    Eigen::Matrix3d H_inv = H.inverse();
    double score = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        double e12, e21;
        transferError2(pts1[i], pts2[i], H, H_inv, e12, e21);
        if (e12 < TH_H) score += GAMMA - e12;
        if (e21 < TH_H) score += GAMMA - e21;
    }
    return score;
}

/**
 * @brief ORB-SLAM truncated score for a pixel-domain fundamental matrix
 */
static double scoreFundamental(
    const std::vector<Eigen::Vector2d>& pts1,
    const std::vector<Eigen::Vector2d>& pts2,
    const Eigen::Matrix3d& F) {

    double score = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        Eigen::Vector3d l2 = F * pts1[i].homogeneous();
        double num2 = l2.dot(pts2[i].homogeneous());
        double e2 = num2 * num2 / l2.head<2>().squaredNorm();

        Eigen::Vector3d l1 = F.transpose() * pts2[i].homogeneous();
        double num1 = l1.dot(pts1[i].homogeneous());
        double e1 = num1 * num1 / l1.head<2>().squaredNorm();

        if (e2 < TH_F) score += GAMMA - e2;
        if (e1 < TH_F) score += GAMMA - e1;
    }
    return score;
}

/**
 * @brief Run selection on one real image pair
 */
static void selectModel(
    const std::string& name,
    const cv::Mat& img1,
    const cv::Mat& img2,
    const Eigen::Matrix3d& K,
    const std::string& expectation) {

    std::cout << "\n=== " << name << " ===" << std::endl;

    std::vector<Eigen::Vector2d> pts1, pts2;
    detectAndMatch(img1, img2, pts1, pts2);
    std::cout << "  Matches: " << pts1.size() << std::endl;
    if (pts1.size() < 20) {
        std::cerr << "  Error: not enough matches" << std::endl;
        return;
    }

    const Eigen::Matrix3d K_inv = K.inverse();
    auto b1 = toBearings(pts1, K_inv);
    auto b2 = toBearings(pts2, K_inv);

    int inliers_H = 0, inliers_F = 0;
    Eigen::Matrix3d H =
        ransacHomography(pts1, pts2, b1, b2, K, 3.0, 2000, inliers_H);
    Eigen::Matrix3d F =
        ransacFundamental(pts1, pts2, b1, b2, K, 3.0, 2000, inliers_F);

    double S_H = scoreHomography(pts1, pts2, H);
    double S_F = scoreFundamental(pts1, pts2, F);
    double R_H = (S_H + S_F > 0) ? S_H / (S_H + S_F) : 0.5;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  H inliers (4pt RANSAC): " << inliers_H
              << ", E inliers (5pt RANSAC): " << inliers_F << std::endl;
    std::cout << "  S_H = " << S_H << ", S_F = " << S_F << std::endl;
    std::cout << "  R_H = S_H / (S_H + S_F) = " << R_H << std::endl;

    if (R_H > 0.45) {
        std::cout << "  --> Selected: HOMOGRAPHY (planar or low-parallax scene)"
                  << std::endl;
    } else {
        std::cout << "  --> Selected: ESSENTIAL/FUNDAMENTAL (3D scene with parallax)"
                  << std::endl;
    }
    std::cout << "  Expected: " << expectation << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "H/F Model Selection on Real Images (PoseLib minimal solvers)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    // KITTI odometry seq 00, camera 0 intrinsics (P0 in calib.txt)
    Eigen::Matrix3d K_kitti;
    K_kitti << 718.856, 0.0, 607.1928,
               0.0, 718.856, 185.2157,
               0.0, 0.0, 1.0;

    cv::Mat wall1 = cv::imread(data_dir + "/wall_img1.png");
    cv::Mat wall3 = cv::imread(data_dir + "/wall_img3.png");
    cv::Mat fwd1 = cv::imread(data_dir + "/kitti00_fwd_000024.png");
    cv::Mat fwd2 = cv::imread(data_dir + "/kitti00_fwd_000025.png");
    cv::Mat turn1 = cv::imread(data_dir + "/kitti00_turn_003677.png");
    cv::Mat turn2 = cv::imread(data_dir + "/kitti00_turn_003682.png");
    if (wall1.empty() || wall3.empty() || fwd1.empty() || fwd2.empty() ||
        turn1.empty() || turn2.empty()) {
        std::cerr << "Error: could not load images from " << data_dir
                  << " (run from the chapter root, or pass the data dir as arg)"
                  << std::endl;
        return 1;
    }

    // Approximate K for the wall camera (unknown calibration):
    // fx = fy = image width, principal point at the center.
    Eigen::Matrix3d K_wall;
    K_wall << wall1.cols, 0.0, wall1.cols / 2.0,
              0.0, wall1.cols, wall1.rows / 2.0,
              0.0, 0.0, 1.0;

    selectModel("Planar scene: VGG wall img1 -> img3 (brick wall)",
                wall1, wall3, K_wall, "H (all matches lie on one plane)");
    selectModel("3D scene: KITTI 00 frame 24 -> 25 (forward motion)",
                fwd1, fwd2, K_kitti, "F (street scene with depth, translating camera)");
    selectModel("Turning: KITTI 00 frame 3677 -> 3682 (~21 deg turn)",
                turn1, turn2, K_kitti, "F (rotation-dominant, but still real parallax)");

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "  - PoseLib solvers work on bearing vectors (K^-1 * pixel)" << std::endl;
    std::cout << "  - homography_4pt is a minimal 4-point solver" << std::endl;
    std::cout << "  - relpose_5pt returns up to 10 essential matrix solutions" << std::endl;
    std::cout << "  - Scores are computed in pixels, identical to the OpenCV demo" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
