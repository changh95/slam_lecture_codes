/**
 * @file hf_model_selection.cpp
 * @brief H/F Model Selection on Real Images (ORB-SLAM Style)
 *
 * ORB-SLAM decides how to initialize the map by fitting both a
 * Homography and a Fundamental matrix to the first two views and
 * keeping the model that explains the matches better:
 *
 *   1. Compute H and F in parallel with RANSAC
 *   2. Score each model on ALL matches with a truncated, chi-square
 *      based score (Mur-Artal et al., TRO 2015, Sec. IV.A):
 *        S_H: symmetric transfer error,  threshold 5.991 (chi2 95%, 2 DOF)
 *        S_F: epipolar (point-line) error, threshold 3.841 (chi2 95%, 1 DOF)
 *      Both scores truncate at Gamma = 5.991 so they are comparable.
 *   3. R_H = S_H / (S_H + S_F).  If R_H > 0.45 -> planar / low-parallax
 *      scene, use H; otherwise use F (then E for the pose).
 *
 * Real image pairs used here:
 *   - wall  (Oxford VGG): planar brick wall            -> expect H
 *   - KITTI seq 00, 24->25: forward motion, 3D street  -> expect F
 *   - KITTI seq 00, 3677->3682: turning, but ~2.4 m of
 *     translation gives real parallax                  -> expect F
 */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Chi-square 95% thresholds and common truncation (ORB-SLAM values)
static constexpr double TH_H = 5.991;   // 2-DOF error (2D transfer distance^2)
static constexpr double TH_F = 3.841;   // 1-DOF error (point-line distance^2)
static constexpr double GAMMA = 5.991;  // common truncation for comparability

/**
 * @brief Detect ORB features and match with ratio test
 */
static void detectAndMatch(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2) {

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
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }
}

/**
 * @brief ORB-SLAM score for a homography: truncated symmetric transfer error
 */
static double scoreHomography(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& H) {

    if (H.empty()) return 0.0;
    cv::Mat H_inv = H.inv();

    double score = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        // Forward: pts1 -> pts2
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2_proj = H * p1;
        p2_proj /= p2_proj.at<double>(2);
        double dx = p2_proj.at<double>(0) - pts2[i].x;
        double dy = p2_proj.at<double>(1) - pts2[i].y;
        double e12 = dx * dx + dy * dy;

        // Backward: pts2 -> pts1
        cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1_proj = H_inv * p2;
        p1_proj /= p1_proj.at<double>(2);
        dx = p1_proj.at<double>(0) - pts1[i].x;
        dy = p1_proj.at<double>(1) - pts1[i].y;
        double e21 = dx * dx + dy * dy;

        if (e12 < TH_H) score += GAMMA - e12;
        if (e21 < TH_H) score += GAMMA - e21;
    }
    return score;
}

/**
 * @brief ORB-SLAM score for a fundamental matrix: truncated epipolar error
 *
 * Uses the squared point-to-epipolar-line distance in both images.
 */
static double scoreFundamental(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& F) {

    if (F.empty()) return 0.0;

    double score = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

        // Epipolar line in image 2 and distance of x2 to it
        cv::Mat l2 = F * x1;
        double a2 = l2.at<double>(0), b2 = l2.at<double>(1);
        double num2 = x2.dot(l2);
        double e2 = num2 * num2 / (a2 * a2 + b2 * b2);

        // Epipolar line in image 1 and distance of x1 to it
        cv::Mat l1 = F.t() * x2;
        double a1 = l1.at<double>(0), b1 = l1.at<double>(1);
        double num1 = x1.dot(l1);
        double e1 = num1 * num1 / (a1 * a1 + b1 * b1);

        if (e2 < TH_F) score += GAMMA - e2;
        if (e1 < TH_F) score += GAMMA - e1;
    }
    return score;
}

/**
 * @brief Fit H and F, score both, and select the model
 */
static void selectModel(
    const std::string& name,
    const cv::Mat& img1,
    const cv::Mat& img2,
    const std::string& expectation) {

    std::cout << "\n=== " << name << " ===" << std::endl;

    std::vector<cv::Point2f> pts1, pts2;
    detectAndMatch(img1, img2, pts1, pts2);
    std::cout << "  Matches: " << pts1.size() << std::endl;
    if (pts1.size() < 20) {
        std::cerr << "  Error: not enough matches" << std::endl;
        return;
    }

    cv::Mat mask_H, mask_F;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask_H, 2000, 0.995);
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.995, mask_F);

    double S_H = scoreHomography(pts1, pts2, H);
    double S_F = scoreFundamental(pts1, pts2, F);
    double R_H = (S_H + S_F > 0) ? S_H / (S_H + S_F) : 0.5;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  H inliers: " << cv::countNonZero(mask_H) << ", F inliers: "
              << cv::countNonZero(mask_F) << std::endl;
    std::cout << "  S_H = " << S_H << ", S_F = " << S_F << std::endl;
    std::cout << "  R_H = S_H / (S_H + S_F) = " << R_H << std::endl;

    if (R_H > 0.45) {
        std::cout << "  --> Selected: HOMOGRAPHY (planar or low-parallax scene)"
                  << std::endl;
    } else {
        std::cout << "  --> Selected: FUNDAMENTAL MATRIX (3D scene with parallax)"
                  << std::endl;
    }
    std::cout << "  Expected: " << expectation << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "H/F Model Selection on Real Images (ORB-SLAM style)" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Custom pair from the command line
    if (argc == 3) {
        cv::Mat img1 = cv::imread(argv[1]);
        cv::Mat img2 = cv::imread(argv[2]);
        if (img1.empty() || img2.empty()) {
            std::cerr << "Error: could not load " << argv[1] << " / " << argv[2]
                      << std::endl;
            return 1;
        }
        selectModel("Custom pair", img1, img2, "unknown");
        return 0;
    }

    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    struct Case {
        std::string name, file1, file2, expectation;
    };
    const std::vector<Case> cases = {
        {"Planar scene: VGG wall img1 -> img3 (brick wall, viewpoint change)",
         "wall_img1.png", "wall_img3.png",
         "H (all matches lie on one plane)"},
        {"3D scene: KITTI 00 frame 24 -> 25 (forward motion, ~0.9 m)",
         "kitti00_fwd_000024.png", "kitti00_fwd_000025.png",
         "F (street scene with depth, translating camera)"},
        {"Turning: KITTI 00 frame 3677 -> 3682 (~21 deg turn, ~2.4 m)",
         "kitti00_turn_003677.png", "kitti00_turn_003682.png",
         "F (rotation-dominant, but 2.4 m of travel still creates parallax)"},
    };

    for (const auto& c : cases) {
        cv::Mat img1 = cv::imread(data_dir + "/" + c.file1);
        cv::Mat img2 = cv::imread(data_dir + "/" + c.file2);
        if (img1.empty() || img2.empty()) {
            std::cerr << "Error: could not load images from " << data_dir
                      << " (run from the chapter root, or pass the data dir as arg)"
                      << std::endl;
            return 1;
        }
        selectModel(c.name, img1, img2, c.expectation);
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - A single plane in view -> R_H > 0.45 -> initialize with H" << std::endl;
    std::cout << "  - Depth + translation    -> R_H < 0.45 -> initialize with F" << std::endl;
    std::cout << "  - Even a turning car has parallax: rotation alone does not" << std::endl;
    std::cout << "    make H win; the scene must be planar or the translation" << std::endl;
    std::cout << "    negligible relative to scene depth." << std::endl;
    std::cout << "==================================================" << std::endl;
    return 0;
}
