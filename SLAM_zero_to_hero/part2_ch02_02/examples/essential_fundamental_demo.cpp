/**
 * Essential and Fundamental Matrix Estimation
 *
 * Pipeline on a real KITTI stereo pair:
 * 1. ORB feature detection and matching between the two images
 * 2. Fundamental matrix F (8-point + RANSAC, uncalibrated)
 * 3. Essential matrix E (5-point + RANSAC, calibrated with KITTI intrinsics)
 * 4. Pose recovery from E, compared to the known KITTI stereo extrinsic
 * 5. The relationship E = K'^T * F * K
 *
 * Images default to the bundled data/ pair; pass two paths to override.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

/**
 * Resolve a file inside the bundled data/ folder, trying ../data first so it
 * works when run from build/ (and data/ when run from the project root).
 */
static std::string resolveDataPath(const std::string& name) {
    for (const std::string& base : {"../data/", "data/", "./data/"}) {
        if (std::filesystem::exists(base + name)) return base + name;
    }
    return "../data/" + name;
}

/**
 * Detect and match ORB features between two images (Lowe's ratio test).
 */
void detectAndMatchFeatures(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2) {

    auto orb = cv::ORB::create(2000);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    std::cout << "Keypoints: " << kp1.size() << " in image 1, "
              << kp2.size() << " in image 2" << std::endl;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    pts1.clear();
    pts2.clear();
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
            pts1.push_back(kp1[m[0].queryIdx].pt);
            pts2.push_back(kp2[m[0].trainIdx].pt);
        }
    }

    std::cout << "Good matches: " << pts1.size() << std::endl;
}

/**
 * Average absolute epipolar constraint residual |x2^T F x1| over correspondences.
 */
double verifyEpipolarConstraint(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& F) {

    if (pts1.empty()) return 0.0;
    double total_error = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat err = x2.t() * F * x1;
        total_error += std::abs(err.at<double>(0, 0));
    }
    return total_error / pts1.size();
}

int main(int argc, char* argv[]) {
    std::cout << "=== Essential and Fundamental Matrix Estimation (KITTI stereo pair) ===\n"
              << std::endl;

    // Real KITTI stereo pair (cam0/cam1). Override with two image paths.
    std::string left  = (argc >= 3) ? argv[1] : resolveDataPath("left.png");
    std::string right = (argc >= 3) ? argv[2] : resolveDataPath("right.png");

    cv::Mat img1 = cv::imread(left, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(right, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: failed to load images:\n  " << left << "\n  " << right
                  << "\nPass two image paths, or run from build/ so ../data resolves."
                  << std::endl;
        return 1;
    }
    std::cout << "Left:  " << left << "  (" << img1.cols << "x" << img1.rows << ")\n";
    std::cout << "Right: " << right << "  (" << img2.cols << "x" << img2.rows << ")\n"
              << std::endl;

    // KITTI odometry seq 00-02 rectified intrinsics (image size 1241x376).
    const double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);
    std::cout << "Camera Intrinsic Matrix K (KITTI):\n" << K << std::endl << std::endl;

    // Ground-truth extrinsic (cam1 relative to cam0): P1 = K [I | t_gt],
    // t_gt = [P1(0,3)/fx, 0, 0] = [-0.5372, 0, 0] m, no relative rotation.
    const double baseline_m = 386.1448 / fx;   // ~0.5372 m
    cv::Mat R_gt = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_gt = (cv::Mat_<double>(3, 1) << -baseline_m, 0.0, 0.0);

    // ORB correspondences from the real pair.
    std::vector<cv::Point2f> pts1, pts2;
    detectAndMatchFeatures(img1, img2, pts1, pts2);
    if (pts1.size() < 8) {
        std::cerr << "Not enough matches for matrix estimation!" << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // =========================================================
    // Fundamental Matrix: RANSAC (robust) then 8-point on inliers
    // =========================================================
    std::cout << "=== RANSAC (Fundamental Matrix) ===" << std::endl;

    cv::Mat inlier_mask;
    cv::Mat F_ransac = cv::findFundamentalMat(
        pts1, pts2, cv::FM_RANSAC, 3.0, 0.99, inlier_mask);

    std::vector<cv::Point2f> in1, in2;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (inlier_mask.at<uchar>(i)) {
            in1.push_back(pts1[i]);
            in2.push_back(pts2[i]);
        }
    }
    std::cout << "Inliers: " << in1.size() << "/" << pts1.size() << std::endl;
    std::cout << "Estimated F (RANSAC):\n" << F_ransac << std::endl;
    std::cout << "Average epipolar error (inliers): "
              << verifyEpipolarConstraint(in1, in2, F_ransac) << std::endl;
    std::cout << std::endl;

    std::cout << "=== 8-Point Algorithm (Fundamental Matrix, on inliers) ===" << std::endl;
    cv::Mat F_8point;
    if (in1.size() >= 8) {
        F_8point = cv::findFundamentalMat(in1, in2, cv::FM_8POINT);
        std::cout << "Estimated F (8-point):\n" << F_8point << std::endl;
        std::cout << "Average epipolar error (inliers): "
                  << verifyEpipolarConstraint(in1, in2, F_8point) << std::endl;
    } else {
        std::cout << "Not enough inliers for a stable 8-point fit." << std::endl;
    }
    std::cout << std::endl;

    // =========================================================
    // Essential Matrix: 5-Point + RANSAC (calibrated)
    // =========================================================
    std::cout << "=== 5-Point Algorithm (Essential Matrix) ===" << std::endl;

    cv::Mat E_mask;
    cv::Mat E = cv::findEssentialMat(
        pts1, pts2, K, cv::RANSAC, 0.999, 1.0, E_mask);

    int num_inliers = cv::countNonZero(E_mask);
    std::cout << "Inliers: " << num_inliers << "/" << pts1.size() << std::endl;
    std::cout << "Estimated E (5-point):\n" << E << std::endl;

    cv::Mat U, S, Vt;
    cv::SVD::compute(E, S, U, Vt);
    std::cout << "Essential Matrix Singular Values:" << std::endl;
    std::cout << "  s1 = " << S.at<double>(0) << std::endl;
    std::cout << "  s2 = " << S.at<double>(1) << " (should ~= s1)" << std::endl;
    std::cout << "  s3 = " << S.at<double>(2) << " (should be ~0)" << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Recover Pose from Essential Matrix + compare to ground truth
    // =========================================================
    std::cout << "=== Pose Recovery from Essential Matrix ===" << std::endl;

    cv::Mat R_recovered, t_recovered;
    int num_good = cv::recoverPose(
        E, pts1, pts2, K, R_recovered, t_recovered, E_mask);

    std::cout << "Points in front of both cameras: " << num_good << std::endl;
    std::cout << "Recovered Rotation:\n" << R_recovered << std::endl;
    std::cout << "Recovered Translation (up to scale):\n" << t_recovered.t() << std::endl;
    std::cout << std::endl;

    std::cout << "=== Comparison with Ground-Truth Extrinsic ===" << std::endl;
    std::cout << "GT: R = I, t direction = " << (t_gt / cv::norm(t_gt)).t()
              << "  (KITTI baseline " << baseline_m << " m)" << std::endl;

    cv::Mat dR = R_recovered * R_gt.t();
    cv::Mat rvec_err;
    cv::Rodrigues(dR, rvec_err);
    double rot_err_deg = cv::norm(rvec_err) * 180.0 / CV_PI;

    cv::Mat t_gt_unit = t_gt / cv::norm(t_gt);
    double cos_ang = std::abs(t_recovered.dot(t_gt_unit));
    cos_ang = std::min(1.0, std::max(-1.0, cos_ang));
    double t_err_deg = std::acos(cos_ang) * 180.0 / CV_PI;

    std::cout << "Rotation error vs identity:   " << rot_err_deg << " deg" << std::endl;
    std::cout << "Translation direction error:  " << t_err_deg << " deg" << std::endl;
    std::cout << "(Sign of t may flip with image order; direction error uses |cos|.)"
              << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Relationship between F and E
    // =========================================================
    std::cout << "=== Relationship: E = K'^T * F * K ===" << std::endl;

    cv::Mat E_from_F = K.t() * F_ransac * K;
    E_from_F = E_from_F / cv::norm(E_from_F, cv::NORM_L2);
    cv::Mat E_norm = E / cv::norm(E, cv::NORM_L2);

    std::cout << "E from F (normalized):\n" << E_from_F << std::endl;
    std::cout << "E from 5-point (normalized):\n" << E_norm << std::endl;

    double diff1 = cv::norm(E_from_F - E_norm, cv::NORM_L2);
    double diff2 = cv::norm(E_from_F + E_norm, cv::NORM_L2);  // sign ambiguity
    std::cout << "Difference (Frobenius, sign-resolved): "
              << std::min(diff1, diff2) << std::endl;
    std::cout << std::endl;

    std::cout << "=== Demo Complete ===" << std::endl;

    return 0;
}
