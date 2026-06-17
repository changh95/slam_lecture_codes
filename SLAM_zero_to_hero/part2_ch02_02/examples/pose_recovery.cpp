/**
 * Pose Recovery from Essential Matrix
 *
 * Pipeline on a real KITTI stereo pair:
 * 1. ORB feature detection and matching between the two images
 * 2. Essential matrix estimation with RANSAC (KITTI intrinsics)
 * 3. Decomposition of E into four possible (R, t) pairs
 * 4. Cheirality check to select the correct solution
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
 * Detect and match ORB features between two images
 */
void detectAndMatchFeatures(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2,
    std::vector<cv::DMatch>& good_matches) {

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

    // Lowe's ratio test
    const float ratio_thresh = 0.75f;
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    std::cout << "Good matches: " << good_matches.size() << std::endl;

    pts1.clear();
    pts2.clear();
    for (const auto& m : good_matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
}

/**
 * Triangulate a single point from two views (DLT)
 */
cv::Mat triangulatePoint(
    const cv::Mat& P1,
    const cv::Mat& P2,
    const cv::Point2f& pt1,
    const cv::Point2f& pt2) {

    cv::Mat A(4, 4, CV_64F);

    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);

    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt);

    cv::Mat X = Vt.row(3).t();
    X = X / X.at<double>(3);

    return X.rowRange(0, 3).clone();
}

/**
 * Check cheirality (points in front of both cameras)
 */
int countPointsInFront(
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Mat& K,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2) {

    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);  // [I | 0]

    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P2 = K * Rt;

    int count = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat X = triangulatePoint(P1, P2, pts1[i], pts2[i]);

        double z1 = X.at<double>(2);
        cv::Mat X2 = R * X + t;
        double z2 = X2.at<double>(2);

        if (z1 > 0 && z2 > 0) {
            count++;
        }
    }

    return count;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Pose Recovery from Essential Matrix (KITTI stereo pair) ===\n"
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

    std::cout << "Camera Intrinsics K (KITTI):\n" << K << std::endl << std::endl;

    // Ground-truth extrinsic of the KITTI stereo rig (cam1 relative to cam0),
    // from the seq 00-02 projection matrices: P1 = K [I | t_gt] with
    // t_gt = [P1(0,3)/fx, 0, 0] = [-0.5372, 0, 0] m and no relative rotation
    // (rectified pair). recoverPose returns t up to scale, so we compare the
    // recovered translation *direction* against this baseline.
    const double baseline_m = 386.1448 / fx;   // ~0.5372 m
    cv::Mat R_gt = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_gt = (cv::Mat_<double>(3, 1) << -baseline_m, 0.0, 0.0);
    std::cout << "Ground-truth extrinsic (KITTI stereo, cam1 rel. cam0):\n";
    std::cout << "  R_gt = I,  t_gt = " << t_gt.t()
              << "  (baseline " << baseline_m << " m)\n" << std::endl;

    // Detect and match features
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<cv::DMatch> matches;
    detectAndMatchFeatures(img1, img2, pts1, pts2, matches);

    if (pts1.size() < 8) {
        std::cerr << "Not enough matches for Essential matrix estimation!" << std::endl;
        return 1;
    }

    // Estimate Essential matrix
    std::cout << "\n=== Essential Matrix Estimation ===" << std::endl;

    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        pts1, pts2, K,
        cv::RANSAC,
        0.999,      // Confidence
        1.0,        // RANSAC threshold in pixels
        inlier_mask
    );

    int num_inliers = cv::countNonZero(inlier_mask);
    std::cout << "RANSAC inliers: " << num_inliers << "/" << pts1.size() << std::endl;
    std::cout << "Essential Matrix E:\n" << E << std::endl;

    cv::Mat U, S, Vt;
    cv::SVD::compute(E, S, U, Vt);
    std::cout << "\nSVD of E:" << std::endl;
    std::cout << "  Singular values: " << S.t() << std::endl;
    double sv_ratio = S.at<double>(0) / S.at<double>(1);
    std::cout << "  s1/s2 ratio: " << sv_ratio << " (should be ~1)" << std::endl;
    std::cout << "  s3: " << S.at<double>(2) << " (should be ~0)" << std::endl;

    // Decompose Essential matrix
    std::cout << "\n=== Pose Decomposition ===" << std::endl;

    cv::Mat R1, R2, t_decomp;
    cv::decomposeEssentialMat(E, R1, R2, t_decomp);

    std::cout << "Possible Rotations:" << std::endl;
    std::cout << "R1:\n" << R1 << std::endl;
    std::cout << "R2:\n" << R2 << std::endl;
    std::cout << "t (direction):\n" << t_decomp.t() << std::endl;

    // Four possible solutions: (R1, t), (R1, -t), (R2, t), (R2, -t)
    std::cout << "\n=== Cheirality Check ===" << std::endl;

    std::vector<cv::Point2f> pts1_inliers, pts2_inliers;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (inlier_mask.at<uchar>(i)) {
            pts1_inliers.push_back(pts1[i]);
            pts2_inliers.push_back(pts2[i]);
        }
    }

    std::vector<std::pair<cv::Mat, cv::Mat>> solutions = {
        {R1, t_decomp},
        {R1, -t_decomp},
        {R2, t_decomp},
        {R2, -t_decomp}
    };

    int best_count = 0;
    int best_idx = 0;

    for (size_t i = 0; i < solutions.size(); ++i) {
        int count = countPointsInFront(
            solutions[i].first, solutions[i].second, K,
            pts1_inliers, pts2_inliers);

        std::cout << "Solution " << (i + 1) << ": "
                  << count << "/" << pts1_inliers.size()
                  << " points in front" << std::endl;

        if (count > best_count) {
            best_count = count;
            best_idx = i;
        }
    }

    std::cout << "\nBest solution: " << (best_idx + 1)
              << " (" << best_count << " valid points)" << std::endl;

    // Use recoverPose for verified solution
    std::cout << "\n=== OpenCV recoverPose ===" << std::endl;

    cv::Mat R_final, t_final;
    int valid_points = cv::recoverPose(E, pts1, pts2, K, R_final, t_final, inlier_mask);

    std::cout << "Valid points (cheirality): " << valid_points << std::endl;
    std::cout << "\nFinal Rotation:\n" << R_final << std::endl;
    std::cout << "\nFinal Translation (unit):\n" << t_final.t() << std::endl;

    cv::Mat rvec;
    cv::Rodrigues(R_final, rvec);
    double angle = cv::norm(rvec) * 180.0 / CV_PI;

    std::cout << "\n=== Interpretation ===" << std::endl;
    std::cout << "Rotation angle: " << angle << " degrees" << std::endl;
    std::cout << "Translation direction: [" << t_final.at<double>(0)
              << ", " << t_final.at<double>(1)
              << ", " << t_final.at<double>(2) << "]" << std::endl;
    std::cout << "(Note: translation scale is unknown from the Essential matrix alone;"
              << " for a rectified stereo pair expect near-pure X translation.)"
              << std::endl;

    // Compare against the known KITTI stereo extrinsic.
    std::cout << "\n=== Comparison with Ground-Truth Extrinsic ===" << std::endl;

    cv::Mat dR = R_final * R_gt.t();
    cv::Mat rvec_err;
    cv::Rodrigues(dR, rvec_err);
    double rot_err_deg = cv::norm(rvec_err) * 180.0 / CV_PI;

    cv::Mat t_gt_unit = t_gt / cv::norm(t_gt);
    double cos_ang = std::abs(t_final.dot(t_gt_unit));  // t_final is already unit
    cos_ang = std::min(1.0, std::max(-1.0, cos_ang));
    double t_err_deg = std::acos(cos_ang) * 180.0 / CV_PI;

    std::cout << "Rotation error vs identity:   " << rot_err_deg << " deg" << std::endl;
    std::cout << "Translation direction error:  " << t_err_deg << " deg" << std::endl;
    std::cout << "Recovered t (unit):  " << t_final.t() << std::endl;
    std::cout << "GT baseline (unit):  " << t_gt_unit.t() << std::endl;
    std::cout << "(Sign of t may flip with image order; direction error uses |cos|.)"
              << std::endl;

    std::cout << "\n=== Demo Complete ===" << std::endl;

    return 0;
}
