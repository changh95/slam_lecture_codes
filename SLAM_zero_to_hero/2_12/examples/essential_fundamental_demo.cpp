/**
 * Essential and Fundamental Matrix Estimation
 *
 * This example demonstrates:
 * 1. Finding correspondences between two images
 * 2. Estimating the Fundamental matrix (uncalibrated cameras)
 * 3. Estimating the Essential matrix (calibrated cameras)
 * 4. The relationship between F and E matrices
 *
 * Mathematical Background:
 * - Fundamental matrix F: x'^T * F * x = 0 (image coordinates)
 * - Essential matrix E: x'^T * E * x = 0 (normalized coordinates)
 * - Relationship: E = K'^T * F * K
 *
 * Reference: Hartley & Zisserman, "Multiple View Geometry"
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

/**
 * Generate synthetic 3D points and their projections
 */
void generateSyntheticData(
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t,
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2,
    int num_points = 50) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> x_dist(-2.0, 2.0);
    std::uniform_real_distribution<> y_dist(-2.0, 2.0);
    std::uniform_real_distribution<> z_dist(5.0, 15.0);

    pts1.clear();
    pts2.clear();

    for (int i = 0; i < num_points; ++i) {
        // Random 3D point
        cv::Mat P = (cv::Mat_<double>(3, 1) <<
            x_dist(gen), y_dist(gen), z_dist(gen));

        // Project to camera 1 (at origin)
        cv::Mat p1_h = K * P;
        cv::Point2f p1(p1_h.at<double>(0) / p1_h.at<double>(2),
                       p1_h.at<double>(1) / p1_h.at<double>(2));

        // Transform point to camera 2 frame
        cv::Mat P2 = R * P + t;

        // Project to camera 2
        cv::Mat p2_h = K * P2;
        cv::Point2f p2(p2_h.at<double>(0) / p2_h.at<double>(2),
                       p2_h.at<double>(1) / p2_h.at<double>(2));

        // Only add if both projections are within image bounds
        if (p1.x > 0 && p1.x < 640 && p1.y > 0 && p1.y < 480 &&
            p2.x > 0 && p2.x < 640 && p2.y > 0 && p2.y < 480) {
            pts1.push_back(p1);
            pts2.push_back(p2);
        }
    }
}

/**
 * Add noise to point correspondences
 */
void addNoise(std::vector<cv::Point2f>& pts, double sigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, sigma);

    for (auto& p : pts) {
        p.x += noise(gen);
        p.y += noise(gen);
    }
}

/**
 * Convert skew-symmetric matrix to vector
 */
cv::Mat skewSymmetric(const cv::Mat& v) {
    return (cv::Mat_<double>(3, 3) <<
        0, -v.at<double>(2), v.at<double>(1),
        v.at<double>(2), 0, -v.at<double>(0),
        -v.at<double>(1), v.at<double>(0), 0);
}

/**
 * Compute Essential matrix from R and t
 */
cv::Mat computeEssentialFromRT(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat t_x = skewSymmetric(t);
    return t_x * R;
}

/**
 * Verify epipolar constraint: x'^T * F * x = 0
 */
double verifyEpipolarConstraint(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& F) {

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
    std::cout << "=== Essential and Fundamental Matrix Estimation ===\n" << std::endl;

    // Camera intrinsic matrix
    double fx = 500, fy = 500;  // Focal lengths
    double cx = 320, cy = 240;  // Principal point
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);

    std::cout << "Camera Intrinsic Matrix K:" << std::endl;
    std::cout << K << std::endl << std::endl;

    // Camera 2 pose relative to Camera 1
    // Rotation (10 degrees around Y axis)
    double angle = 10.0 * CV_PI / 180.0;
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        std::cos(angle), 0, std::sin(angle),
        0, 1, 0,
        -std::sin(angle), 0, std::cos(angle));

    // Translation
    cv::Mat t = (cv::Mat_<double>(3, 1) << 0.5, 0.0, 0.0);

    std::cout << "Ground Truth Camera 2 Rotation:\n" << R << std::endl;
    std::cout << "Ground Truth Camera 2 Translation:\n" << t.t() << std::endl;
    std::cout << std::endl;

    // Compute ground truth Essential and Fundamental matrices
    cv::Mat E_gt = computeEssentialFromRT(R, t);
    cv::Mat K_inv = K.inv();
    cv::Mat F_gt = K_inv.t() * E_gt * K_inv;

    std::cout << "Ground Truth Essential Matrix:\n" << E_gt << std::endl;
    std::cout << "Ground Truth Fundamental Matrix:\n" << F_gt << std::endl;
    std::cout << std::endl;

    // Generate synthetic correspondences
    std::vector<cv::Point2f> pts1, pts2;
    generateSyntheticData(K, R, t, pts1, pts2, 100);

    std::cout << "Generated " << pts1.size() << " point correspondences"
              << std::endl << std::endl;

    // Add some noise
    std::vector<cv::Point2f> pts1_noisy = pts1;
    std::vector<cv::Point2f> pts2_noisy = pts2;
    addNoise(pts1_noisy, 1.0);  // 1 pixel noise
    addNoise(pts2_noisy, 1.0);

    // =========================================================
    // Method 1: 8-Point Algorithm for Fundamental Matrix
    // =========================================================
    std::cout << "=== 8-Point Algorithm (Fundamental Matrix) ===" << std::endl;

    cv::Mat F_8point = cv::findFundamentalMat(
        pts1_noisy, pts2_noisy,
        cv::FM_8POINT  // 8-point algorithm
    );

    double F_8point_error = verifyEpipolarConstraint(pts1_noisy, pts2_noisy, F_8point);
    std::cout << "Estimated F (8-point):\n" << F_8point << std::endl;
    std::cout << "Average epipolar error: " << F_8point_error << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Method 2: RANSAC for Fundamental Matrix (robust to outliers)
    // =========================================================
    std::cout << "=== RANSAC (Fundamental Matrix) ===" << std::endl;

    // Add some outliers
    std::vector<cv::Point2f> pts1_outliers = pts1_noisy;
    std::vector<cv::Point2f> pts2_outliers = pts2_noisy;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> outlier_dist(0, 640);

    // Add 20% outliers
    int num_outliers = pts1_outliers.size() * 0.2;
    for (int i = 0; i < num_outliers; ++i) {
        pts1_outliers.push_back(cv::Point2f(outlier_dist(gen), outlier_dist(gen)));
        pts2_outliers.push_back(cv::Point2f(outlier_dist(gen), outlier_dist(gen)));
    }

    cv::Mat inlier_mask;
    cv::Mat F_ransac = cv::findFundamentalMat(
        pts1_outliers, pts2_outliers,
        cv::FM_RANSAC,  // RANSAC
        3.0,            // RANSAC reprojection threshold
        0.99,           // Confidence
        inlier_mask
    );

    int num_inliers = cv::countNonZero(inlier_mask);
    std::cout << "Inliers: " << num_inliers << "/" << pts1_outliers.size() << std::endl;

    double F_ransac_error = verifyEpipolarConstraint(pts1_noisy, pts2_noisy, F_ransac);
    std::cout << "Estimated F (RANSAC):\n" << F_ransac << std::endl;
    std::cout << "Average epipolar error: " << F_ransac_error << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Method 3: 5-Point Algorithm for Essential Matrix
    // =========================================================
    std::cout << "=== 5-Point Algorithm (Essential Matrix) ===" << std::endl;

    cv::Mat E_5point = cv::findEssentialMat(
        pts1_noisy, pts2_noisy,
        K,
        cv::RANSAC,     // Method
        0.999,          // Confidence
        1.0,            // RANSAC threshold
        inlier_mask
    );

    num_inliers = cv::countNonZero(inlier_mask);
    std::cout << "Inliers: " << num_inliers << "/" << pts1_noisy.size() << std::endl;

    std::cout << "Estimated E (5-point):\n" << E_5point << std::endl;
    std::cout << std::endl;

    // Verify Essential matrix properties
    // E should be rank 2 and its two non-zero singular values should be equal
    cv::Mat U, S, Vt;
    cv::SVD::compute(E_5point, S, U, Vt);
    std::cout << "Essential Matrix Singular Values:" << std::endl;
    std::cout << "  s1 = " << S.at<double>(0) << std::endl;
    std::cout << "  s2 = " << S.at<double>(1) << std::endl;
    std::cout << "  s3 = " << S.at<double>(2) << " (should be ~0)" << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Recover Pose from Essential Matrix
    // =========================================================
    std::cout << "=== Pose Recovery from Essential Matrix ===" << std::endl;

    cv::Mat R_recovered, t_recovered;
    int num_good = cv::recoverPose(
        E_5point, pts1_noisy, pts2_noisy, K,
        R_recovered, t_recovered, inlier_mask
    );

    std::cout << "Points in front of both cameras: " << num_good << std::endl;
    std::cout << "\nRecovered Rotation:\n" << R_recovered << std::endl;
    std::cout << "\nRecovered Translation (up to scale):\n" << t_recovered.t() << std::endl;
    std::cout << std::endl;

    // Compare with ground truth
    std::cout << "=== Comparison with Ground Truth ===" << std::endl;

    // Rotation error (Frobenius norm)
    cv::Mat R_diff = R_recovered - R;
    double rot_error = cv::norm(R_diff, cv::NORM_FRO);
    std::cout << "Rotation error (Frobenius): " << rot_error << std::endl;

    // Translation direction error (since scale is unknown)
    cv::Mat t_normalized = t / cv::norm(t);
    double t_dot = t_normalized.dot(t_recovered);
    double trans_angle = std::acos(std::abs(t_dot)) * 180.0 / CV_PI;
    std::cout << "Translation direction error: " << trans_angle << " degrees" << std::endl;
    std::cout << std::endl;

    // =========================================================
    // Relationship between F and E
    // =========================================================
    std::cout << "=== Relationship: E = K'^T * F * K ===" << std::endl;

    cv::Mat E_from_F = K.t() * F_ransac * K;

    // Normalize for comparison (Essential matrix is defined up to scale)
    E_from_F = E_from_F / cv::norm(E_from_F, cv::NORM_FRO);
    cv::Mat E_5point_norm = E_5point / cv::norm(E_5point, cv::NORM_FRO);

    std::cout << "E from F (normalized):\n" << E_from_F << std::endl;
    std::cout << "E from 5-point (normalized):\n" << E_5point_norm << std::endl;

    cv::Mat E_diff = E_from_F - E_5point_norm;
    // Handle sign ambiguity
    double diff1 = cv::norm(E_diff, cv::NORM_FRO);
    double diff2 = cv::norm(E_from_F + E_5point_norm, cv::NORM_FRO);
    double min_diff = std::min(diff1, diff2);

    std::cout << "Difference (Frobenius): " << min_diff << std::endl;
    std::cout << std::endl;

    std::cout << "=== Demo Complete ===" << std::endl;

    return 0;
}
