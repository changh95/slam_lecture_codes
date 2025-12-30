/**
 * @file homography_poselib.cpp
 * @brief Homography solver demonstration using PoseLib
 *
 * This example demonstrates homography estimation from PoseLib:
 * - homography_4pt: Minimal 4-point homography solver
 *
 * The homography relates points on a plane between two views:
 *   x2 ~ H * x1
 *
 * PoseLib: https://github.com/PoseLib/PoseLib
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <PoseLib/PoseLib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

/**
 * @brief Generate synthetic planar correspondences
 *
 * Creates points on a plane and projects them to two camera views
 */
void generatePlanarCorrespondences(
    const Eigen::Matrix3d& H,
    std::vector<Eigen::Vector3d>& pts1,
    std::vector<Eigen::Vector3d>& pts2,
    int num_points,
    double noise_sigma = 0.0) {

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> y_dist(-1.0, 1.0);
    std::normal_distribution<double> noise(0.0, noise_sigma);

    pts1.clear();
    pts2.clear();

    for (int i = 0; i < num_points; i++) {
        // Point in normalized coordinates (homogeneous)
        Eigen::Vector3d p1(x_dist(rng), y_dist(rng), 1.0);

        // Transform using homography
        Eigen::Vector3d p2 = H * p1;
        p2 /= p2.z();  // Normalize

        // Add noise
        if (noise_sigma > 0) {
            p1.x() += noise(rng);
            p1.y() += noise(rng);
            p2.x() += noise(rng);
            p2.y() += noise(rng);
        }

        pts1.push_back(p1);
        pts2.push_back(p2);
    }
}

/**
 * @brief Compute homography reprojection error
 */
double computeHomographyError(const Eigen::Matrix3d& H,
                               const std::vector<Eigen::Vector3d>& pts1,
                               const std::vector<Eigen::Vector3d>& pts2) {
    double total_error = 0.0;
    for (size_t i = 0; i < pts1.size(); i++) {
        Eigen::Vector3d p2_proj = H * pts1[i];
        p2_proj /= p2_proj.z();

        double dx = p2_proj.x() - pts2[i].x();
        double dy = p2_proj.y() - pts2[i].y();
        total_error += std::sqrt(dx * dx + dy * dy);
    }
    return total_error / pts1.size();
}

/**
 * @brief Compute rotation error in degrees
 */
double rotationError(const Eigen::Matrix3d& R_est, const Eigen::Matrix3d& R_gt) {
    Eigen::Matrix3d dR = R_est.transpose() * R_gt;
    double trace = dR.trace();
    double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
    return angle * 180.0 / M_PI;
}

/**
 * @brief Compute translation direction error in degrees
 */
double translationDirectionError(const Eigen::Vector3d& t_est, const Eigen::Vector3d& t_gt) {
    if (t_est.norm() < 1e-10 || t_gt.norm() < 1e-10) return 0.0;
    Eigen::Vector3d t1 = t_est.normalized();
    Eigen::Vector3d t2 = t_gt.normalized();
    double dot = std::clamp(std::abs(t1.dot(t2)), 0.0, 1.0);
    return std::acos(dot) * 180.0 / M_PI;
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "Homography Demo using PoseLib (4-Point)\n";
    std::cout << "===========================================\n\n";

    // Ground truth relative pose (camera 2 relative to camera 1)
    Eigen::Matrix3d R_gt;
    R_gt = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX());

    Eigen::Vector3d t_gt(0.3, 0.1, 0.0);

    // Plane parameters: n'X = d (plane normal and distance)
    Eigen::Vector3d n_gt(0, 0, 1);  // Plane perpendicular to Z axis
    double d_gt = 2.0;              // Plane at Z = 2

    // Ground truth homography: H = R - (t * n^T) / d
    Eigen::Matrix3d H_gt = R_gt - (t_gt * n_gt.transpose()) / d_gt;
    H_gt /= H_gt(2, 2);  // Normalize

    std::cout << "Ground truth rotation R:\n" << R_gt << "\n\n";
    std::cout << "Ground truth translation t: " << t_gt.transpose() << "\n";
    std::cout << "Plane normal n: " << n_gt.transpose() << "\n";
    std::cout << "Plane distance d: " << d_gt << "\n\n";
    std::cout << "Ground truth homography H:\n" << H_gt << "\n\n";

    // Generate correspondences
    const int numPoints = 10;
    std::vector<Eigen::Vector3d> pts1, pts2;
    const double noise_sigma = 0.001;

    generatePlanarCorrespondences(H_gt, pts1, pts2, numPoints, noise_sigma);
    std::cout << "Generated " << numPoints << " correspondences with noise sigma = "
              << noise_sigma << "\n\n";

    // =========================================================================
    // PoseLib 4-Point Homography Solver
    // =========================================================================
    std::cout << "--- PoseLib 4-Point Homography Solver ---\n";
    std::cout << "Using first 4 points (minimal solver)\n\n";

    // Extract first 4 points
    std::vector<Eigen::Vector3d> x1(pts1.begin(), pts1.begin() + 4);
    std::vector<Eigen::Vector3d> x2(pts2.begin(), pts2.begin() + 4);

    // Call PoseLib homography_4pt
    Eigen::Matrix3d H_poselib;
    int num_solutions = poselib::homography_4pt(x1, x2, &H_poselib);

    if (num_solutions > 0) {
        // Normalize homography
        H_poselib /= H_poselib(2, 2);

        double reproj_error = computeHomographyError(H_poselib, pts1, pts2);

        std::cout << "Estimated homography H:\n" << H_poselib << "\n\n";
        std::cout << "Reprojection error (all points): " << std::fixed
                  << std::setprecision(6) << reproj_error << "\n\n";

        // Compare with ground truth
        Eigen::Matrix3d H_diff = H_poselib - H_gt;
        std::cout << "Frobenius norm of (H_est - H_gt): " << H_diff.norm() << "\n\n";
    } else {
        std::cout << "No solution found!\n\n";
    }

    // =========================================================================
    // Compare with OpenCV findHomography
    // =========================================================================
    std::cout << "--- OpenCV findHomography (for comparison) ---\n";

    std::vector<cv::Point2f> cvPts1, cvPts2;
    for (size_t i = 0; i < pts1.size(); i++) {
        cvPts1.emplace_back(pts1[i].x(), pts1[i].y());
        cvPts2.emplace_back(pts2[i].x(), pts2[i].y());
    }

    cv::Mat H_cv = cv::findHomography(cvPts1, cvPts2, 0);  // No RANSAC

    Eigen::Matrix3d H_opencv;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            H_opencv(i, j) = H_cv.at<double>(i, j);

    H_opencv /= H_opencv(2, 2);

    double reproj_opencv = computeHomographyError(H_opencv, pts1, pts2);
    std::cout << "OpenCV reprojection error: " << reproj_opencv << "\n\n";

    // =========================================================================
    // Homography Decomposition
    // =========================================================================
    std::cout << "--- Homography Decomposition ---\n";
    std::cout << "(Using OpenCV decomposeHomographyMat)\n\n";

    // Use identity for K (we're in normalized coordinates)
    cv::Mat K_cv = cv::Mat::eye(3, 3, CV_64F);

    std::vector<cv::Mat> Rs, ts, normals;
    cv::decomposeHomographyMat(H_cv, K_cv, Rs, ts, normals);

    std::cout << "Number of decomposition solutions: " << Rs.size() << "\n\n";

    double best_rot_error = std::numeric_limits<double>::max();
    int best_idx = -1;

    for (size_t i = 0; i < Rs.size(); i++) {
        Eigen::Matrix3d R_est;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                R_est(r, c) = Rs[i].at<double>(r, c);

        Eigen::Vector3d t_est(ts[i].at<double>(0), ts[i].at<double>(1), ts[i].at<double>(2));
        Eigen::Vector3d n_est(normals[i].at<double>(0), normals[i].at<double>(1), normals[i].at<double>(2));

        double rot_err = rotationError(R_est, R_gt);
        double trans_err = translationDirectionError(t_est, t_gt);

        std::cout << "Solution " << i + 1 << ":\n";
        std::cout << "  Rotation error:    " << std::fixed << std::setprecision(4)
                  << rot_err << " deg\n";
        std::cout << "  Translation dir error: " << trans_err << " deg\n";
        std::cout << "  Normal: " << n_est.transpose() << "\n";

        if (rot_err < best_rot_error) {
            best_rot_error = rot_err;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx >= 0) {
        std::cout << "\nBest solution: #" << best_idx + 1 << "\n";
    }

    std::cout << "\n===========================================\n";
    std::cout << "Notes:\n";
    std::cout << "- Homography requires coplanar points\n";
    std::cout << "- 4-point is the minimal solver\n";
    std::cout << "- Decomposition gives up to 4 (R, t, n) solutions\n";
    std::cout << "- Additional points needed to disambiguate\n";
    std::cout << "===========================================\n";

    return 0;
}
