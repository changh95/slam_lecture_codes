/**
 * @file pnp_poselib.cpp
 * @brief P3P solver demonstration using PoseLib
 *
 * This example demonstrates the P3P (Perspective-3-Point) solver from PoseLib.
 * P3P is a minimal solver that estimates camera pose from exactly 3 point
 * correspondences. It returns up to 4 possible solutions.
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
 * @brief Generate synthetic 3D points
 */
std::vector<Eigen::Vector3d> generatePoints3D(int numPoints) {
    std::vector<Eigen::Vector3d> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int i = 0; i < numPoints; i++) {
        // Points in front of camera (Z > 0)
        points.emplace_back(dist(rng), dist(rng), 2.0 + dist(rng));
    }
    return points;
}

/**
 * @brief Convert 2D image point to bearing vector (normalized ray)
 *
 * bearing = K^{-1} * [u, v, 1]^T, then normalized
 */
Eigen::Vector3d pixelToBearing(const Eigen::Vector2d& pixel, const Eigen::Matrix3d& K) {
    Eigen::Vector3d bearing;
    bearing << (pixel.x() - K(0, 2)) / K(0, 0),
               (pixel.y() - K(1, 2)) / K(1, 1),
               1.0;
    return bearing.normalized();
}

/**
 * @brief Project 3D point to 2D pixel
 */
Eigen::Vector2d projectPoint(const Eigen::Vector3d& point3d,
                              const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& t,
                              const Eigen::Matrix3d& K) {
    Eigen::Vector3d p_cam = R * point3d + t;
    Eigen::Vector3d p_proj = K * p_cam;
    return Eigen::Vector2d(p_proj.x() / p_proj.z(), p_proj.y() / p_proj.z());
}

/**
 * @brief Add Gaussian noise to 2D points
 */
void addNoise(std::vector<Eigen::Vector2d>& points, double sigma) {
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, sigma);
    for (auto& p : points) {
        p.x() += noise(rng);
        p.y() += noise(rng);
    }
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
 * @brief Compute translation error (Euclidean distance)
 */
double translationError(const Eigen::Vector3d& t_est, const Eigen::Vector3d& t_gt) {
    return (t_est - t_gt).norm();
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "PnP Demo using PoseLib (P3P Solver)\n";
    std::cout << "===========================================\n\n";

    // Camera intrinsics (KITTI-like)
    Eigen::Matrix3d K;
    K << 718.856, 0.0, 607.193,
         0.0, 718.856, 185.216,
         0.0, 0.0, 1.0;

    std::cout << "Camera intrinsics K:\n" << K << "\n\n";

    // Ground truth pose (camera looking at scene)
    Eigen::Matrix3d R_gt;
    R_gt = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ());

    Eigen::Vector3d t_gt(0.5, -0.3, 0.2);

    std::cout << "Ground truth rotation R:\n" << R_gt << "\n";
    std::cout << "Ground truth translation t: " << t_gt.transpose() << "\n\n";

    // Generate 3D points and project to 2D
    const int numPoints = 10;
    auto points3d = generatePoints3D(numPoints);

    std::vector<Eigen::Vector2d> points2d;
    for (const auto& p3d : points3d) {
        points2d.push_back(projectPoint(p3d, R_gt, t_gt, K));
    }

    // Add noise
    const double noise_sigma = 0.5;  // pixels
    addNoise(points2d, noise_sigma);
    std::cout << "Added Gaussian noise with sigma = " << noise_sigma << " pixels\n\n";

    // Convert to bearing vectors for PoseLib
    std::vector<Eigen::Vector3d> bearings;
    for (const auto& p2d : points2d) {
        bearings.push_back(pixelToBearing(p2d, K));
    }

    // =========================================================================
    // PoseLib P3P Solver
    // =========================================================================
    std::cout << "--- PoseLib P3P Solver ---\n";
    std::cout << "Using first 3 points (minimal solver)\n\n";

    // P3P uses exactly 3 points
    std::vector<Eigen::Vector3d> x_p3p(bearings.begin(), bearings.begin() + 3);
    std::vector<Eigen::Vector3d> X_p3p(points3d.begin(), points3d.begin() + 3);

    // Call PoseLib P3P
    std::vector<poselib::CameraPose> solutions;
    int num_solutions = poselib::p3p(x_p3p, X_p3p, &solutions);

    std::cout << "Number of solutions: " << num_solutions << "\n\n";

    // Evaluate each solution
    double best_rot_error = std::numeric_limits<double>::max();
    int best_idx = -1;

    for (int i = 0; i < num_solutions; i++) {
        const auto& pose = solutions[i];

        // Convert quaternion to rotation matrix
        Eigen::Quaterniond q(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
        Eigen::Matrix3d R_est = q.toRotationMatrix();
        Eigen::Vector3d t_est = pose.t;

        double rot_err = rotationError(R_est, R_gt);
        double trans_err = translationError(t_est, t_gt);

        std::cout << "Solution " << i + 1 << ":\n";
        std::cout << "  Rotation error:    " << std::fixed << std::setprecision(4)
                  << rot_err << " deg\n";
        std::cout << "  Translation error: " << trans_err << " m\n";

        if (rot_err < best_rot_error) {
            best_rot_error = rot_err;
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        std::cout << "\nBest solution: #" << best_idx + 1
                  << " (rotation error = " << best_rot_error << " deg)\n";

        const auto& best_pose = solutions[best_idx];
        Eigen::Quaterniond q(best_pose.q[0], best_pose.q[1], best_pose.q[2], best_pose.q[3]);
        std::cout << "\nEstimated rotation:\n" << q.toRotationMatrix() << "\n";
        std::cout << "Estimated translation: " << best_pose.t.transpose() << "\n";
    }

    // =========================================================================
    // Compare with OpenCV P3P
    // =========================================================================
    std::cout << "\n--- OpenCV P3P (for comparison) ---\n";

    // Convert to OpenCV format
    std::vector<cv::Point3f> cvPoints3d;
    std::vector<cv::Point2f> cvPoints2d;
    for (int i = 0; i < 3; i++) {
        cvPoints3d.emplace_back(points3d[i].x(), points3d[i].y(), points3d[i].z());
        cvPoints2d.emplace_back(points2d[i].x(), points2d[i].y());
    }

    cv::Mat cvK = (cv::Mat_<double>(3, 3) <<
        K(0, 0), K(0, 1), K(0, 2),
        K(1, 0), K(1, 1), K(1, 2),
        K(2, 0), K(2, 1), K(2, 2));

    cv::Mat rvec, tvec;
    // Need 4 points for P3P in OpenCV (uses one extra for disambiguation)
    cvPoints3d.emplace_back(points3d[3].x(), points3d[3].y(), points3d[3].z());
    cvPoints2d.emplace_back(points2d[3].x(), points2d[3].y());

    bool success = cv::solvePnP(cvPoints3d, cvPoints2d, cvK, cv::Mat(),
                                 rvec, tvec, false, cv::SOLVEPNP_P3P);

    if (success) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix3d R_opencv;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R_opencv(i, j) = R_cv.at<double>(i, j);

        Eigen::Vector3d t_opencv(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        double rot_err = rotationError(R_opencv, R_gt);
        double trans_err = translationError(t_opencv, t_gt);

        std::cout << "OpenCV P3P result:\n";
        std::cout << "  Rotation error:    " << rot_err << " deg\n";
        std::cout << "  Translation error: " << trans_err << " m\n";
    }

    std::cout << "\n===========================================\n";
    std::cout << "Notes:\n";
    std::cout << "- P3P returns up to 4 solutions (need extra point to disambiguate)\n";
    std::cout << "- PoseLib uses bearing vectors (normalized rays)\n";
    std::cout << "- OpenCV uses pixel coordinates with camera matrix\n";
    std::cout << "===========================================\n";

    return 0;
}
