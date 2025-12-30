/**
 * @file relpose_poselib.cpp
 * @brief 5-Point relative pose solver demonstration using PoseLib
 *
 * This example demonstrates the 5-point essential matrix solver from PoseLib.
 * The solver estimates the relative pose (R, t) between two cameras from
 * 5 point correspondences (minimal solver).
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
    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> y_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> z_dist(5.0, 15.0);

    for (int i = 0; i < numPoints; i++) {
        points.emplace_back(x_dist(rng), y_dist(rng), z_dist(rng));
    }
    return points;
}

/**
 * @brief Project 3D point to bearing vector
 */
Eigen::Vector3d projectToBearing(const Eigen::Vector3d& point3d,
                                  const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& t) {
    Eigen::Vector3d p_cam = R * point3d + t;
    return p_cam.normalized();
}

/**
 * @brief Add noise to bearing vectors
 */
void addNoise(std::vector<Eigen::Vector3d>& bearings, double sigma_rad) {
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, sigma_rad);

    for (auto& b : bearings) {
        Eigen::Vector3d perturb(noise(rng), noise(rng), noise(rng));
        perturb = perturb - b.dot(perturb) * b;
        b = (b + perturb).normalized();
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
 * @brief Compute translation direction error in degrees
 * (translation is only up to scale for essential matrix)
 */
double translationDirectionError(const Eigen::Vector3d& t_est, const Eigen::Vector3d& t_gt) {
    Eigen::Vector3d t1 = t_est.normalized();
    Eigen::Vector3d t2 = t_gt.normalized();
    double dot = std::clamp(std::abs(t1.dot(t2)), 0.0, 1.0);
    return std::acos(dot) * 180.0 / M_PI;
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "Relative Pose Demo using PoseLib (5-Point)\n";
    std::cout << "===========================================\n\n";

    // Ground truth relative pose (camera 2 relative to camera 1)
    Eigen::Matrix3d R_gt;
    R_gt = Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY());  // 15 deg rotation around Y

    Eigen::Vector3d t_gt(0.5, 0.0, 0.0);  // Translation along X

    std::cout << "Ground truth rotation R:\n" << R_gt << "\n";
    std::cout << "Ground truth translation t: " << t_gt.transpose() << "\n\n";

    // Generate 3D points
    const int numPoints = 20;
    auto points3d = generatePoints3D(numPoints);

    // Project to bearing vectors in both cameras
    // Camera 1 is at origin (identity pose)
    std::vector<Eigen::Vector3d> bearings1;
    std::vector<Eigen::Vector3d> bearings2;

    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1 = Eigen::Vector3d::Zero();

    for (const auto& p3d : points3d) {
        bearings1.push_back(projectToBearing(p3d, R1, t1));
        bearings2.push_back(projectToBearing(p3d, R_gt, t_gt));
    }

    // Add small angular noise
    const double noise_sigma = 0.005;  // radians (~0.3 degrees)
    addNoise(bearings1, noise_sigma);
    addNoise(bearings2, noise_sigma);
    std::cout << "Added angular noise with sigma = " << noise_sigma * 180.0 / M_PI << " deg\n\n";

    // =========================================================================
    // PoseLib 5-Point Essential Matrix Solver
    // =========================================================================
    std::cout << "--- PoseLib 5-Point Solver ---\n";
    std::cout << "Using first 5 points (minimal solver)\n\n";

    // Use first 5 points
    std::vector<Eigen::Vector3d> x1(bearings1.begin(), bearings1.begin() + 5);
    std::vector<Eigen::Vector3d> x2(bearings2.begin(), bearings2.begin() + 5);

    // Call PoseLib relpose_5pt
    std::vector<poselib::CameraPose> solutions;
    int num_solutions = poselib::relpose_5pt(x1, x2, &solutions);

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
        double trans_err = translationDirectionError(t_est, t_gt);

        std::cout << "Solution " << i + 1 << ":\n";
        std::cout << "  Rotation error:    " << std::fixed << std::setprecision(4)
                  << rot_err << " deg\n";
        std::cout << "  Translation dir error: " << trans_err << " deg\n";

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
        std::cout << "Estimated translation (up to scale): " << best_pose.t.transpose() << "\n";
    }

    // =========================================================================
    // Compare with OpenCV 5-Point
    // =========================================================================
    std::cout << "\n--- OpenCV 5-Point (for comparison) ---\n";

    // Use camera matrix to convert bearings to pixels
    Eigen::Matrix3d K;
    K << 500.0, 0.0, 320.0,
         0.0, 500.0, 240.0,
         0.0, 0.0, 1.0;

    std::vector<cv::Point2f> cvPts1, cvPts2;
    for (size_t i = 0; i < bearings1.size(); i++) {
        // Convert bearing to pixel
        Eigen::Vector3d px1 = K * bearings1[i] / bearings1[i].z();
        Eigen::Vector3d px2 = K * bearings2[i] / bearings2[i].z();
        cvPts1.emplace_back(px1.x(), px1.y());
        cvPts2.emplace_back(px2.x(), px2.y());
    }

    cv::Mat cvK = (cv::Mat_<double>(3, 3) <<
        K(0, 0), K(0, 1), K(0, 2),
        K(1, 0), K(1, 1), K(1, 2),
        K(2, 0), K(2, 1), K(2, 2));

    cv::Mat E_cv = cv::findEssentialMat(cvPts1, cvPts2, cvK, cv::RANSAC, 0.999, 1.0);

    cv::Mat R_cv, t_cv;
    cv::recoverPose(E_cv, cvPts1, cvPts2, cvK, R_cv, t_cv);

    Eigen::Matrix3d R_opencv;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_opencv(i, j) = R_cv.at<double>(i, j);

    Eigen::Vector3d t_opencv(t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2));

    std::cout << "Rotation error:    " << rotationError(R_opencv, R_gt) << " deg\n";
    std::cout << "Translation dir error: " << translationDirectionError(t_opencv, t_gt) << " deg\n";

    std::cout << "\n===========================================\n";
    std::cout << "Notes:\n";
    std::cout << "- 5-point solver returns up to 10 solutions\n";
    std::cout << "- PoseLib uses bearing vectors (normalized rays)\n";
    std::cout << "- Translation is only recovered up to scale\n";
    std::cout << "- OpenCV uses RANSAC internally for robustness\n";
    std::cout << "===========================================\n";

    return 0;
}
