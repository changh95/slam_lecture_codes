/**
 * @file pnp_opengv.cpp
 * @brief P3P solver demonstration using OpenGV
 *
 * This example demonstrates P3P (Perspective-3-Point) solvers from OpenGV.
 * OpenGV provides multiple P3P implementations:
 * - p3p_kneip: Kneip's P3P solver
 * - p3p_gao: Gao's P3P solver
 * - epnp: EPnP for n>=4 points (non-minimal)
 *
 * OpenGV: https://laurentkneip.github.io/opengv/
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

/**
 * @brief Generate synthetic 3D points
 */
opengv::points_t generatePoints3D(int numPoints) {
    opengv::points_t points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int i = 0; i < numPoints; i++) {
        // Points in front of camera (Z > 0)
        opengv::point_t pt;
        pt << dist(rng), dist(rng), 2.0 + dist(rng);
        points.push_back(pt);
    }
    return points;
}

/**
 * @brief Project 3D point to bearing vector
 *
 * Projects point to camera frame, then normalizes to unit vector
 */
opengv::bearingVector_t projectToBearing(const opengv::point_t& point3d,
                                          const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& t) {
    Eigen::Vector3d p_cam = R * point3d + t;
    return p_cam.normalized();
}

/**
 * @brief Add noise to bearing vectors
 */
void addNoise(opengv::bearingVectors_t& bearings, double sigma_rad) {
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, sigma_rad);

    for (auto& b : bearings) {
        // Add angular noise by perturbing in tangent plane
        Eigen::Vector3d perturb;
        perturb << noise(rng), noise(rng), noise(rng);

        // Project perturbation to tangent plane of bearing
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
 * @brief Compute translation error (Euclidean distance)
 */
double translationError(const Eigen::Vector3d& t_est, const Eigen::Vector3d& t_gt) {
    return (t_est - t_gt).norm();
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "PnP Demo using OpenGV (P3P Solvers)\n";
    std::cout << "===========================================\n\n";

    // Ground truth pose (camera looking at scene)
    Eigen::Matrix3d R_gt;
    R_gt = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ());

    Eigen::Vector3d t_gt(0.5, -0.3, 0.2);

    std::cout << "Ground truth rotation R:\n" << R_gt << "\n";
    std::cout << "Ground truth translation t: " << t_gt.transpose() << "\n\n";

    // Generate 3D points
    const int numPoints = 10;
    auto points3d = generatePoints3D(numPoints);

    // Project to bearing vectors
    opengv::bearingVectors_t bearings;
    for (const auto& p3d : points3d) {
        bearings.push_back(projectToBearing(p3d, R_gt, t_gt));
    }

    // Add small angular noise (~0.5 degrees)
    const double noise_sigma = 0.01;  // radians
    addNoise(bearings, noise_sigma);
    std::cout << "Added angular noise with sigma = " << noise_sigma * 180.0 / M_PI << " deg\n\n";

    // Create adapter for OpenGV
    // Note: OpenGV uses world-to-camera convention
    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearings, points3d);

    // =========================================================================
    // OpenGV P3P Kneip Solver
    // =========================================================================
    std::cout << "--- OpenGV P3P Kneip Solver ---\n";
    std::cout << "Using first 3 points (minimal solver)\n\n";

    // P3P uses indices 0, 1, 2
    opengv::transformations_t solutions_kneip = opengv::absolute_pose::p3p_kneip(adapter);

    std::cout << "Number of solutions: " << solutions_kneip.size() << "\n\n";

    double best_rot_error = std::numeric_limits<double>::max();
    int best_idx = -1;

    for (size_t i = 0; i < solutions_kneip.size(); i++) {
        const auto& T = solutions_kneip[i];
        Eigen::Matrix3d R_est = T.block<3, 3>(0, 0);
        Eigen::Vector3d t_est = T.block<3, 1>(0, 3);

        double rot_err = rotationError(R_est, R_gt);
        double trans_err = translationError(t_est, t_gt);

        std::cout << "Solution " << i + 1 << ":\n";
        std::cout << "  Rotation error:    " << std::fixed << std::setprecision(4)
                  << rot_err << " deg\n";
        std::cout << "  Translation error: " << trans_err << " m\n";

        if (rot_err < best_rot_error) {
            best_rot_error = rot_err;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx >= 0) {
        std::cout << "\nBest solution: #" << best_idx + 1
                  << " (rotation error = " << best_rot_error << " deg)\n";

        const auto& best_T = solutions_kneip[best_idx];
        std::cout << "\nEstimated rotation:\n" << best_T.block<3, 3>(0, 0) << "\n";
        std::cout << "Estimated translation: " << best_T.block<3, 1>(0, 3).transpose() << "\n";
    }

    // =========================================================================
    // OpenGV P3P Gao Solver
    // =========================================================================
    std::cout << "\n--- OpenGV P3P Gao Solver ---\n";

    opengv::transformations_t solutions_gao = opengv::absolute_pose::p3p_gao(adapter);

    std::cout << "Number of solutions: " << solutions_gao.size() << "\n";

    best_rot_error = std::numeric_limits<double>::max();
    best_idx = -1;

    for (size_t i = 0; i < solutions_gao.size(); i++) {
        const auto& T = solutions_gao[i];
        Eigen::Matrix3d R_est = T.block<3, 3>(0, 0);
        Eigen::Vector3d t_est = T.block<3, 1>(0, 3);

        double rot_err = rotationError(R_est, R_gt);

        if (rot_err < best_rot_error) {
            best_rot_error = rot_err;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx >= 0) {
        const auto& best_T = solutions_gao[best_idx];
        std::cout << "Best solution: #" << best_idx + 1
                  << " (rotation error = " << best_rot_error << " deg)\n";
        std::cout << "Translation error: "
                  << translationError(best_T.block<3, 1>(0, 3), t_gt) << " m\n";
    }

    // =========================================================================
    // OpenGV EPnP (non-minimal, uses all points)
    // =========================================================================
    std::cout << "\n--- OpenGV EPnP Solver ---\n";
    std::cout << "Using all " << numPoints << " points\n";

    opengv::transformation_t T_epnp = opengv::absolute_pose::epnp(adapter);

    Eigen::Matrix3d R_epnp = T_epnp.block<3, 3>(0, 0);
    Eigen::Vector3d t_epnp = T_epnp.block<3, 1>(0, 3);

    std::cout << "Rotation error:    " << rotationError(R_epnp, R_gt) << " deg\n";
    std::cout << "Translation error: " << translationError(t_epnp, t_gt) << " m\n";

    // =========================================================================
    // Compare with OpenCV P3P
    // =========================================================================
    std::cout << "\n--- OpenCV P3P (for comparison) ---\n";

    // Camera intrinsics (simple pinhole for comparison)
    Eigen::Matrix3d K;
    K << 718.856, 0.0, 607.193,
         0.0, 718.856, 185.216,
         0.0, 0.0, 1.0;

    // Convert bearings back to pixel coordinates for OpenCV
    std::vector<cv::Point3f> cvPoints3d;
    std::vector<cv::Point2f> cvPoints2d;
    for (int i = 0; i < 4; i++) {  // Need 4 points for OpenCV P3P
        cvPoints3d.emplace_back(points3d[i].x(), points3d[i].y(), points3d[i].z());

        // Project 3D point to camera, then to pixels
        Eigen::Vector3d p_cam = R_gt * points3d[i] + t_gt;
        Eigen::Vector3d p_px = K * p_cam;
        cvPoints2d.emplace_back(p_px.x() / p_px.z(), p_px.y() / p_px.z());
    }

    cv::Mat cvK = (cv::Mat_<double>(3, 3) <<
        K(0, 0), K(0, 1), K(0, 2),
        K(1, 0), K(1, 1), K(1, 2),
        K(2, 0), K(2, 1), K(2, 2));

    cv::Mat rvec, tvec;
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

        std::cout << "Rotation error:    " << rotationError(R_opencv, R_gt) << " deg\n";
        std::cout << "Translation error: " << translationError(t_opencv, t_gt) << " m\n";
    }

    std::cout << "\n===========================================\n";
    std::cout << "Notes:\n";
    std::cout << "- P3P returns up to 4 solutions (need extra point to disambiguate)\n";
    std::cout << "- OpenGV uses bearing vectors (unit vectors toward 3D points)\n";
    std::cout << "- EPnP uses all points for better accuracy\n";
    std::cout << "- p3p_kneip and p3p_gao use first 3 points by default\n";
    std::cout << "===========================================\n";

    return 0;
}
