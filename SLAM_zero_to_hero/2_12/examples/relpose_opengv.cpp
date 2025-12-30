/**
 * @file relpose_opengv.cpp
 * @brief 5-Point relative pose solver demonstration using OpenGV
 *
 * This example demonstrates relative pose solvers from OpenGV:
 * - fivept_nister: Nister's 5-point algorithm
 * - fivept_stewenius: Stewenius' 5-point algorithm
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

#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

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
void addNoise(opengv::bearingVectors_t& bearings, double sigma_rad) {
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
 */
double translationDirectionError(const Eigen::Vector3d& t_est, const Eigen::Vector3d& t_gt) {
    Eigen::Vector3d t1 = t_est.normalized();
    Eigen::Vector3d t2 = t_gt.normalized();
    double dot = std::clamp(std::abs(t1.dot(t2)), 0.0, 1.0);
    return std::acos(dot) * 180.0 / M_PI;
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "Relative Pose Demo using OpenGV (5-Point)\n";
    std::cout << "===========================================\n\n";

    // Ground truth relative pose (camera 2 relative to camera 1)
    Eigen::Matrix3d R_gt;
    R_gt = Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY());

    Eigen::Vector3d t_gt(0.5, 0.0, 0.0);

    std::cout << "Ground truth rotation R:\n" << R_gt << "\n";
    std::cout << "Ground truth translation t: " << t_gt.transpose() << "\n\n";

    // Generate 3D points
    const int numPoints = 20;
    auto points3d = generatePoints3D(numPoints);

    // Project to bearing vectors
    opengv::bearingVectors_t bearings1;
    opengv::bearingVectors_t bearings2;

    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1 = Eigen::Vector3d::Zero();

    for (const auto& p3d : points3d) {
        bearings1.push_back(projectToBearing(p3d, R1, t1));
        bearings2.push_back(projectToBearing(p3d, R_gt, t_gt));
    }

    // Add noise
    const double noise_sigma = 0.005;
    addNoise(bearings1, noise_sigma);
    addNoise(bearings2, noise_sigma);
    std::cout << "Added angular noise with sigma = " << noise_sigma * 180.0 / M_PI << " deg\n\n";

    // Create adapter
    opengv::relative_pose::CentralRelativeAdapter adapter(bearings1, bearings2);

    // =========================================================================
    // OpenGV 5-Point Nister
    // =========================================================================
    std::cout << "--- OpenGV 5-Point Nister Solver ---\n";

    opengv::essentials_t E_nister = opengv::relative_pose::fivept_nister(adapter);

    std::cout << "Number of essential matrices: " << E_nister.size() << "\n\n";

    double best_rot_error = std::numeric_limits<double>::max();
    int best_idx = -1;
    Eigen::Matrix3d best_R;
    Eigen::Vector3d best_t;

    for (size_t i = 0; i < E_nister.size(); i++) {
        // Decompose Essential matrix to R, t
        // E = t_x * R, SVD gives 2 possible R and 2 possible t (4 combinations)
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(E_nister[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        // W matrix for decomposition
        Eigen::Matrix3d W;
        W << 0, -1, 0,
             1, 0, 0,
             0, 0, 1;

        // Two possible rotations
        Eigen::Matrix3d R1_cand = U * W * V.transpose();
        Eigen::Matrix3d R2_cand = U * W.transpose() * V.transpose();

        // Ensure proper rotation (det = 1)
        if (R1_cand.determinant() < 0) R1_cand = -R1_cand;
        if (R2_cand.determinant() < 0) R2_cand = -R2_cand;

        // Translation (from U's last column)
        Eigen::Vector3d t_cand = U.col(2);

        // Test all 4 combinations
        std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> candidates = {
            {R1_cand, t_cand},
            {R1_cand, -t_cand},
            {R2_cand, t_cand},
            {R2_cand, -t_cand}
        };

        for (const auto& [R_cand, t_cand_signed] : candidates) {
            double rot_err = rotationError(R_cand, R_gt);
            if (rot_err < best_rot_error) {
                best_rot_error = rot_err;
                best_idx = static_cast<int>(i);
                best_R = R_cand;
                best_t = t_cand_signed;
            }
        }
    }

    if (best_idx >= 0) {
        double trans_err = translationDirectionError(best_t, t_gt);
        std::cout << "Best solution from Essential #" << best_idx + 1 << ":\n";
        std::cout << "  Rotation error:       " << std::fixed << std::setprecision(4)
                  << best_rot_error << " deg\n";
        std::cout << "  Translation dir error: " << trans_err << " deg\n";
        std::cout << "\nEstimated rotation:\n" << best_R << "\n";
        std::cout << "Estimated translation: " << best_t.transpose() << "\n";
    }

    // =========================================================================
    // OpenGV 5-Point Stewenius
    // =========================================================================
    std::cout << "\n--- OpenGV 5-Point Stewenius Solver ---\n";

    opengv::essentials_t E_stew = opengv::relative_pose::fivept_stewenius(adapter);

    std::cout << "Number of essential matrices: " << E_stew.size() << "\n";

    best_rot_error = std::numeric_limits<double>::max();
    for (size_t i = 0; i < E_stew.size(); i++) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(E_stew[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        Eigen::Matrix3d W;
        W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

        Eigen::Matrix3d R1_cand = U * W * V.transpose();
        Eigen::Matrix3d R2_cand = U * W.transpose() * V.transpose();

        if (R1_cand.determinant() < 0) R1_cand = -R1_cand;
        if (R2_cand.determinant() < 0) R2_cand = -R2_cand;

        Eigen::Vector3d t_cand = U.col(2);

        std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> candidates = {
            {R1_cand, t_cand}, {R1_cand, -t_cand}, {R2_cand, t_cand}, {R2_cand, -t_cand}
        };

        for (const auto& [R_c, t_c] : candidates) {
            double rot_err = rotationError(R_c, R_gt);
            if (rot_err < best_rot_error) {
                best_rot_error = rot_err;
                best_R = R_c;
                best_t = t_c;
            }
        }
    }

    std::cout << "Best rotation error: " << best_rot_error << " deg\n";
    std::cout << "Best translation dir error: " << translationDirectionError(best_t, t_gt) << " deg\n";

    // =========================================================================
    // Compare with OpenCV
    // =========================================================================
    std::cout << "\n--- OpenCV 5-Point (for comparison) ---\n";

    Eigen::Matrix3d K;
    K << 500.0, 0.0, 320.0,
         0.0, 500.0, 240.0,
         0.0, 0.0, 1.0;

    std::vector<cv::Point2f> cvPts1, cvPts2;
    for (size_t i = 0; i < bearings1.size(); i++) {
        Eigen::Vector3d px1 = K * bearings1[i] / bearings1[i].z();
        Eigen::Vector3d px2 = K * bearings2[i] / bearings2[i].z();
        cvPts1.emplace_back(px1.x(), px1.y());
        cvPts2.emplace_back(px2.x(), px2.y());
    }

    cv::Mat cvK = (cv::Mat_<double>(3, 3) <<
        K(0, 0), K(0, 1), K(0, 2),
        K(1, 0), K(1, 1), K(1, 2),
        K(2, 0), K(2, 1), K(2, 2));

    cv::Mat E_cv = cv::findEssentialMat(cvPts1, cvPts2, cvK, cv::RANSAC);

    cv::Mat R_cv, t_cv;
    cv::recoverPose(E_cv, cvPts1, cvPts2, cvK, R_cv, t_cv);

    Eigen::Matrix3d R_opencv;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_opencv(i, j) = R_cv.at<double>(i, j);

    Eigen::Vector3d t_opencv(t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2));

    std::cout << "Rotation error:       " << rotationError(R_opencv, R_gt) << " deg\n";
    std::cout << "Translation dir error: " << translationDirectionError(t_opencv, t_gt) << " deg\n";

    std::cout << "\n===========================================\n";
    std::cout << "Notes:\n";
    std::cout << "- OpenGV returns Essential matrices (need decomposition)\n";
    std::cout << "- Nister and Stewenius are both 5-point solvers\n";
    std::cout << "- Essential matrix decomposition gives 4 possible (R,t) pairs\n";
    std::cout << "- Cheirality check needed to select correct solution\n";
    std::cout << "===========================================\n";

    return 0;
}
