/**
 * @file hf_model_selection_poselib.cpp
 * @brief H/F Model Selection using PoseLib minimal solvers
 *
 * This example demonstrates H/F model selection using:
 * - PoseLib homography_4pt for homography estimation
 * - PoseLib relpose_5pt for essential matrix estimation
 *
 * PoseLib: https://github.com/PoseLib/PoseLib
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <PoseLib/PoseLib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

/**
 * @brief Generate synthetic 3D points on a plane
 */
std::vector<Eigen::Vector3d> generatePlanarPoints(int numPoints, double planeZ = 5.0) {
    std::vector<Eigen::Vector3d> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);

    for (int i = 0; i < numPoints; i++) {
        points.emplace_back(xy_dist(rng), xy_dist(rng), planeZ);
    }
    return points;
}

/**
 * @brief Generate synthetic 3D points in 3D space
 */
std::vector<Eigen::Vector3d> generate3DPoints(int numPoints) {
    std::vector<Eigen::Vector3d> points;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> xy_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> z_dist(3.0, 10.0);

    for (int i = 0; i < numPoints; i++) {
        points.emplace_back(xy_dist(rng), xy_dist(rng), z_dist(rng));
    }
    return points;
}

/**
 * @brief Project 3D point to bearing vector
 */
Eigen::Vector3d projectToBearing(
    const Eigen::Vector3d& point3d,
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
 * @brief Random sample indices
 */
std::vector<int> randomSample(int n, int k, std::mt19937& gen) {
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    indices.resize(k);
    return indices;
}

/**
 * @brief Compute angular error for bearing vector transformation
 */
double computeBearingError(
    const Eigen::Vector3d& b1,
    const Eigen::Vector3d& b2,
    const Eigen::Matrix3d& H) {

    Eigen::Vector3d b2_pred = (H * b1).normalized();
    double dot = std::abs(b2.dot(b2_pred));
    return std::acos(std::clamp(dot, 0.0, 1.0));
}

/**
 * @brief RANSAC for Homography using PoseLib 4-point solver
 */
Eigen::Matrix3d ransacHomographyPoseLib(
    const std::vector<Eigen::Vector3d>& bearings1,
    const std::vector<Eigen::Vector3d>& bearings2,
    double threshold_rad,
    int max_iterations,
    int& best_inlier_count) {

    std::mt19937 gen(42);
    best_inlier_count = 0;
    Eigen::Matrix3d best_H = Eigen::Matrix3d::Identity();

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Random sample 4 points
        auto indices = randomSample(static_cast<int>(bearings1.size()), 4, gen);

        std::vector<Eigen::Vector3d> x1_sample, x2_sample;
        for (int idx : indices) {
            x1_sample.push_back(bearings1[idx]);
            x2_sample.push_back(bearings2[idx]);
        }

        // PoseLib 4-point homography solver
        Eigen::Matrix3d H;
        int num_solutions = poselib::homography_4pt(x1_sample, x2_sample, &H);

        if (num_solutions == 0) continue;

        // Count inliers
        int inliers = 0;
        for (size_t i = 0; i < bearings1.size(); ++i) {
            double error = computeBearingError(bearings1[i], bearings2[i], H);
            if (error < threshold_rad) {
                inliers++;
            }
        }

        if (inliers > best_inlier_count) {
            best_inlier_count = inliers;
            best_H = H;
        }
    }

    return best_H;
}

/**
 * @brief RANSAC for Essential Matrix using PoseLib 5-point solver
 */
Eigen::Matrix3d ransacEssentialPoseLib(
    const std::vector<Eigen::Vector3d>& bearings1,
    const std::vector<Eigen::Vector3d>& bearings2,
    double threshold_rad,
    int max_iterations,
    int& best_inlier_count) {

    std::mt19937 gen(42);
    best_inlier_count = 0;
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d best_t = Eigen::Vector3d::UnitX();

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Random sample 5 points
        auto indices = randomSample(static_cast<int>(bearings1.size()), 5, gen);

        std::vector<Eigen::Vector3d> x1_sample, x2_sample;
        for (int idx : indices) {
            x1_sample.push_back(bearings1[idx]);
            x2_sample.push_back(bearings2[idx]);
        }

        // PoseLib 5-point relative pose solver
        std::vector<poselib::CameraPose> solutions;
        int num_solutions = poselib::relpose_5pt(x1_sample, x2_sample, &solutions);

        if (num_solutions == 0) continue;

        // Evaluate each solution
        for (const auto& pose : solutions) {
            Eigen::Quaterniond q(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
            Eigen::Matrix3d R = q.toRotationMatrix();
            Eigen::Vector3d t = pose.t;

            // Count inliers using epipolar constraint
            int inliers = 0;
            for (size_t i = 0; i < bearings1.size(); ++i) {
                // Essential matrix: E = [t]_x * R
                Eigen::Matrix3d t_skew;
                t_skew << 0, -t.z(), t.y(),
                          t.z(), 0, -t.x(),
                          -t.y(), t.x(), 0;
                Eigen::Matrix3d E = t_skew * R;

                // Epipolar constraint: x2^T * E * x1 = 0
                double error = std::abs(bearings2[i].transpose() * E * bearings1[i]);
                if (error < threshold_rad) {
                    inliers++;
                }
            }

            if (inliers > best_inlier_count) {
                best_inlier_count = inliers;
                best_R = R;
                best_t = t;
            }
        }
    }

    // Return Essential matrix: E = [t]_x * R
    Eigen::Matrix3d t_skew;
    t_skew << 0, -best_t.z(), best_t.y(),
              best_t.z(), 0, -best_t.x(),
              -best_t.y(), best_t.x(), 0;

    return t_skew * best_R;
}

/**
 * @brief Compute score for Homography (using bearing vectors)
 */
double computeHomographyScore(
    const std::vector<Eigen::Vector3d>& bearings1,
    const std::vector<Eigen::Vector3d>& bearings2,
    const Eigen::Matrix3d& H,
    double threshold_rad) {

    double total_error = 0;
    int count = 0;

    Eigen::Matrix3d H_inv = H.inverse();

    for (size_t i = 0; i < bearings1.size(); ++i) {
        // Forward error
        Eigen::Vector3d b2_pred = (H * bearings1[i]).normalized();
        double e_fwd = std::acos(std::clamp(std::abs(bearings2[i].dot(b2_pred)), 0.0, 1.0));

        // Backward error
        Eigen::Vector3d b1_pred = (H_inv * bearings2[i]).normalized();
        double e_bwd = std::acos(std::clamp(std::abs(bearings1[i].dot(b1_pred)), 0.0, 1.0));

        if (e_fwd < threshold_rad && e_bwd < threshold_rad) {
            total_error += e_fwd * e_fwd + e_bwd * e_bwd;
            count++;
        }
    }

    return (count > 0) ? total_error / count : std::numeric_limits<double>::max();
}

/**
 * @brief Compute score for Essential matrix
 */
double computeEssentialScore(
    const std::vector<Eigen::Vector3d>& bearings1,
    const std::vector<Eigen::Vector3d>& bearings2,
    const Eigen::Matrix3d& E,
    double threshold_rad) {

    double total_error = 0;
    int count = 0;

    for (size_t i = 0; i < bearings1.size(); ++i) {
        // Sampson-like error for Essential matrix
        Eigen::Vector3d Ex1 = E * bearings1[i];
        Eigen::Vector3d Etx2 = E.transpose() * bearings2[i];
        double x2tEx1 = bearings2[i].transpose() * E * bearings1[i];

        double denom = Ex1.head<2>().squaredNorm() + Etx2.head<2>().squaredNorm();
        if (denom > 1e-10) {
            double error = (x2tEx1 * x2tEx1) / denom;
            if (error < threshold_rad * threshold_rad) {
                total_error += error;
                count++;
            }
        }
    }

    return (count > 0) ? total_error / count : std::numeric_limits<double>::max();
}

/**
 * @brief Run model selection test with PoseLib
 */
void runTest(
    const std::string& scenario_name,
    const std::vector<Eigen::Vector3d>& points3D,
    const Eigen::Matrix3d& R1, const Eigen::Vector3d& t1,
    const Eigen::Matrix3d& R2, const Eigen::Vector3d& t2,
    double noise_sigma_rad) {

    std::cout << "\n=== " << scenario_name << " ===" << std::endl;

    // Project to bearing vectors
    std::vector<Eigen::Vector3d> bearings1, bearings2;
    for (const auto& p3d : points3D) {
        bearings1.push_back(projectToBearing(p3d, R1, t1));
        bearings2.push_back(projectToBearing(p3d, R2, t2));
    }

    // Add noise
    addNoise(bearings1, noise_sigma_rad);
    addNoise(bearings2, noise_sigma_rad);

    // RANSAC parameters
    const double threshold_rad = 0.01;  // ~0.5 degrees
    const int max_iterations = 1000;

    // Estimate Homography
    int inliers_H;
    Eigen::Matrix3d H = ransacHomographyPoseLib(bearings1, bearings2, threshold_rad, max_iterations, inliers_H);

    // Estimate Essential matrix
    int inliers_E;
    Eigen::Matrix3d E = ransacEssentialPoseLib(bearings1, bearings2, threshold_rad, max_iterations, inliers_E);

    // Compute scores
    double score_H = computeHomographyScore(bearings1, bearings2, H, threshold_rad);
    double score_E = computeEssentialScore(bearings1, bearings2, E, threshold_rad);

    // Compute ratio
    double sum = score_H + score_E;
    double ratio = (sum > 1e-10) ? score_E / sum : 0.5;
    bool use_homography = (ratio > 0.45);

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Points: " << bearings1.size() << std::endl;
    std::cout << "  H inliers: " << inliers_H << "/" << bearings1.size() << std::endl;
    std::cout << "  E inliers: " << inliers_E << "/" << bearings1.size() << std::endl;
    std::cout << "  Score H: " << score_H << std::endl;
    std::cout << "  Score E: " << score_E << std::endl;
    std::cout << "  Ratio R_H: " << ratio << std::endl;

    if (use_homography) {
        std::cout << "  --> Selected: HOMOGRAPHY (planar scene)" << std::endl;
    } else {
        std::cout << "  --> Selected: ESSENTIAL MATRIX (3D scene)" << std::endl;
    }
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "H/F Model Selection Demo using PoseLib" << std::endl;
    std::cout << "================================================" << std::endl;

    // Camera 1: at origin
    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1 = Eigen::Vector3d::Zero();

    // Camera 2: rotated and translated
    double angle = 15.0 * M_PI / 180.0;
    Eigen::Matrix3d R2;
    R2 = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY());
    Eigen::Vector3d t2(0.5, 0.0, 0.0);

    const int numPoints = 50;
    const double noise_sigma_rad = 0.005;  // ~0.3 degrees

    std::cout << "\nSetup:" << std::endl;
    std::cout << "  Rotation: " << angle * 180.0 / M_PI << " deg around Y" << std::endl;
    std::cout << "  Translation: [" << t2.transpose() << "]" << std::endl;
    std::cout << "  Noise: " << noise_sigma_rad * 180.0 / M_PI << " degrees" << std::endl;

    // Test 1: Planar Scene
    auto planarPoints = generatePlanarPoints(numPoints, 5.0);
    runTest("Scenario 1: PLANAR SCENE", planarPoints, R1, t1, R2, t2, noise_sigma_rad);

    // Test 2: 3D Scene
    auto points3D = generate3DPoints(numPoints);
    runTest("Scenario 2: 3D SCENE", points3D, R1, t1, R2, t2, noise_sigma_rad);

    // Test 3: Mixed scene
    std::vector<Eigen::Vector3d> mixedPoints;
    auto planar = generatePlanarPoints(numPoints * 3 / 4, 5.0);
    auto nonplanar = generate3DPoints(numPoints / 4);
    mixedPoints.insert(mixedPoints.end(), planar.begin(), planar.end());
    mixedPoints.insert(mixedPoints.end(), nonplanar.begin(), nonplanar.end());
    runTest("Scenario 3: MIXED SCENE", mixedPoints, R1, t1, R2, t2, noise_sigma_rad);

    std::cout << "\n================================================" << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "  - PoseLib uses bearing vectors (normalized 3D directions)" << std::endl;
    std::cout << "  - homography_4pt is a minimal 4-point solver" << std::endl;
    std::cout << "  - relpose_5pt returns up to 10 essential matrix solutions" << std::endl;
    std::cout << "================================================" << std::endl;

    return 0;
}
