/**
 * Ceres-Solver Tutorial: Basics
 *
 * This example demonstrates:
 * 1. Basic Ceres problem setup
 * 2. Cost functions and automatic differentiation
 * 3. 2D pose graph optimization
 */

#include <iostream>
#include <cmath>
#include <vector>

#include <ceres/ceres.h>

using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;

// ============================================================================
// Example 1: Simple curve fitting
// ============================================================================

// Cost functor for curve fitting: y = exp(m*x + c)
struct ExponentialResidual {
    ExponentialResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const m, const T* const c, T* residual) const {
        residual[0] = y_ - exp(m[0] * x_ + c[0]);
        return true;
    }

private:
    const double x_;
    const double y_;
};

void curveFittingExample() {
    cout << "=== Example 1: Curve Fitting ===\n" << endl;

    // Generate synthetic data: y = exp(0.3*x + 0.1) + noise
    const int NUM_OBSERVATIONS = 50;
    vector<double> data_x, data_y;

    srand(42);
    for (int i = 0; i < NUM_OBSERVATIONS; ++i) {
        double x = static_cast<double>(i) / NUM_OBSERVATIONS;
        double y = exp(0.3 * x + 0.1) + 0.01 * (rand() % 100 - 50) / 50.0;
        data_x.push_back(x);
        data_y.push_back(y);
    }

    // Initial guess
    double m = 0.0;
    double c = 0.0;

    cout << "Ground truth: m=0.3, c=0.1" << endl;
    cout << "Initial guess: m=" << m << ", c=" << c << endl;

    // Build the problem
    Problem problem;

    for (int i = 0; i < NUM_OBSERVATIONS; ++i) {
        CostFunction* cost_function =
            new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                new ExponentialResidual(data_x[i], data_y[i]));
        problem.AddResidualBlock(cost_function, nullptr, &m, &c);
    }

    // Solve
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << "\n" << summary.BriefReport() << endl;
    cout << "Estimated: m=" << m << ", c=" << c << endl;
}

// ============================================================================
// Example 2: 2D Pose Graph Optimization
// ============================================================================

// 2D pose: [x, y, theta]
struct Pose2d {
    double x, y, theta;

    Pose2d() : x(0), y(0), theta(0) {}
    Pose2d(double x_, double y_, double theta_) : x(x_), y(y_), theta(theta_) {}
};

// Normalize angle to [-pi, pi]
template <typename T>
T NormalizeAngle(const T& angle) {
    T two_pi = T(2.0 * M_PI);
    return angle - two_pi * floor((angle + T(M_PI)) / two_pi);
}

// Cost function for odometry constraint between two poses
struct PoseGraphCostFunction {
    PoseGraphCostFunction(double dx, double dy, double dtheta,
                          double weight_xy, double weight_theta)
        : dx_(dx), dy_(dy), dtheta_(dtheta),
          weight_xy_(weight_xy), weight_theta_(weight_theta) {}

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residual) const {
        // pose_i = [x_i, y_i, theta_i]
        // pose_j = [x_j, y_j, theta_j]

        // Compute relative pose in pose_i's frame
        T cos_theta_i = cos(pose_i[2]);
        T sin_theta_i = sin(pose_i[2]);

        T dx = pose_j[0] - pose_i[0];
        T dy = pose_j[1] - pose_i[1];

        // Transform to local frame
        T local_dx = cos_theta_i * dx + sin_theta_i * dy;
        T local_dy = -sin_theta_i * dx + cos_theta_i * dy;
        T local_dtheta = NormalizeAngle(pose_j[2] - pose_i[2]);

        // Residuals
        residual[0] = weight_xy_ * (local_dx - T(dx_));
        residual[1] = weight_xy_ * (local_dy - T(dy_));
        residual[2] = weight_theta_ * NormalizeAngle(local_dtheta - T(dtheta_));

        return true;
    }

    static CostFunction* Create(double dx, double dy, double dtheta,
                                 double weight_xy, double weight_theta) {
        return new AutoDiffCostFunction<PoseGraphCostFunction, 3, 3, 3>(
            new PoseGraphCostFunction(dx, dy, dtheta, weight_xy, weight_theta));
    }

private:
    const double dx_, dy_, dtheta_;
    const double weight_xy_, weight_theta_;
};

void poseGraphExample() {
    cout << "\n=== Example 2: 2D Pose Graph Optimization ===\n" << endl;

    // Ground truth: square trajectory
    vector<Pose2d> ground_truth = {
        Pose2d(0.0, 0.0, 0.0),
        Pose2d(1.0, 0.0, 0.0),
        Pose2d(1.0, 1.0, M_PI / 2),
        Pose2d(0.0, 1.0, M_PI),
        Pose2d(0.0, 0.0, -M_PI / 2)
    };

    // Initialize poses with noise
    vector<array<double, 3>> poses(5);
    for (size_t i = 0; i < 5; ++i) {
        double noise_xy = 0.1 * (rand() % 100 - 50) / 50.0;
        double noise_theta = 0.05 * (rand() % 100 - 50) / 50.0;
        poses[i] = {
            ground_truth[i].x + noise_xy,
            ground_truth[i].y + noise_xy,
            ground_truth[i].theta + noise_theta
        };
    }

    cout << "Initial poses (with noise):" << endl;
    for (size_t i = 0; i < 5; ++i) {
        printf("  x%zu: (%.3f, %.3f, %.3f)\n", i, poses[i][0], poses[i][1], poses[i][2]);
    }

    // Build problem
    Problem problem;

    // Odometry constraints
    vector<tuple<int, int, double, double, double>> edges = {
        {0, 1, 1.0, 0.0, 0.0},        // 0->1
        {1, 2, 0.0, 1.0, M_PI / 2},   // 1->2
        {2, 3, -1.0, 0.0, M_PI / 2},  // 2->3
        {3, 4, 0.0, -1.0, -M_PI / 2}, // 3->4
        {4, 0, 0.0, 0.0, M_PI / 2}    // 4->0 (loop closure)
    };

    for (const auto& edge : edges) {
        int i = get<0>(edge);
        int j = get<1>(edge);
        double dx = get<2>(edge);
        double dy = get<3>(edge);
        double dtheta = get<4>(edge);

        problem.AddResidualBlock(
            PoseGraphCostFunction::Create(dx, dy, dtheta, 10.0, 5.0),
            nullptr,
            poses[i].data(),
            poses[j].data());
    }

    // Fix first pose
    problem.SetParameterBlockConstant(poses[0].data());

    // Solve
    Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;

    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << "\n" << summary.BriefReport() << endl;

    // Print results
    cout << "\nOptimized poses vs Ground truth:" << endl;
    cout << "Pose | Optimized           | Ground Truth        | Error" << endl;
    cout << string(65, '-') << endl;

    for (size_t i = 0; i < 5; ++i) {
        double error = sqrt(
            pow(poses[i][0] - ground_truth[i].x, 2) +
            pow(poses[i][1] - ground_truth[i].y, 2)
        );
        printf("  x%zu | (%5.3f, %5.3f, %5.3f) | (%5.3f, %5.3f, %5.3f) | %.4f\n",
               i,
               poses[i][0], poses[i][1], poses[i][2],
               ground_truth[i].x, ground_truth[i].y, ground_truth[i].theta,
               error);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    cout << "=== Ceres-Solver Tutorial ===\n" << endl;

    curveFittingExample();
    poseGraphExample();

    cout << "\n=== Complete ===" << endl;

    return 0;
}
