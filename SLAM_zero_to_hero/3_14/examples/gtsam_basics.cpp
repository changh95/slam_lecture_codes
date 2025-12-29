/**
 * GTSAM Tutorial: Factor Graph Basics
 *
 * This example demonstrates:
 * 1. Basic factor graph structure
 * 2. Adding factors and variables
 * 3. 2D pose graph optimization
 */

#include <iostream>
#include <cmath>

// GTSAM headers
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

using namespace std;
using namespace gtsam;

int main(int argc, char** argv) {
    cout << "=== GTSAM Tutorial: 2D Pose Graph Optimization ===\n" << endl;

    // =========================================================================
    // Step 1: Create factor graph
    // =========================================================================
    cout << "Step 1: Creating factor graph..." << endl;

    NonlinearFactorGraph graph;

    // =========================================================================
    // Step 2: Add prior factor on first pose
    // =========================================================================
    cout << "Step 2: Adding prior factor..." << endl;

    // Prior on first pose (anchor)
    Pose2 prior_pose(0.0, 0.0, 0.0);  // x, y, theta
    auto prior_noise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));  // std dev

    graph.addPrior(Symbol('x', 0), prior_pose, prior_noise);
    cout << "  Added prior on x0: " << prior_pose << endl;

    // =========================================================================
    // Step 3: Add odometry factors (between factors)
    // =========================================================================
    cout << "\nStep 3: Adding odometry factors..." << endl;

    // Noise model for odometry
    auto odometry_noise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));

    // Odometry measurements (relative poses)
    vector<Pose2> odometry = {
        Pose2(1.0, 0.0, 0.0),           // x0 -> x1: forward 1m
        Pose2(1.0, 0.0, M_PI / 2),      // x1 -> x2: forward + turn 90°
        Pose2(1.0, 0.0, 0.0),           // x2 -> x3: forward 1m
        Pose2(1.0, 0.0, M_PI / 2)       // x3 -> x4: forward + turn 90°
    };

    for (size_t i = 0; i < odometry.size(); ++i) {
        graph.add(BetweenFactor<Pose2>(
            Symbol('x', i), Symbol('x', i + 1),
            odometry[i], odometry_noise));
        cout << "  Added odometry x" << i << " -> x" << (i + 1)
             << ": " << odometry[i] << endl;
    }

    // =========================================================================
    // Step 4: Add loop closure factor
    // =========================================================================
    cout << "\nStep 4: Adding loop closure factor..." << endl;

    // Loop closure: x4 sees x0 again
    Pose2 loop_closure(1.0, 0.0, M_PI / 2);  // Should close the square
    auto loop_noise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));

    graph.add(BetweenFactor<Pose2>(
        Symbol('x', 4), Symbol('x', 0),
        loop_closure, loop_noise));
    cout << "  Added loop closure x4 -> x0: " << loop_closure << endl;

    // =========================================================================
    // Step 5: Create initial estimates
    // =========================================================================
    cout << "\nStep 5: Creating initial estimates..." << endl;

    Values initial_estimate;

    // Ground truth trajectory (square)
    vector<Pose2> ground_truth = {
        Pose2(0.0, 0.0, 0.0),
        Pose2(1.0, 0.0, 0.0),
        Pose2(1.0, 1.0, M_PI / 2),
        Pose2(0.0, 1.0, M_PI),
        Pose2(0.0, 0.0, -M_PI / 2)
    };

    // Add noisy initial estimates
    for (size_t i = 0; i < 5; ++i) {
        double noise_x = 0.1 * ((rand() % 100 - 50) / 50.0);
        double noise_y = 0.1 * ((rand() % 100 - 50) / 50.0);
        double noise_theta = 0.05 * ((rand() % 100 - 50) / 50.0);

        Pose2 noisy_pose(
            ground_truth[i].x() + noise_x,
            ground_truth[i].y() + noise_y,
            ground_truth[i].theta() + noise_theta
        );
        initial_estimate.insert(Symbol('x', i), noisy_pose);
        cout << "  Initial x" << i << ": " << noisy_pose << endl;
    }

    // =========================================================================
    // Step 6: Optimize
    // =========================================================================
    cout << "\nStep 6: Optimizing..." << endl;

    // Print initial error
    cout << "  Initial error: " << graph.error(initial_estimate) << endl;

    // Optimize using Levenberg-Marquardt
    LevenbergMarquardtParams params;
    params.setVerbosity("SILENT");
    params.setMaxIterations(100);

    LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
    Values result = optimizer.optimize();

    cout << "  Final error: " << graph.error(result) << endl;
    cout << "  Iterations: " << optimizer.iterations() << endl;

    // =========================================================================
    // Step 7: Print results
    // =========================================================================
    cout << "\n=== Optimization Results ===" << endl;
    cout << "\nPose | Ground Truth        | Optimized           | Error" << endl;
    cout << string(65, '-') << endl;

    for (size_t i = 0; i < 5; ++i) {
        Pose2 optimized = result.at<Pose2>(Symbol('x', i));
        Pose2 gt = ground_truth[i];

        double error = sqrt(
            pow(optimized.x() - gt.x(), 2) +
            pow(optimized.y() - gt.y(), 2)
        );

        printf("  x%zu | (%5.2f, %5.2f, %5.2f) | (%5.2f, %5.2f, %5.2f) | %.4f\n",
               i,
               gt.x(), gt.y(), gt.theta(),
               optimized.x(), optimized.y(), optimized.theta(),
               error);
    }

    // =========================================================================
    // Step 8: Compute marginal covariances
    // =========================================================================
    cout << "\n=== Marginal Covariances ===" << endl;

    Marginals marginals(graph, result);

    for (size_t i = 0; i < 5; ++i) {
        Matrix cov = marginals.marginalCovariance(Symbol('x', i));
        cout << "x" << i << " covariance (diagonal): ["
             << sqrt(cov(0, 0)) << ", "
             << sqrt(cov(1, 1)) << ", "
             << sqrt(cov(2, 2)) << "]" << endl;
    }

    cout << "\n=== Complete ===" << endl;

    return 0;
}
