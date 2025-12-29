/**
 * g2o Tutorial: Graph Optimization Basics
 *
 * This example demonstrates:
 * 1. Basic g2o graph structure
 * 2. Adding vertices (parameters) and edges (constraints)
 * 3. 2D pose graph optimization (SLAM-like scenario)
 */

#include <iostream>
#include <cmath>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam2d/types_slam2d.h>

using namespace g2o;
using namespace std;

int main(int argc, char** argv) {
    cout << "=== g2o Tutorial: 2D Pose Graph Optimization ===\n" << endl;

    // =========================================================================
    // Step 1: Create the optimizer
    // =========================================================================
    cout << "Step 1: Creating optimizer..." << endl;

    // Create optimizer
    SparseOptimizer optimizer;
    optimizer.setVerbose(true);

    // Set up the solver
    // BlockSolver_6_3: 6-DOF poses, 3-DOF landmarks (for BA)
    // For 2D SLAM: BlockSolver_3_2: 3-DOF poses (x, y, theta), 2-DOF landmarks
    using BlockSolverType = BlockSolver<BlockSolverTraits<-1, -1>>;
    using LinearSolverType = LinearSolverDense<BlockSolverType::PoseMatrixType>;

    auto solver = new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
    );
    optimizer.setAlgorithm(solver);

    // =========================================================================
    // Step 2: Add vertices (robot poses)
    // =========================================================================
    cout << "Step 2: Adding vertices (poses)..." << endl;

    // Create a simple square trajectory with 5 poses
    // Ground truth: (0,0,0) -> (1,0,0) -> (1,1,pi/2) -> (0,1,pi) -> (0,0,-pi/2)
    vector<SE2> true_poses = {
        SE2(0.0, 0.0, 0.0),
        SE2(1.0, 0.0, 0.0),
        SE2(1.0, 1.0, M_PI / 2),
        SE2(0.0, 1.0, M_PI),
        SE2(0.0, 0.0, -M_PI / 2)
    };

    // Add noise to initial estimates
    double noise_xy = 0.1;
    double noise_theta = 0.05;

    for (size_t i = 0; i < true_poses.size(); ++i) {
        VertexSE2* v = new VertexSE2();
        v->setId(i);

        // Add noise to pose (except first one)
        if (i == 0) {
            v->setEstimate(true_poses[i]);
            v->setFixed(true);  // Fix first pose (anchor)
        } else {
            double noisy_x = true_poses[i].translation().x() + noise_xy * ((rand() % 100 - 50) / 50.0);
            double noisy_y = true_poses[i].translation().y() + noise_xy * ((rand() % 100 - 50) / 50.0);
            double noisy_theta = true_poses[i].rotation().angle() + noise_theta * ((rand() % 100 - 50) / 50.0);
            v->setEstimate(SE2(noisy_x, noisy_y, noisy_theta));
        }

        optimizer.addVertex(v);
        cout << "  Added vertex " << i << ": (" << v->estimate().translation().x()
             << ", " << v->estimate().translation().y()
             << ", " << v->estimate().rotation().angle() << ")" << endl;
    }

    // =========================================================================
    // Step 3: Add edges (odometry constraints between consecutive poses)
    // =========================================================================
    cout << "\nStep 3: Adding odometry edges..." << endl;

    // Information matrix (inverse covariance)
    Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
    information(0, 0) = 100.0;  // x precision
    information(1, 1) = 100.0;  // y precision
    information(2, 2) = 100.0;  // theta precision

    // Odometry measurements (relative transformations)
    vector<SE2> odometry = {
        SE2(1.0, 0.0, 0.0),           // 0 -> 1
        SE2(0.0, 1.0, M_PI / 2),      // 1 -> 2
        SE2(-1.0, 0.0, M_PI / 2),     // 2 -> 3
        SE2(0.0, -1.0, -M_PI / 2)     // 3 -> 4
    };

    for (size_t i = 0; i < odometry.size(); ++i) {
        EdgeSE2* e = new EdgeSE2();
        e->setVertex(0, optimizer.vertex(i));
        e->setVertex(1, optimizer.vertex(i + 1));
        e->setMeasurement(odometry[i]);
        e->setInformation(information);

        optimizer.addEdge(e);
        cout << "  Added edge " << i << " -> " << (i + 1) << endl;
    }

    // =========================================================================
    // Step 4: Add loop closure edge (pose 4 back to pose 0)
    // =========================================================================
    cout << "\nStep 4: Adding loop closure edge (4 -> 0)..." << endl;

    EdgeSE2* loop_edge = new EdgeSE2();
    loop_edge->setVertex(0, optimizer.vertex(4));
    loop_edge->setVertex(1, optimizer.vertex(0));
    loop_edge->setMeasurement(SE2(0.0, 0.0, M_PI / 2));  // Close the loop
    loop_edge->setInformation(information);
    optimizer.addEdge(loop_edge);

    // =========================================================================
    // Step 5: Optimize
    // =========================================================================
    cout << "\nStep 5: Optimizing..." << endl;

    optimizer.initializeOptimization();
    int iterations = optimizer.optimize(20);
    cout << "Optimization finished in " << iterations << " iterations." << endl;

    // =========================================================================
    // Step 6: Print results
    // =========================================================================
    cout << "\n=== Optimization Results ===" << endl;
    cout << "\nPose | True (x, y, theta) | Optimized (x, y, theta) | Error" << endl;
    cout << string(70, '-') << endl;

    for (size_t i = 0; i < true_poses.size(); ++i) {
        VertexSE2* v = static_cast<VertexSE2*>(optimizer.vertex(i));
        SE2 optimized = v->estimate();
        SE2 ground_truth = true_poses[i];

        double error_x = optimized.translation().x() - ground_truth.translation().x();
        double error_y = optimized.translation().y() - ground_truth.translation().y();
        double error = sqrt(error_x * error_x + error_y * error_y);

        printf("  %zu  | (%5.2f, %5.2f, %5.2f) | (%5.2f, %5.2f, %5.2f) | %.4f\n",
               i,
               ground_truth.translation().x(), ground_truth.translation().y(),
               ground_truth.rotation().angle(),
               optimized.translation().x(), optimized.translation().y(),
               optimized.rotation().angle(),
               error);
    }

    cout << "\n=== Complete ===" << endl;

    return 0;
}
