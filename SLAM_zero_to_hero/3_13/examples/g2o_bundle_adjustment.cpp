/**
 * g2o Tutorial: Bundle Adjustment with BAL Dataset
 *
 * This example demonstrates:
 * 1. Loading BAL (Bundle Adjustment in the Large) dataset
 * 2. Setting up bundle adjustment problem in g2o
 * 3. Optimizing camera poses and 3D points
 *
 * BAL dataset format:
 *   <num_cameras> <num_points> <num_observations>
 *   <camera_idx> <point_idx> <x> <y>  (observations)
 *   ...
 *   <camera params: rotation(3), translation(3), focal, k1, k2>
 *   ...
 *   <point: x, y, z>
 *   ...
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace g2o;
using namespace std;

// Convert angle-axis to rotation matrix
Eigen::Matrix3d AngleAxisToRotationMatrix(const Eigen::Vector3d& angle_axis) {
    double theta = angle_axis.norm();
    if (theta < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d axis = angle_axis / theta;
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
        -axis.y(), axis.x(), 0;
    return Eigen::Matrix3d::Identity() + sin(theta) * K + (1 - cos(theta)) * K * K;
}

int main(int argc, char** argv) {
    cout << "=== g2o Tutorial: Bundle Adjustment with BAL Dataset ===\n" << endl;

    // Check for input file
    string bal_file = "problem-49-7776-pre.txt";  // Default BAL problem
    if (argc > 1) {
        bal_file = argv[1];
    }

    cout << "Reading BAL file: " << bal_file << endl;

    ifstream fin(bal_file);
    if (!fin) {
        cerr << "Error: Cannot open file " << bal_file << endl;
        cerr << "\nDownload a BAL dataset from:" << endl;
        cerr << "https://grail.cs.washington.edu/projects/bal/" << endl;
        cerr << "\nExample datasets:" << endl;
        cerr << "  problem-49-7776-pre.txt  (49 cameras, 7776 points)" << endl;
        cerr << "  problem-21-11315-pre.txt (21 cameras, 11315 points)" << endl;
        return 1;
    }

    // =========================================================================
    // Step 1: Read BAL dataset
    // =========================================================================
    cout << "\nStep 1: Reading BAL dataset..." << endl;

    int num_cameras, num_points, num_observations;
    fin >> num_cameras >> num_points >> num_observations;

    cout << "  Cameras: " << num_cameras << endl;
    cout << "  Points: " << num_points << endl;
    cout << "  Observations: " << num_observations << endl;

    // Read observations
    struct Observation {
        int camera_idx;
        int point_idx;
        double x, y;
    };
    vector<Observation> observations(num_observations);

    for (int i = 0; i < num_observations; ++i) {
        fin >> observations[i].camera_idx >> observations[i].point_idx
            >> observations[i].x >> observations[i].y;
    }

    // Read camera parameters
    // BAL format: rotation(3), translation(3), focal, k1, k2
    struct Camera {
        Eigen::Vector3d rotation;  // angle-axis
        Eigen::Vector3d translation;
        double focal;
        double k1, k2;  // distortion
    };
    vector<Camera> cameras(num_cameras);

    for (int i = 0; i < num_cameras; ++i) {
        fin >> cameras[i].rotation.x() >> cameras[i].rotation.y() >> cameras[i].rotation.z();
        fin >> cameras[i].translation.x() >> cameras[i].translation.y() >> cameras[i].translation.z();
        fin >> cameras[i].focal >> cameras[i].k1 >> cameras[i].k2;
    }

    // Read 3D points
    vector<Eigen::Vector3d> points(num_points);
    for (int i = 0; i < num_points; ++i) {
        fin >> points[i].x() >> points[i].y() >> points[i].z();
    }
    fin.close();

    // =========================================================================
    // Step 2: Set up g2o optimizer
    // =========================================================================
    cout << "\nStep 2: Setting up optimizer..." << endl;

    SparseOptimizer optimizer;
    optimizer.setVerbose(true);

    // BlockSolver_6_3: 6-DOF camera poses, 3-DOF points
    using BlockSolverType = BlockSolver<BlockSolverTraits<6, 3>>;
    using LinearSolverType = LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto solver = new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
    );
    optimizer.setAlgorithm(solver);

    // =========================================================================
    // Step 3: Add camera vertices
    // =========================================================================
    cout << "\nStep 3: Adding camera vertices..." << endl;

    for (int i = 0; i < num_cameras; ++i) {
        // Convert BAL camera to g2o SE3
        Eigen::Matrix3d R = AngleAxisToRotationMatrix(cameras[i].rotation);
        Eigen::Vector3d t = cameras[i].translation;

        // g2o uses inverse camera pose (world to camera)
        SE3Quat pose(R, t);

        VertexSE3Expmap* v = new VertexSE3Expmap();
        v->setId(i);
        v->setEstimate(pose);

        // Fix first camera to remove gauge freedom
        if (i == 0) {
            v->setFixed(true);
        }

        optimizer.addVertex(v);
    }
    cout << "  Added " << num_cameras << " camera vertices" << endl;

    // =========================================================================
    // Step 4: Add point vertices
    // =========================================================================
    cout << "\nStep 4: Adding point vertices..." << endl;

    for (int i = 0; i < num_points; ++i) {
        VertexPointXYZ* v = new VertexPointXYZ();
        v->setId(num_cameras + i);
        v->setEstimate(points[i]);
        v->setMarginalized(true);  // Schur complement

        optimizer.addVertex(v);
    }
    cout << "  Added " << num_points << " point vertices" << endl;

    // =========================================================================
    // Step 5: Add observation edges
    // =========================================================================
    cout << "\nStep 5: Adding observation edges..." << endl;

    for (int i = 0; i < num_observations; ++i) {
        EdgeProjectXYZ2UV* edge = new EdgeProjectXYZ2UV();

        edge->setVertex(0, optimizer.vertex(num_cameras + observations[i].point_idx));
        edge->setVertex(1, optimizer.vertex(observations[i].camera_idx));

        Eigen::Vector2d measurement(observations[i].x, observations[i].y);
        edge->setMeasurement(measurement);

        // Information matrix (assume unit variance)
        edge->setInformation(Eigen::Matrix2d::Identity());

        // Robust kernel for outlier rejection
        edge->setRobustKernel(new RobustKernelHuber());

        // Camera intrinsics (simplified: only focal length)
        double focal = cameras[observations[i].camera_idx].focal;
        CameraParameters* cam_params = new CameraParameters(focal, Eigen::Vector2d(0, 0), 0);
        cam_params->setId(0);

        if (!optimizer.parameter(0)) {
            optimizer.addParameter(cam_params);
        }
        edge->setParameterId(0, 0);

        optimizer.addEdge(edge);
    }
    cout << "  Added " << num_observations << " observation edges" << endl;

    // =========================================================================
    // Step 6: Optimize
    // =========================================================================
    cout << "\nStep 6: Optimizing..." << endl;

    optimizer.initializeOptimization();

    // Compute initial error
    optimizer.computeActiveErrors();
    double initial_chi2 = optimizer.chi2();
    cout << "  Initial chi2: " << initial_chi2 << endl;

    // Run optimization
    int iterations = optimizer.optimize(50);
    cout << "  Optimization finished in " << iterations << " iterations" << endl;

    // Compute final error
    optimizer.computeActiveErrors();
    double final_chi2 = optimizer.chi2();
    cout << "  Final chi2: " << final_chi2 << endl;
    cout << "  Improvement: " << (1.0 - final_chi2 / initial_chi2) * 100 << "%" << endl;

    // =========================================================================
    // Step 7: Output results
    // =========================================================================
    cout << "\n=== Results Summary ===" << endl;
    cout << "Problem size:" << endl;
    cout << "  Cameras: " << num_cameras << endl;
    cout << "  Points: " << num_points << endl;
    cout << "  Observations: " << num_observations << endl;
    cout << "Optimization:" << endl;
    cout << "  Iterations: " << iterations << endl;
    cout << "  Initial error: " << initial_chi2 << endl;
    cout << "  Final error: " << final_chi2 << endl;
    cout << "  RMSE: " << sqrt(final_chi2 / num_observations) << " pixels" << endl;

    cout << "\n=== Complete ===" << endl;

    return 0;
}
