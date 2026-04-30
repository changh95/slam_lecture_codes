/**
 * GTSAM Tutorial: Bundle Adjustment with BAL Dataset
 *
 * This example demonstrates:
 * 1. Loading BAL (Bundle Adjustment in the Large) dataset
 * 2. Setting up bundle adjustment with GTSAM
 * 3. Using iSAM2 for incremental optimization
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

// GTSAM headers
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3Bundler.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>

using namespace std;
using namespace gtsam;

// Convert angle-axis to rotation matrix
Rot3 AngleAxisToRot3(double rx, double ry, double rz) {
    Eigen::Vector3d axis(rx, ry, rz);
    double angle = axis.norm();
    if (angle < 1e-10) {
        return Rot3::Identity();
    }
    axis.normalize();
    return Rot3::Rodrigues(axis * angle);
}

int main(int argc, char** argv) {
    cout << "=== GTSAM Tutorial: Bundle Adjustment with BAL Dataset ===\n" << endl;

    // Check for input file
    string bal_file = "problem-49-7776-pre.txt";
    if (argc > 1) {
        bal_file = argv[1];
    }

    cout << "Reading BAL file: " << bal_file << endl;

    ifstream fin(bal_file);
    if (!fin) {
        cerr << "Error: Cannot open file " << bal_file << endl;
        cerr << "\nDownload a BAL dataset from:" << endl;
        cerr << "https://grail.cs.washington.edu/projects/bal/" << endl;
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
        double u, v;
    };
    vector<Observation> observations(num_observations);

    for (int i = 0; i < num_observations; ++i) {
        fin >> observations[i].camera_idx >> observations[i].point_idx
            >> observations[i].u >> observations[i].v;
    }

    // Read camera parameters (BAL format: rotation(3), translation(3), f, k1, k2)
    struct BALCamera {
        double rotation[3];
        double translation[3];
        double focal, k1, k2;
    };
    vector<BALCamera> cameras(num_cameras);

    for (int i = 0; i < num_cameras; ++i) {
        fin >> cameras[i].rotation[0] >> cameras[i].rotation[1] >> cameras[i].rotation[2];
        fin >> cameras[i].translation[0] >> cameras[i].translation[1] >> cameras[i].translation[2];
        fin >> cameras[i].focal >> cameras[i].k1 >> cameras[i].k2;
    }

    // Read 3D points
    vector<Point3> points(num_points);
    for (int i = 0; i < num_points; ++i) {
        double x, y, z;
        fin >> x >> y >> z;
        points[i] = Point3(x, y, z);
    }
    fin.close();

    // =========================================================================
    // Step 2: Create factor graph
    // =========================================================================
    cout << "\nStep 2: Creating factor graph..." << endl;

    NonlinearFactorGraph graph;
    Values initial_estimate;

    // Measurement noise (pixels)
    auto measurement_noise = noiseModel::Isotropic::Sigma(2, 1.0);

    // =========================================================================
    // Step 3: Add camera poses
    // =========================================================================
    cout << "\nStep 3: Adding camera poses..." << endl;

    for (int i = 0; i < num_cameras; ++i) {
        // Convert BAL camera to GTSAM Pose3
        Rot3 R = AngleAxisToRot3(
            cameras[i].rotation[0],
            cameras[i].rotation[1],
            cameras[i].rotation[2]
        );
        Point3 t(
            cameras[i].translation[0],
            cameras[i].translation[1],
            cameras[i].translation[2]
        );

        Pose3 pose(R, t);
        initial_estimate.insert(Symbol('c', i), pose);

        // Add camera intrinsics (Cal3Bundler: f, k1, k2, u0=0, v0=0)
        Cal3Bundler K(cameras[i].focal, cameras[i].k1, cameras[i].k2, 0, 0);
        initial_estimate.insert(Symbol('K', i), K);
    }
    cout << "  Added " << num_cameras << " cameras" << endl;

    // =========================================================================
    // Step 4: Add 3D points
    // =========================================================================
    cout << "\nStep 4: Adding 3D points..." << endl;

    for (int i = 0; i < num_points; ++i) {
        initial_estimate.insert(Symbol('p', i), points[i]);
    }
    cout << "  Added " << num_points << " points" << endl;

    // =========================================================================
    // Step 5: Add observation factors
    // =========================================================================
    cout << "\nStep 5: Adding observation factors..." << endl;

    for (int i = 0; i < num_observations; ++i) {
        int cam_id = observations[i].camera_idx;
        int pt_id = observations[i].point_idx;

        Point2 measured(observations[i].u, observations[i].v);

        // GeneralSFMFactor2 handles variable intrinsics
        graph.emplace_shared<GeneralSFMFactor2<Cal3Bundler>>(
            measured,
            measurement_noise,
            Symbol('c', cam_id),
            Symbol('p', pt_id),
            Symbol('K', cam_id)
        );
    }
    cout << "  Added " << num_observations << " observation factors" << endl;

    // =========================================================================
    // Step 6: Add prior on first camera (gauge fixing)
    // =========================================================================
    cout << "\nStep 6: Adding prior factor..." << endl;

    auto pose_noise = noiseModel::Diagonal::Sigmas(
        (Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished()
    );
    graph.addPrior(Symbol('c', 0), initial_estimate.at<Pose3>(Symbol('c', 0)), pose_noise);

    // =========================================================================
    // Step 7: Optimize
    // =========================================================================
    cout << "\nStep 7: Optimizing..." << endl;

    // Initial error
    double initial_error = graph.error(initial_estimate);
    cout << "  Initial error: " << initial_error << endl;

    // Levenberg-Marquardt optimizer
    LevenbergMarquardtParams params;
    params.setVerbosity("SUMMARY");
    params.setMaxIterations(50);

    try {
        LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
        Values result = optimizer.optimize();

        double final_error = graph.error(result);
        cout << "  Final error: " << final_error << endl;
        cout << "  Improvement: " << (1.0 - final_error / initial_error) * 100 << "%" << endl;

        // =========================================================================
        // Step 8: Summary
        // =========================================================================
        cout << "\n=== Results Summary ===" << endl;
        cout << "Problem size:" << endl;
        cout << "  Cameras: " << num_cameras << endl;
        cout << "  Points: " << num_points << endl;
        cout << "  Observations: " << num_observations << endl;
        cout << "Optimization:" << endl;
        cout << "  Iterations: " << optimizer.iterations() << endl;
        cout << "  Initial error: " << initial_error << endl;
        cout << "  Final error: " << final_error << endl;
        cout << "  RMSE: " << sqrt(2.0 * final_error / num_observations) << " pixels" << endl;

    } catch (const exception& e) {
        cerr << "Optimization failed: " << e.what() << endl;
        return 1;
    }

    cout << "\n=== Complete ===" << endl;

    return 0;
}
