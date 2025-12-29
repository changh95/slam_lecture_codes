/**
 * Ceres-Solver Tutorial: Bundle Adjustment with BAL Dataset
 *
 * This example demonstrates:
 * 1. Loading BAL dataset
 * 2. Setting up bundle adjustment with Ceres
 * 3. Using loss functions for robustness
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

// ============================================================================
// BAL Camera Model
// ============================================================================

// BAL uses angle-axis rotation (3 params) + translation (3) + f, k1, k2
// Projection: p = -P/P.z, p' = f * (1 + k1*r^2 + k2*r^4) * p

struct BALReprojectionError {
    BALReprojectionError(double observed_x, double observed_y)
        : observed_x_(observed_x), observed_y_(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // camera[0,1,2]: angle-axis rotation
        // camera[3,4,5]: translation
        // camera[6]: focal length
        // camera[7]: k1 (radial distortion)
        // camera[8]: k2 (radial distortion)

        // Rotate point
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Translate
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Perspective projection
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Radial distortion
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + camera[7] * r2 + camera[8] * r2 * r2;

        // Final projected point
        T predicted_x = camera[6] * distortion * xp;
        T predicted_y = camera[6] * distortion * yp;

        // Residuals
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;

        return true;
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y) {
        return new ceres::AutoDiffCostFunction<BALReprojectionError, 2, 9, 3>(
            new BALReprojectionError(observed_x, observed_y));
    }

private:
    double observed_x_;
    double observed_y_;
};

// ============================================================================
// BAL Dataset Reader
// ============================================================================

class BALProblem {
public:
    bool LoadFile(const string& filename) {
        ifstream fin(filename);
        if (!fin) {
            cerr << "Error: Cannot open " << filename << endl;
            return false;
        }

        fin >> num_cameras_ >> num_points_ >> num_observations_;

        // Allocate memory
        camera_index_.resize(num_observations_);
        point_index_.resize(num_observations_);
        observations_.resize(2 * num_observations_);
        cameras_.resize(9 * num_cameras_);
        points_.resize(3 * num_points_);

        // Read observations
        for (int i = 0; i < num_observations_; ++i) {
            fin >> camera_index_[i] >> point_index_[i]
                >> observations_[2 * i] >> observations_[2 * i + 1];
        }

        // Read cameras
        for (int i = 0; i < num_cameras_ * 9; ++i) {
            fin >> cameras_[i];
        }

        // Read points
        for (int i = 0; i < num_points_ * 3; ++i) {
            fin >> points_[i];
        }

        return true;
    }

    int num_cameras() const { return num_cameras_; }
    int num_points() const { return num_points_; }
    int num_observations() const { return num_observations_; }

    double* mutable_camera(int i) { return &cameras_[9 * i]; }
    double* mutable_point(int i) { return &points_[3 * i]; }

    int camera_index(int i) const { return camera_index_[i]; }
    int point_index(int i) const { return point_index_[i]; }

    double observation_x(int i) const { return observations_[2 * i]; }
    double observation_y(int i) const { return observations_[2 * i + 1]; }

private:
    int num_cameras_;
    int num_points_;
    int num_observations_;

    vector<int> camera_index_;
    vector<int> point_index_;
    vector<double> observations_;
    vector<double> cameras_;
    vector<double> points_;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    cout << "=== Ceres-Solver Tutorial: Bundle Adjustment ===\n" << endl;

    string bal_file = "problem-49-7776-pre.txt";
    if (argc > 1) {
        bal_file = argv[1];
    }

    // Load BAL dataset
    BALProblem bal_problem;
    if (!bal_problem.LoadFile(bal_file)) {
        cerr << "\nDownload a BAL dataset from:" << endl;
        cerr << "https://grail.cs.washington.edu/projects/bal/" << endl;
        return 1;
    }

    cout << "Loaded BAL problem:" << endl;
    cout << "  Cameras: " << bal_problem.num_cameras() << endl;
    cout << "  Points: " << bal_problem.num_points() << endl;
    cout << "  Observations: " << bal_problem.num_observations() << endl;

    // Build Ceres problem
    ceres::Problem problem;

    // Add residual blocks
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction* cost_function = BALReprojectionError::Create(
            bal_problem.observation_x(i),
            bal_problem.observation_y(i));

        // Huber loss for robustness
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

        problem.AddResidualBlock(
            cost_function,
            loss_function,
            bal_problem.mutable_camera(bal_problem.camera_index(i)),
            bal_problem.mutable_point(bal_problem.point_index(i)));
    }

    cout << "\nBuilt Ceres problem:" << endl;
    cout << "  Residual blocks: " << problem.NumResidualBlocks() << endl;
    cout << "  Parameter blocks: " << problem.NumParameterBlocks() << endl;

    // Solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.num_threads = 4;

    // Solve
    cout << "\nOptimizing..." << endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << "\n" << summary.FullReport() << endl;

    // Results
    cout << "\n=== Results ===" << endl;
    cout << "Initial cost: " << summary.initial_cost << endl;
    cout << "Final cost: " << summary.final_cost << endl;
    cout << "Improvement: " << (1.0 - summary.final_cost / summary.initial_cost) * 100 << "%" << endl;
    cout << "RMSE: " << sqrt(2.0 * summary.final_cost / bal_problem.num_observations())
         << " pixels" << endl;

    cout << "\n=== Complete ===" << endl;

    return 0;
}
