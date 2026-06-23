/**
 * Ceres-Solver Tutorial: Bundle Adjustment with the BAL dataset
 *
 * Loads a "Bundle Adjustment in the Large" problem, jointly optimizes camera
 * poses + intrinsics + 3D points with a Huber-robust reprojection cost, and
 * captures the landmark cloud + camera centers at *every* solver iteration (via
 * a ceres::IterationCallback). Dumps the per-iteration structure to
 * `bundle_adjustment.txt` so viz/show_bundle_adjustment.py can animate the
 * optimization on a rerun timeline.
 *
 * BAL camera model (9 params): angle-axis(3), translation(3), f, k1, k2.
 * Projection: p = -P/P.z;  p' = f * (1 + k1*r^2 + k2*r^4) * p.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

struct BALReprojectionError {
    BALReprojectionError(double ox, double oy) : ox_(ox), oy_(oy) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* res) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + camera[7] * r2 + camera[8] * r2 * r2;

        res[0] = camera[6] * distortion * xp - ox_;
        res[1] = camera[6] * distortion * yp - oy_;
        return true;
    }

    static ceres::CostFunction* Create(double ox, double oy) {
        return new ceres::AutoDiffCostFunction<BALReprojectionError, 2, 9, 3>(
            new BALReprojectionError(ox, oy));
    }

private:
    double ox_, oy_;
};

// Camera centre in world coords from a BAL camera: C = -R^T * t.
static void CameraCenter(const double* cam, double* C) {
    double inv_aa[3] = {-cam[0], -cam[1], -cam[2]};
    double t[3] = {cam[3], cam[4], cam[5]};
    double Rt_t[3];
    ceres::AngleAxisRotatePoint(inv_aa, t, Rt_t);
    C[0] = -Rt_t[0];
    C[1] = -Rt_t[1];
    C[2] = -Rt_t[2];
}

// Snapshots the current landmarks + camera centres after every iteration.
struct StructureRecorder : public ceres::IterationCallback {
    StructureRecorder(const vector<double>& cameras, const vector<double>& points,
                      int num_cameras, int num_points)
        : cameras_(cameras), points_(points),
          num_cameras_(num_cameras), num_points_(num_points) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override {
        point_frames_.push_back(points_);  // copy current landmark positions
        vector<double> centers(3 * num_cameras_);
        for (int i = 0; i < num_cameras_; ++i)
            CameraCenter(&cameras_[9 * i], &centers[3 * i]);
        camera_frames_.push_back(std::move(centers));
        errors_.push_back(2.0 * s.cost);  // total squared reprojection error
        return ceres::SOLVER_CONTINUE;
    }

    const vector<double>&cameras_, &points_;
    int num_cameras_, num_points_;
    vector<vector<double>> point_frames_, camera_frames_;
    vector<double> errors_;
};

int main(int argc, char** argv) {
    cout << "=== Ceres Tutorial: Bundle Adjustment (BAL) ===\n" << endl;

    string bal_file = (argc > 1) ? argv[1] : "problem-21-11315-pre.txt";
    ifstream fin(bal_file);
    if (!fin) {
        cerr << "Error: cannot open " << bal_file << "\n"
             << "Download a BAL dataset from "
                "https://grail.cs.washington.edu/projects/bal/\n";
        return 1;
    }

    int num_cameras, num_points, num_obs;
    fin >> num_cameras >> num_points >> num_obs;
    cout << "Cameras: " << num_cameras << "  Points: " << num_points
         << "  Observations: " << num_obs << endl;

    vector<int> cam_idx(num_obs), pt_idx(num_obs);
    vector<double> obs(2 * num_obs);
    for (int i = 0; i < num_obs; ++i)
        fin >> cam_idx[i] >> pt_idx[i] >> obs[2 * i] >> obs[2 * i + 1];

    vector<double> cameras(9 * num_cameras), points(3 * num_points);
    for (double& v : cameras) fin >> v;
    for (double& v : points) fin >> v;
    fin.close();

    ceres::Problem problem;
    for (int i = 0; i < num_obs; ++i) {
        problem.AddResidualBlock(
            BALReprojectionError::Create(obs[2 * i], obs[2 * i + 1]),
            new ceres::HuberLoss(1.0),
            &cameras[9 * cam_idx[i]], &points[3 * pt_idx[i]]);
    }

    // Record the structure at every iteration (iteration 0 = initial state).
    StructureRecorder recorder(cameras, points, num_cameras, num_points);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 30;
    options.num_threads = 4;
    options.update_state_every_iteration = true;  // refresh params before callback
    options.callbacks.push_back(&recorder);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << "\n" << summary.BriefReport() << endl;
    cout << "Initial cost: " << summary.initial_cost
         << "  Final cost: " << summary.final_cost << endl;
    cout << "RMSE: " << sqrt(2.0 * summary.final_cost / num_obs) << " px" << endl;

    const auto& pf = recorder.point_frames_;
    const auto& cf = recorder.camera_frames_;
    const int K = static_cast<int>(pf.size());

    ofstream out("bundle_adjustment.txt");
    out << "points " << num_points << "\n";
    out << "cameras " << num_cameras << "\n";
    out << "steps " << K << "\n";
    for (int k = 0; k < K; ++k) {
        out << "step " << k << " " << recorder.errors_[k] << "\n";
        for (int i = 0; i < num_points; ++i)
            out << pf[k][3 * i] << " " << pf[k][3 * i + 1] << " " << pf[k][3 * i + 2] << "\n";
        for (int i = 0; i < num_cameras; ++i)
            out << cf[k][3 * i] << " " << cf[k][3 * i + 1] << " " << cf[k][3 * i + 2] << "\n";
    }
    out.close();
    cout << "\nWrote bundle_adjustment.txt (" << K << " iterations) -> visualize with "
            "viz/show_bundle_adjustment.py" << endl;

    return 0;
}
