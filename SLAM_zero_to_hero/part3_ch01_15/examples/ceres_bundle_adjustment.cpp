/**
 * Ceres-Solver Tutorial: Bundle Adjustment with the BAL dataset
 *
 * Loads a "Bundle Adjustment in the Large" problem and optimizes the 6-DoF
 * camera poses and the 3D points. The per-camera intrinsics (f, k1, k2) are read
 * from the dataset and held FIXED - they are functor constants, not parameters -
 * so the problem is the same one the sibling chapters solve.
 *
 * There is no robust loss here. Ceres ships ceres::HuberLoss and friends, but
 * they are deliberately off in this series so the reported error IS the
 * objective being minimized and the four chapters' numbers compare directly.
 *
 * Every solver iteration is streamed to a rerun viewer (landmark cloud, camera
 * centres, and the reprojection-error graphs) via a ceres::IterationCallback.
 *
 * BAL camera model: angle-axis(3), translation(3), f, k1, k2.
 * Projection: P_cam = R*X + t;  p = (-P.x/P.z, -P.y/P.z);
 *             p' = f * (1 + k1*r^2 + k2*r^4) * p,  r^2 = |p|^2.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "rerun_viz.hpp"

using namespace std;

static constexpr int kMaxIterations = 30;

// Reprojection error of one observation. The camera parameter block is 6-DoF
// (angle-axis + translation); f, k1, k2 come from the dataset and are constants
// captured per observation from its camera.
struct BALReprojectionError {
    BALReprojectionError(double ox, double oy, double f, double k1, double k2)
        : ox_(ox), oy_(oy), f_(f), k1_(k1), k2_(k2) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* res) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        const T xp = -p[0] / p[2];
        const T yp = -p[1] / p[2];

        const T r2 = xp * xp + yp * yp;
        const T distortion = T(1.0) + T(k1_) * r2 + T(k2_) * r2 * r2;

        res[0] = T(f_) * distortion * xp - T(ox_);
        res[1] = T(f_) * distortion * yp - T(oy_);
        return true;
    }

    static ceres::CostFunction* Create(double ox, double oy, double f, double k1,
                                       double k2) {
        return new ceres::AutoDiffCostFunction<BALReprojectionError, 2, 6, 3>(
            new BALReprojectionError(ox, oy, f, k1, k2));
    }

private:
    double ox_, oy_, f_, k1_, k2_;
};

// Camera centre in world coords from a BAL camera: C = -R^T * t.
static void CameraCenter(const double* cam, double* C) {
    const double inv_aa[3] = {-cam[0], -cam[1], -cam[2]};
    const double t[3] = {cam[3], cam[4], cam[5]};
    double Rt_t[3];
    ceres::AngleAxisRotatePoint(inv_aa, t, Rt_t);
    C[0] = -Rt_t[0];
    C[1] = -Rt_t[1];
    C[2] = -Rt_t[2];
}

// The whole problem, laid out the way Ceres wants it.
struct BALProblem {
    int num_cameras = 0, num_points = 0, num_obs = 0;
    vector<int> cam_idx, pt_idx;
    vector<double> obs;         // 2 per observation
    vector<double> cameras;     // 6 per camera: angle-axis + translation (optimized)
    vector<double> intrinsics;  // 3 per camera: f, k1, k2 (FIXED)
    vector<double> points;      // 3 per point (optimized)
};

// Sum of squared reprojection error over all observations, in pixels^2.
// Computed from the parameters rather than taken from summary.final_cost, so the
// number is exactly the metric defined for this series and stays valid whatever
// the solver's internal scaling.
static double SqError(const BALProblem& p) {
    double total = 0.0;
    for (int i = 0; i < p.num_obs; ++i) {
        const double* cam = &p.cameras[6 * p.cam_idx[i]];
        const double* K = &p.intrinsics[3 * p.cam_idx[i]];
        const double* X = &p.points[3 * p.pt_idx[i]];
        double c[3];
        ceres::AngleAxisRotatePoint(cam, X, c);
        c[0] += cam[3];
        c[1] += cam[4];
        c[2] += cam[5];
        const double xp = -c[0] / c[2];
        const double yp = -c[1] / c[2];
        const double r2 = xp * xp + yp * yp;
        const double d = 1.0 + K[1] * r2 + K[2] * r2 * r2;
        const double ex = K[0] * d * xp - p.obs[2 * i];
        const double ey = K[0] * d * yp - p.obs[2 * i + 1];
        total += ex * ex + ey * ey;
    }
    return total;
}

// Per-observation RMS reprojection error in pixels (NOT per residual component).
static double RmsePx(double sq_error, int num_obs) {
    return sqrt(sq_error / num_obs);
}

// Streams the landmark cloud + camera centres after every accepted step. Ceres
// calls this once with iteration 0 for the initial state, then once per
// iteration.
struct StructureStreamer : public ceres::IterationCallback {
    StructureStreamer(part3viz::Viz& viz, const BALProblem& problem)
        : viz_(viz), problem_(problem) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override {
        const double sq = SqError(problem_);
        const double rmse = RmsePx(sq, problem_.num_obs);
        if (viz_.active()) {
            vector<part3viz::Vec3> pts(problem_.num_points);
            for (int i = 0; i < problem_.num_points; ++i) {
                pts[i] = {problem_.points[3 * i], problem_.points[3 * i + 1],
                          problem_.points[3 * i + 2]};
            }
            vector<part3viz::Vec3> centers(problem_.num_cameras);
            for (int i = 0; i < problem_.num_cameras; ++i) {
                double C[3];
                CameraCenter(&problem_.cameras[6 * i], C);
                centers[i] = {C[0], C[1], C[2]};
            }
            viz_.baIteration(s.iteration, pts, centers, sq, rmse);
        }
        steps_ = s.iteration;
        return ceres::SOLVER_CONTINUE;
    }

    part3viz::Viz& viz_;
    const BALProblem& problem_;
    int steps_ = 0;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    cout << "=== Ceres Tutorial: Bundle Adjustment (BAL) ===\n" << endl;

    const string bal_file = (argc > 1) ? argv[1] : "problem-21-11315-pre.txt";
    ifstream fin(bal_file);
    if (!fin) {
        cerr << "Error: cannot open " << bal_file << "\n"
             << "Download a BAL dataset from "
                "https://grail.cs.washington.edu/projects/bal/\n";
        return 1;
    }

    BALProblem p;
    fin >> p.num_cameras >> p.num_points >> p.num_obs;
    cout << "Cameras: " << p.num_cameras << "  Points: " << p.num_points
         << "  Observations: " << p.num_obs << endl;

    p.cam_idx.resize(p.num_obs);
    p.pt_idx.resize(p.num_obs);
    p.obs.resize(2 * p.num_obs);
    for (int i = 0; i < p.num_obs; ++i) {
        fin >> p.cam_idx[i] >> p.pt_idx[i] >> p.obs[2 * i] >> p.obs[2 * i + 1];
    }

    // BAL stores 9 numbers per camera; the first 6 are optimized, the last 3
    // (f, k1, k2) are split off into a separate array and never touched.
    p.cameras.resize(6 * p.num_cameras);
    p.intrinsics.resize(3 * p.num_cameras);
    for (int i = 0; i < p.num_cameras; ++i) {
        for (int k = 0; k < 6; ++k) fin >> p.cameras[6 * i + k];
        for (int k = 0; k < 3; ++k) fin >> p.intrinsics[3 * i + k];
    }
    p.points.resize(3 * p.num_points);
    for (double& v : p.points) fin >> v;
    fin.close();

    cout << "Intrinsics are FIXED (read from the dataset): camera 0 has "
         << "f=" << p.intrinsics[0] << " k1=" << p.intrinsics[1]
         << " k2=" << p.intrinsics[2] << endl;
    cout << "Optimizing " << 6 * p.num_cameras << " pose parameters (camera 0 fixed) + "
         << 3 * p.num_points << " point parameters" << endl;

    // Snapshot the starting cloud so the viewer can show where it came from.
    vector<part3viz::Vec3> initial_points(p.num_points);
    for (int i = 0; i < p.num_points; ++i) {
        initial_points[i] = {p.points[3 * i], p.points[3 * i + 1], p.points[3 * i + 2]};
    }

    part3viz::Viz viz(part3viz::kBundleAdjustmentRecording, "ceres");
    viz.baSetup(initial_points);

    ceres::Problem problem;
    for (int i = 0; i < p.num_obs; ++i) {
        const double* K = &p.intrinsics[3 * p.cam_idx[i]];
        problem.AddResidualBlock(
            BALReprojectionError::Create(p.obs[2 * i], p.obs[2 * i + 1], K[0], K[1],
                                         K[2]),
            /* no robust loss - see the file header */ nullptr,
            &p.cameras[6 * p.cam_idx[i]], &p.points[3 * p.pt_idx[i]]);
    }
    // Gauge: hold camera 0's pose. That removes 6 of the 7 gauge degrees of
    // freedom; overall scale stays free because the BAL projection is
    // scale-invariant, and LM's damping term takes care of that one remaining
    // direction. Fixing a point as well would over-constrain the problem.
    problem.SetParameterBlockConstant(&p.cameras[0]);

    const double sq0 = SqError(p);
    cout << "Initial sq_error: " << sq0 << " px^2   RMSE: " << RmsePx(sq0, p.num_obs)
         << " px" << endl;

    StructureStreamer streamer(viz, p);

    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = kMaxIterations;
    options.num_threads = 4;
    // With the scale gauge left free the normal equations stay singular in one
    // direction, so waiting for a tiny gradient never happens and the run would
    // end in NO_CONVERGENCE after burning the whole budget. The relative
    // function-decrease test is the criterion that actually means "done" here:
    // 1e-6, the same threshold the GTSAM chapter uses, ends the run in
    // CONVERGENCE inside the shared 30-iteration budget.
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-14;
    options.parameter_tolerance = 1e-10;
    options.update_state_every_iteration = true;  // refresh params before the callback
    options.callbacks.push_back(&streamer);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << "\n" << summary.BriefReport() << endl;
    cout << "Stopped because: " << summary.message << endl;

    const double sq1 = SqError(p);
    cout << "\nIterations : " << streamer.steps_ << " (frame 0 is the initial state)"
         << endl;
    cout << "sq_error   : " << sq0 << " -> " << sq1 << " px^2" << endl;
    cout << "RMSE       : " << RmsePx(sq0, p.num_obs) << " -> "
         << RmsePx(sq1, p.num_obs) << " px" << endl;
    cout << "Reduction  : " << 100.0 * (1.0 - sq1 / sq0) << " % of sq_error, "
         << 100.0 * (1.0 - RmsePx(sq1, p.num_obs) / RmsePx(sq0, p.num_obs))
         << " % of RMSE" << endl;

    return 0;
}
