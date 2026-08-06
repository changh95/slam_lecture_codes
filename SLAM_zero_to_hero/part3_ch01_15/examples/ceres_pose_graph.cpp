/**
 * Ceres-Solver Tutorial: 2D Pose-Graph Optimization (PGO)
 *
 * A robot drives a square loop. The odometry and the single loop closure
 * (x4 -> x0) are the *exact* relative transforms taken from the ground truth -
 * only the initial estimate is corrupted with noise - so the optimum is the
 * ground truth itself and the demo shows purely how the solver gets there.
 * Every iteration is streamed to a rerun viewer.
 *
 * This is the shared 2D pose-graph exercise of part3 chapter 1: same poses,
 * same edges, same seed, same noise model and same chi-squared definition as
 * the g2o / GTSAM / SymForce / Kimera-RPGO chapters.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include "rerun_viz.hpp"

using namespace std;
using Pose = part3viz::Pose2;  // [x, y, theta]

// Shared problem definition (identical in all chapters of this series).
static constexpr int kNumPoses = 5;
static constexpr int kMaxIterations = 30;
// Noise model sigma = (0.1, 0.1, 0.05) -> information diag(100, 100, 400).
// Ceres has no explicit information matrix: the residual is pre-multiplied by
// the square root of the information, i.e. by 1/sigma.
static constexpr double kWeightXY = 10.0;   // 1 / 0.1
static constexpr double kWeightTheta = 20.0;  // 1 / 0.05

template <typename T>
T NormalizeAngle(const T& a) {
    const T two_pi = T(2.0 * M_PI);
    return a - two_pi * floor((a + T(M_PI)) / two_pi);
}

// Relative transform b expressed in a's local frame: a^{-1} * b.
static Pose Relative(const Pose& a, const Pose& b) {
    const double c = cos(a[2]), s = sin(a[2]);
    const double dx = b[0] - a[0], dy = b[1] - a[1];
    return {c * dx + s * dy, -s * dx + c * dy, NormalizeAngle(b[2] - a[2])};
}

// Odometry / loop constraint between two 2D poses.
struct RelativeMotion {
    RelativeMotion(double dx, double dy, double dth) : dx_(dx), dy_(dy), dth_(dth) {}

    template <typename T>
    bool operator()(const T* const pi, const T* const pj, T* r) const {
        const T c = cos(pi[2]), s = sin(pi[2]);
        const T dx = pj[0] - pi[0], dy = pj[1] - pi[1];
        r[0] = T(kWeightXY) * (c * dx + s * dy - T(dx_));
        r[1] = T(kWeightXY) * (-s * dx + c * dy - T(dy_));
        r[2] = T(kWeightTheta) *
               NormalizeAngle(NormalizeAngle(pj[2] - pi[2]) - T(dth_));
        return true;
    }

    static ceres::CostFunction* Create(double dx, double dy, double dth) {
        return new ceres::AutoDiffCostFunction<RelativeMotion, 3, 3, 3>(
            new RelativeMotion(dx, dy, dth));
    }

private:
    const double dx_, dy_, dth_;
};

// The one shared chi-squared formula: per edge, the tangent-space residual
// between the measured and the current relative pose (angle wrapped to
// (-pi, pi]), weighted by information diag(100, 100, 400), summed over edges.
// Computed here rather than read from the solver so the number means exactly
// the same thing in every chapter of the series.
static double Chi2(const vector<Pose>& poses, const vector<part3viz::Edge>& edges,
                   const vector<Pose>& measurements) {
    double chi2 = 0.0;
    for (size_t e = 0; e < edges.size(); ++e) {
        const Pose cur = Relative(poses[edges[e].i], poses[edges[e].j]);
        const double rx = cur[0] - measurements[e][0];
        const double ry = cur[1] - measurements[e][1];
        const double rt = NormalizeAngle(cur[2] - measurements[e][2]);
        chi2 += kWeightXY * kWeightXY * (rx * rx + ry * ry) +
                kWeightTheta * kWeightTheta * rt * rt;
    }
    return chi2;
}

// Streams the trajectory after every accepted step. Ceres calls this once with
// iteration 0 for the initial state, then once per iteration.
struct IterationStreamer : public ceres::IterationCallback {
    IterationStreamer(part3viz::Viz& viz, const vector<Pose>& poses,
                      const vector<part3viz::Edge>& edges,
                      const vector<Pose>& measurements)
        : viz_(viz), poses_(poses), edges_(edges), measurements_(measurements) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override {
        const double chi2 = Chi2(poses_, edges_, measurements_);
        viz_.poseGraphIteration(s.iteration, poses_, chi2, edges_);
        steps_ = s.iteration;
        return ceres::SOLVER_CONTINUE;
    }

    part3viz::Viz& viz_;
    const vector<Pose>& poses_;
    const vector<part3viz::Edge>& edges_;
    const vector<Pose>& measurements_;
    int steps_ = 0;
};

int main(int /*argc*/, char** argv) {
    google::InitGoogleLogging(argv[0]);
    cout << "=== Ceres Tutorial: 2D Pose-Graph Optimization ===\n" << endl;

    part3viz::Viz viz(part3viz::kPoseGraphRecording, "ceres");

    // Ground-truth square trajectory.
    vector<Pose> gt = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, M_PI / 2},
        {0.0, 1.0, M_PI},
        {0.0, 0.0, -M_PI / 2},
    };

    // 4 odometry edges + 1 loop closure, measurements derived from ground truth.
    vector<part3viz::Edge> edges = {
        {0, 1, part3viz::EdgeKind::Odometry},
        {1, 2, part3viz::EdgeKind::Odometry},
        {2, 3, part3viz::EdgeKind::Odometry},
        {3, 4, part3viz::EdgeKind::Odometry},
        {4, 0, part3viz::EdgeKind::Loop},
    };
    vector<Pose> measurements;
    measurements.reserve(edges.size());
    for (const auto& e : edges) measurements.push_back(Relative(gt[e.i], gt[e.j]));

    // Noisy initial estimate, seed 7. Pose 0 is the gauge anchor: it is held at
    // ground truth by the solver, so it gets no noise draw either - the draws
    // are made for poses 1..4 only, in the same order as the sibling chapters.
    //
    // The three deviates are drawn into named locals, one statement each. That
    // ordering is the shared convention of this chapter group and it is load
    // bearing: nxy and nth each cache a spare value internally, so the order in
    // which they consume the single mt19937(7) stream decides the perturbation.
    // Drawing them inside a constructor's argument list instead would leave the
    // order unspecified (GCC evaluates arguments right to left) and silently
    // produce a different problem from the sibling chapters.
    mt19937 rng(7);
    normal_distribution<double> nxy(0.0, 0.15), nth(0.0, 0.08);
    vector<Pose> init(kNumPoses);
    init[0] = gt[0];
    for (int i = 1; i < kNumPoses; ++i) {
        const double dx = nxy(rng);
        const double dy = nxy(rng);
        const double dth = nth(rng);
        init[i] = {gt[i][0] + dx, gt[i][1] + dy, gt[i][2] + dth};
    }
    vector<Pose> poses = init;

    cout << "Initial chi2 : " << Chi2(init, edges, measurements) << endl;
    viz.poseGraphSetup(gt, init, edges);

    ceres::Problem problem;
    for (size_t e = 0; e < edges.size(); ++e) {
        problem.AddResidualBlock(
            RelativeMotion::Create(measurements[e][0], measurements[e][1],
                                   measurements[e][2]),
            nullptr, poses[edges[e].i].data(), poses[edges[e].j].data());
    }
    // Gauge freedom: a pose graph with only relative constraints is invariant
    // under a global rigid transform. Ceres removes it by declaring the first
    // pose's parameter block constant - the equivalent of g2o's setFixed(true)
    // and cheaper than GTSAM's tight prior, because the block leaves the
    // linear system entirely.
    problem.SetParameterBlockConstant(poses[0].data());

    IterationStreamer streamer(viz, poses, edges, measurements);

    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = kMaxIterations;
    options.update_state_every_iteration = true;  // refresh poses before the callback
    options.callbacks.push_back(&streamer);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << "\n" << summary.BriefReport() << endl;
    cout << "Stopped because: " << summary.message << endl;

    cout << "\nPose | ground truth        | optimized           | error" << endl;
    cout << string(63, '-') << endl;
    for (int i = 0; i < kNumPoses; ++i) {
        const double err = hypot(poses[i][0] - gt[i][0], poses[i][1] - gt[i][1]);
        printf("  x%d | (%5.2f,%5.2f,%5.2f) | (%5.2f,%5.2f,%5.2f) | %.4f\n", i,
               gt[i][0], gt[i][1], gt[i][2], poses[i][0], poses[i][1], poses[i][2],
               err);
    }

    cout << "\nFinal chi2   : " << Chi2(poses, edges, measurements) << endl;
    cout << "Iterations   : " << streamer.steps_ << " (frame 0 is the initial state)"
         << endl;

    return 0;
}
