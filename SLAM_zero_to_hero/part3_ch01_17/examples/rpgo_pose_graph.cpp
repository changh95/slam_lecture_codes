/**
 * Kimera-RPGO Tutorial: 2D Pose-Graph Optimization (PGO)
 *
 * A robot drives a square loop. The odometry and the single loop closure
 * (x4 -> x0) are the *exact* relative transforms taken from the ground truth -
 * only the initial estimate is corrupted with noise - so the optimum is the
 * ground truth itself and the demo shows purely how the solver gets there.
 *
 * This is the shared 2D pose-graph exercise of part3 chapter 1: same poses,
 * same edges, same seed, same noise model and same chi-squared definition as
 * the g2o / GTSAM / Ceres / SymForce chapters, so the streamed results overlay
 * in one rerun recording and can be compared directly.
 *
 * There are no outliers here on purpose - Kimera-RPGO's outlier rejection gets
 * its own demo in rpgo_outlier_rejection. This example exists to be diffable
 * against the four sibling chapters.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <KimeraRPGO/RobustSolver.h>
#include <KimeraRPGO/SolverParams.h>

#include "rerun_viz.hpp"

using namespace std;
using Pose = part3viz::Pose2;  // [x, y, theta]
using gtsam::symbol_shorthand::X;

// Shared problem definition (identical in all chapters of this series).
static constexpr int kNumPoses = 5;
// Noise model sigma = (0.1, 0.1, 0.05) -> information diag(100, 100, 400).
static constexpr double kWeightXY = 10.0;     // 1 / 0.1
static constexpr double kWeightTheta = 20.0;  // 1 / 0.05

static double NormalizeAngle(double a) {
    const double two_pi = 2.0 * M_PI;
    return a - two_pi * std::floor((a + M_PI) / two_pi);
}

// Relative transform b expressed in a's local frame: a^{-1} * b.
static Pose Relative(const Pose& a, const Pose& b) {
    const double c = cos(a[2]), s = sin(a[2]);
    const double dx = b[0] - a[0], dy = b[1] - a[1];
    return {c * dx + s * dy, -s * dx + c * dy, NormalizeAngle(b[2] - a[2])};
}

// The one shared chi-squared formula: per edge, the tangent-space residual
// between the measured and the current relative pose (angle wrapped to
// (-pi, pi]), weighted by information diag(100, 100, 400), summed over edges.
// Computed here rather than read from the solver, so the number means exactly
// the same thing in every chapter of the series (GTSAM's own graph error, which
// Kimera-RPGO reports, carries a factor of 0.5 and includes the gauge prior).
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

static vector<Pose> Extract(const gtsam::Values& values) {
    vector<Pose> poses;
    poses.reserve(kNumPoses);
    for (int i = 0; i < kNumPoses; ++i) {
        const gtsam::Pose2 p = values.at<gtsam::Pose2>(X(i));
        poses.push_back({p.x(), p.y(), p.theta()});
    }
    return poses;
}

int main() {
    cout << "=== Kimera-RPGO Tutorial: 2D Pose-Graph Optimization ===\n" << endl;

    part3viz::Viz viz(part3viz::kPoseGraphRecording, "kimera_rpgo");

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
    // The three deviates are pulled into named locals on purpose. nxy and nth
    // are separate distribution objects that each cache a second value, so the
    // order in which they are sampled decides how the shared mt19937 stream is
    // consumed. Drawing them inside an argument list would leave that order
    // unspecified (GCC evaluates right to left) and silently give this chapter
    // a different problem from its siblings. Explicit statements pin it.
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

    cout << "Initial chi2 : " << Chi2(init, edges, measurements) << endl;
    viz.poseGraphSetup(gt, init, edges);
    viz.poseGraphIteration(0, init, Chi2(init, edges, measurements), edges);

    // ---------------------------------------------------------------- solver
    // PCM-Simple 2D thresholds are per-node drift budgets. There are no
    // outliers in this problem, so they are set loose enough to accept the one
    // loop closure; rpgo_outlier_rejection is where they earn their keep.
    KimeraRPGO::RobustSolverParams params;
    params.setPcmSimple2DParams(0.5,  // translation budget, m per node
                                0.1,  // rotation budget, rad per node
                                KimeraRPGO::Verbosity::UPDATE);
    KimeraRPGO::RobustSolver solver(params);

    // GTSAM's Pose2 tangent order is (x, y, theta), so the sigmas read in the
    // physical order here - unlike Pose3, whose order is (rotation,
    // translation). Kimera-RPGO has no equivalent of g2o's setFixed() or
    // Ceres's SetParameterBlockConstant: the gauge is removed with a tight
    // prior on pose 0, the same mechanism GTSAM uses.
    const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(3) << 0.01, 0.01, 0.005).finished());
    const auto edge_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(3) << 0.1, 0.1, 0.05).finished());

    gtsam::NonlinearFactorGraph odom_graph;
    gtsam::Values init_values;
    odom_graph.addPrior(X(0), gtsam::Pose2(gt[0][0], gt[0][1], gt[0][2]), prior_noise);
    for (int i = 0; i < kNumPoses; ++i) {
        init_values.insert(X(i), gtsam::Pose2(init[i][0], init[i][1], init[i][2]));
    }
    for (size_t e = 0; e < edges.size(); ++e) {
        if (edges[e].kind != part3viz::EdgeKind::Odometry) continue;
        odom_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
            X(edges[e].i), X(edges[e].j),
            gtsam::Pose2(measurements[e][0], measurements[e][1], measurements[e][2]),
            edge_noise);
    }

    gtsam::NonlinearFactorGraph loop_graph;
    for (size_t e = 0; e < edges.size(); ++e) {
        if (edges[e].kind != part3viz::EdgeKind::Loop) continue;
        loop_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(
            X(edges[e].i), X(edges[e].j),
            gtsam::Pose2(measurements[e][0], measurements[e][1], measurements[e][2]),
            edge_noise);
    }

    // Kimera-RPGO optimizes to convergence inside update() and exposes no
    // per-iteration hook, so the "iteration" timeline in the viewer carries the
    // solver's incremental stages instead of Levenberg-Marquardt steps:
    //   0 = initial estimate, 1 = after odometry, 2 = after the loop closure.
    solver.update(odom_graph, init_values);
    const vector<Pose> after_odom = Extract(solver.calculateEstimate());
    const double chi2_odom = Chi2(after_odom, edges, measurements);
    viz.poseGraphIteration(1, after_odom, chi2_odom, edges);
    cout << "chi2 after odometry stage : " << chi2_odom << endl;

    solver.update(loop_graph, gtsam::Values());
    const vector<Pose> poses = Extract(solver.calculateEstimate());
    const double chi2_final = Chi2(poses, edges, measurements);
    viz.poseGraphIteration(2, poses, chi2_final, edges);

    cout << "Loop closures seen : " << solver.getNumLC() << ", kept as inliers : "
         << solver.getNumLCInliers() << endl;
    cout << "Factors in the graph : " << solver.getFactorsUnsafe().size()
         << " (1 prior + 4 odometry + 1 loop closure)" << endl;

    cout << "\nPose | ground truth        | optimized           | error" << endl;
    cout << string(63, '-') << endl;
    for (int i = 0; i < kNumPoses; ++i) {
        const double err = hypot(poses[i][0] - gt[i][0], poses[i][1] - gt[i][1]);
        printf("  x%d | (%5.2f,%5.2f,%5.2f) | (%5.2f,%5.2f,%5.2f) | %.4f\n", i,
               gt[i][0], gt[i][1], gt[i][2], poses[i][0], poses[i][1], poses[i][2],
               err);
    }

    cout << "\nFinal chi2   : " << chi2_final << endl;
    cout << "Stages       : 3 (0 = initial, 1 = after odometry, 2 = after loop closure)"
         << endl;

    return 0;
}
