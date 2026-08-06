/**
 * g2o Tutorial: 2D Pose-Graph Optimization (PGO)
 *
 * A robot drives a square loop and returns to its start. The graph uses g2o's
 * built-in VertexSE2 / EdgeSE2: 4 odometry edges plus one loop closure
 * (x4 -> x0). Only the *initial estimate* is corrupted with noise - the
 * measurements themselves are the exact relative transforms taken from ground
 * truth, which is deliberate: the optimum is then exactly the ground truth, so
 * any residual error is the solver's, not the data's.
 *
 * This is the shared 2D pose-graph exercise of part3 chapter 1 - same poses,
 * same edges, same noise model, same reported cost formula as the GTSAM /
 * Ceres / SymForce / Kimera-RPGO chapters. The C++ chapters also consume the
 * same mt19937(7) stream in the same order, so their initial estimates (and
 * therefore their initial chi2) are identical; SymForce draws from numpy and
 * gets a different realization of the same distribution.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <g2o/core/block_solver.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam2d/types_slam2d.h>

#include "rerun_viz.hpp"

namespace {

constexpr int kNumPoses = 5;
constexpr int kMaxIterations = 30;
constexpr unsigned kSeed = 7;
constexpr double kInitSigmaXy = 0.15;     // initial-estimate perturbation
constexpr double kInitSigmaTheta = 0.08;  // initial-estimate perturbation
// Measurement noise model, shared with the sibling chapters:
// sigma = (0.1, 0.1, 0.05) -> information = diag(100, 100, 400).
constexpr double kSigmaXy = 0.1;
constexpr double kSigmaTheta = 0.05;
constexpr double kGainThreshold = 1e-6;

double wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

/**
 * SE(2) residual of one edge, in the tangent space of the measurement:
 *   e = log( measured^-1 * (pose_i^-1 * pose_j) )
 * with the angle wrapped to (-pi, pi]. Written out by hand so the shared cost
 * formula is visible and identical across the chapters instead of being
 * whatever the library's internal error happens to be.
 */
std::array<double, 3> edgeResidual(const part3viz::Pose2& measured,
                                   const part3viz::Pose2& pi,
                                   const part3viz::Pose2& pj) {
    const double dx = pj[0] - pi[0];
    const double dy = pj[1] - pi[1];
    const double ci = std::cos(pi[2]), si = std::sin(pi[2]);
    // pose_i^-1 * pose_j
    const double rx = ci * dx + si * dy;
    const double ry = -si * dx + ci * dy;
    const double rt = wrapAngle(pj[2] - pi[2]);
    // measured^-1 * that
    const double cm = std::cos(measured[2]), sm = std::sin(measured[2]);
    const double ex = cm * (rx - measured[0]) + sm * (ry - measured[1]);
    const double ey = -sm * (rx - measured[0]) + cm * (ry - measured[1]);
    const double et = wrapAngle(rt - measured[2]);
    return {ex, ey, et};
}

/// The shared chi-squared: sum over edges of e^T * information * e.
double poseGraphChi2(const std::vector<part3viz::Pose2>& poses,
                     const std::vector<part3viz::Edge>& edges,
                     const std::vector<part3viz::Pose2>& measurements) {
    const double wxy = 1.0 / (kSigmaXy * kSigmaXy);         // 100
    const double wtheta = 1.0 / (kSigmaTheta * kSigmaTheta);  // 400
    double sum = 0.0;
    for (std::size_t k = 0; k < edges.size(); ++k) {
        const auto e =
            edgeResidual(measurements[k], poses[edges[k].i], poses[edges[k].j]);
        sum += wxy * e[0] * e[0] + wxy * e[1] * e[1] + wtheta * e[2] * e[2];
    }
    return sum;
}

part3viz::Pose2 toPose2(const g2o::SE2& p) {
    return {p.translation().x(), p.translation().y(), wrapAngle(p.rotation().angle())};
}

/**
 * Post-iteration hook: streams the current graph and stops the optimizer once
 * the relative chi2 decrease drops below kGainThreshold. (g2o's own
 * SparseOptimizerTerminateAction does the same job; the test is written out
 * here so it matches the sibling chapters exactly.)
 */
struct IterationRecorder : public g2o::HyperGraphAction {
    IterationRecorder(g2o::SparseOptimizer* opt,
                      const std::vector<part3viz::Edge>& edges,
                      const std::vector<part3viz::Pose2>& measurements,
                      part3viz::Viz* viz, bool* stop)
        : opt_(opt), edges_(edges), measurements_(measurements), viz_(viz), stop_(stop) {}

    double record(int iter) {
        std::vector<part3viz::Pose2> poses(kNumPoses);
        for (int i = 0; i < kNumPoses; ++i) {
            poses[i] = toPose2(static_cast<g2o::VertexSE2*>(opt_->vertex(i))->estimate());
        }
        const double chi2 = poseGraphChi2(poses, edges_, measurements_);
        viz_->poseGraphIteration(iter, poses, chi2, edges_);
        history.push_back({poses, chi2});
        return chi2;
    }

    g2o::HyperGraphAction* operator()(const g2o::HyperGraph*,
                                      Parameters* params) override {
        // Frame 0 is the initial state, so g2o's iteration i is frame i + 1.
        // g2o also fires the post-iteration actions once with iteration = -1
        // (the "about to start" notification) - that is not an iteration.
        int iter = static_cast<int>(history.size());
        if (const auto* p = dynamic_cast<const ParametersIteration*>(params)) {
            if (p->iteration < 0) return this;
            iter = p->iteration + 1;
        }
        const double prev =
            history.empty() ? std::numeric_limits<double>::max() : history.back().chi2;
        const double chi2 = record(iter);
        const double gain = (prev - chi2) / std::max(chi2, 1e-12);
        if (gain >= 0.0 && gain < kGainThreshold) *stop_ = true;
        return this;
    }

    struct Frame {
        std::vector<part3viz::Pose2> poses;
        double chi2;
    };
    std::vector<Frame> history;

private:
    g2o::SparseOptimizer* opt_;
    const std::vector<part3viz::Edge>& edges_;
    const std::vector<part3viz::Pose2>& measurements_;
    part3viz::Viz* viz_;
    bool* stop_;
};

}  // namespace

int main() {
    std::cout << "=== g2o Tutorial: 2D Pose-Graph Optimization ===\n" << std::endl;

    // Ground-truth square trajectory: back to the start, facing a new heading.
    const std::array<g2o::SE2, kNumPoses> gt = {g2o::SE2(0, 0, 0), g2o::SE2(1, 0, 0),
                                                g2o::SE2(1, 1, M_PI / 2),
                                                g2o::SE2(0, 1, M_PI),
                                                g2o::SE2(0, 0, -M_PI / 2)};

    // 4 odometry edges + 1 loop closure.
    const std::vector<part3viz::Edge> edges = {
        {0, 1, part3viz::EdgeKind::Odometry}, {1, 2, part3viz::EdgeKind::Odometry},
        {2, 3, part3viz::EdgeKind::Odometry}, {3, 4, part3viz::EdgeKind::Odometry},
        {4, 0, part3viz::EdgeKind::Loop}};

    // Measurements are the exact relative transforms from ground truth.
    std::vector<part3viz::Pose2> measurements;
    measurements.reserve(edges.size());
    for (const auto& e : edges) {
        measurements.push_back(toPose2(gt[e.i].inverse() * gt[e.j]));
    }

    // Solver: variable block sizes, dense, Levenberg-Marquardt.
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Noisy initial estimate. Pose 0 is the gauge anchor: it stays exactly at
    // ground truth and gets no noise draw, so the perturbations of poses 1..4
    // are the same numbers the other C++ chapters (GTSAM, Ceres, Kimera-RPGO)
    // use. The three deviates are drawn into named locals in a fixed order:
    // inside a constructor's argument list C++ leaves the evaluation order
    // unspecified, which would consume this shared mt19937 stream differently
    // from the sibling chapters and silently pose a different problem.
    std::mt19937 rng(kSeed);
    std::normal_distribution<double> nxy(0.0, kInitSigmaXy), nth(0.0, kInitSigmaTheta);
    std::vector<part3viz::Pose2> init(kNumPoses), gt_viz(kNumPoses);
    for (int i = 0; i < kNumPoses; ++i) {
        gt_viz[i] = toPose2(gt[i]);
        auto* v = new g2o::VertexSE2();
        v->setId(i);
        if (i == 0) {
            v->setEstimate(gt[0]);
            // g2o's idiomatic gauge fix: hard-fix the vertex. It is removed
            // from the linear system entirely, unlike GTSAM's tight prior or
            // Ceres's SetParameterBlockConstant.
            v->setFixed(true);
            init[0] = gt_viz[0];
        } else {
            const double dx = nxy(rng);
            const double dy = nxy(rng);
            const double dth = nth(rng);
            const g2o::SE2 noisy(gt[i].translation().x() + dx,
                                 gt[i].translation().y() + dy,
                                 gt[i].rotation().angle() + dth);
            v->setEstimate(noisy);
            init[i] = toPose2(noisy);
        }
        optimizer.addVertex(v);
    }

    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    information(0, 0) = 1.0 / (kSigmaXy * kSigmaXy);
    information(1, 1) = 1.0 / (kSigmaXy * kSigmaXy);
    information(2, 2) = 1.0 / (kSigmaTheta * kSigmaTheta);
    for (const auto& e : edges) {
        auto* edge = new g2o::EdgeSE2();
        edge->setVertex(0, optimizer.vertex(e.i));
        edge->setVertex(1, optimizer.vertex(e.j));
        edge->setMeasurement(gt[e.i].inverse() * gt[e.j]);  // exact measurement
        edge->setInformation(information);
        optimizer.addEdge(edge);
    }

    std::cout << "Poses: " << kNumPoses << "  edges: " << edges.size()
              << " (4 odometry + 1 loop closure)" << std::endl;
    std::cout << "Measurement sigma: (" << kSigmaXy << ", " << kSigmaXy << ", "
              << kSigmaTheta << ") -> information diag(" << information(0, 0) << ", "
              << information(1, 1) << ", " << information(2, 2) << ")" << std::endl;
    std::cout << "Initial estimate: poses 1..4 perturbed with sigma_xy="
              << kInitSigmaXy << ", sigma_theta=" << kInitSigmaTheta << " (seed "
              << kSeed << "); pose 0 fixed at ground truth\n"
              << std::endl;

    part3viz::Viz viz(part3viz::kPoseGraphRecording, "g2o");
    viz.poseGraphSetup(gt_viz, init, edges);

    bool stop = false;
    IterationRecorder recorder(&optimizer, edges, measurements, &viz, &stop);
    optimizer.setForceStopFlag(&stop);
    optimizer.addPostIterationAction(&recorder);

    optimizer.initializeOptimization();
    recorder.record(0);  // frame 0: the noisy initial estimate
    optimizer.optimize(kMaxIterations);
    optimizer.setForceStopFlag(nullptr);

    const int iterations = static_cast<int>(recorder.history.size()) - 1;
    const double chi2_initial = recorder.history.front().chi2;
    const double chi2_final = recorder.history.back().chi2;
    const auto& opt = recorder.history.back().poses;

    std::cout << "\nPose | ground truth        | initial             | optimized"
                 "           | error"
              << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    double max_err = 0.0;
    for (int i = 0; i < kNumPoses; ++i) {
        const double err = std::hypot(opt[i][0] - gt_viz[i][0], opt[i][1] - gt_viz[i][1]);
        max_err = std::max(max_err, err);
        std::printf("  x%d | (%5.2f,%5.2f,%5.2f) | (%5.2f,%5.2f,%5.2f) | "
                    "(%5.2f,%5.2f,%5.2f) | %.2e\n",
                    i, gt_viz[i][0], gt_viz[i][1], gt_viz[i][2], init[i][0], init[i][1],
                    init[i][2], opt[i][0], opt[i][1], opt[i][2], err);
    }

    std::printf("\nchi2       : %.6f -> %.3e\n", chi2_initial, chi2_final);
    std::printf("Max position error vs ground truth: %.3e m\n", max_err);
    std::printf("Iterations : %d of %d%s\n", iterations, kMaxIterations,
                iterations < kMaxIterations ? " (converged early)" : "");

    return 0;
}
