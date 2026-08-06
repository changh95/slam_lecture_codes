/**
 * GTSAM: 2D pose-graph optimization
 *
 * A robot drives a unit square and returns to its start. The factor graph holds
 * a tight PriorFactor on x0 (GTSAM's idiomatic gauge anchor - it has no
 * "fix this variable" flag), BetweenFactor<Pose2> odometry for (0,1) (1,2)
 * (2,3) (3,4), and one loop closure (4,0). LM runs one step at a time so every
 * iteration streams to a live rerun viewer.
 *
 * Shared exercise setup (identical in the g2o / Ceres / SymForce chapters):
 *   ground truth (0,0,0) (1,0,0) (1,1,pi/2) (0,1,pi) (0,0,-pi/2),
 *   measurements are the EXACT relative transforms from ground truth (so the
 *   optimum is ground truth and any residual error is the solver's),
 *   measurement sigma = (0.1, 0.1, 0.05) -> information diag(100, 100, 400),
 *   initial estimate = ground truth perturbed with seed 7,
 *   sigma_xy = 0.15, sigma_theta = 0.08, on poses 1..4 only.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include "rerun_viz.hpp"

using namespace std;
using namespace gtsam;

// GTSAM's VerbosityLM output rewrites cout's precision while it prints, so this
// puts it back before each of our own lines.
static ostream& msg() { return cout << setprecision(6); }

namespace {

constexpr int kN = 5;

double wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

part3viz::Pose2 toViz(const gtsam::Pose2& p) { return {p.x(), p.y(), p.theta()}; }

}  // namespace

int main() {
    msg() << "=== GTSAM: 2D pose-graph optimization ===\n" << endl;

    part3viz::Viz viz(part3viz::kPoseGraphRecording, "gtsam");

    array<gtsam::Pose2, kN> gt = {gtsam::Pose2(0, 0, 0), gtsam::Pose2(1, 0, 0),
                                 gtsam::Pose2(1, 1, M_PI / 2),
                                 gtsam::Pose2(0, 1, M_PI),
                                 gtsam::Pose2(0, 0, -M_PI / 2)};

    const vector<part3viz::Edge> edges = {
        {0, 1, part3viz::EdgeKind::Odometry}, {1, 2, part3viz::EdgeKind::Odometry},
        {2, 3, part3viz::EdgeKind::Odometry}, {3, 4, part3viz::EdgeKind::Odometry},
        {4, 0, part3viz::EdgeKind::Loop}};

    NonlinearFactorGraph graph;
    // Gauge anchor: GTSAM fixes nothing, so pose 0 is pinned with a prior two
    // orders of magnitude tighter than the measurements.
    auto prior_noise = noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.005));
    auto odo_noise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));

    graph.addPrior(Symbol('x', 0), gt[0], prior_noise);
    for (const auto& e : edges) {
        graph.add(BetweenFactor<gtsam::Pose2>(Symbol('x', e.i), Symbol('x', e.j),
                                              gt[e.i].between(gt[e.j]), odo_noise));
    }

    // Noisy initial estimate. Pose 0 stays exactly at ground truth: it is the
    // anchor, so perturbing it would only shift the whole gauge.
    mt19937 rng(7);
    normal_distribution<double> nxy(0.0, 0.15), nth(0.0, 0.08);
    array<gtsam::Pose2, kN> init;
    Values initial;
    for (int i = 0; i < kN; ++i) {
        if (i == 0) {
            init[0] = gt[0];
        } else {
            // Named locals, drawn in this order on purpose. Inside a constructor
            // call - Pose2(gt.x() + nxy(rng), gt.y() + nxy(rng), ...) - argument
            // evaluation order is unspecified and GCC goes right-to-left, so the
            // perturbation would differ from the other chapters even with the
            // same seed.
            const double dx = nxy(rng);
            const double dy = nxy(rng);
            const double dth = nth(rng);
            init[i] = gtsam::Pose2(gt[i].x() + dx, gt[i].y() + dy,
                                   gt[i].theta() + dth);
        }
        initial.insert(Symbol('x', i), init[i]);
    }

    // Chi-squared over the edges only, with the same formula in every chapter:
    // delta = measured^-1 * (T_i^-1 T_j), residual = (dx, dy, wrap(dtheta)),
    // weighted by information diag(100, 100, 400). GTSAM's graph.error() cannot
    // be used for the comparison plot - it is 0.5 * chi2 and it also includes
    // the anchor prior, which the other chapters express as a hard constraint.
    const auto chi2 = [&](const array<gtsam::Pose2, kN>& poses) {
        const double wxy = 1.0 / (0.1 * 0.1), wth = 1.0 / (0.05 * 0.05);
        double sum = 0.0;
        for (const auto& e : edges) {
            const gtsam::Pose2 measured = gt[e.i].between(gt[e.j]);
            const gtsam::Pose2 predicted = poses[e.i].between(poses[e.j]);
            const gtsam::Pose2 delta = measured.inverse() * predicted;
            const double dth = wrapAngle(delta.theta());
            sum += wxy * (delta.x() * delta.x() + delta.y() * delta.y()) +
                   wth * dth * dth;
        }
        return sum;
    };

    const auto current = [&](const Values& v) {
        array<gtsam::Pose2, kN> poses;
        for (int i = 0; i < kN; ++i) poses[i] = v.at<gtsam::Pose2>(Symbol('x', i));
        return poses;
    };
    const auto vizPoses = [](const array<gtsam::Pose2, kN>& poses) {
        vector<part3viz::Pose2> out;
        out.reserve(kN);
        for (const auto& p : poses) out.push_back(toViz(p));
        return out;
    };

    viz.poseGraphSetup(vizPoses(gt), vizPoses(init), edges);

    LevenbergMarquardtParams params;
    // "SUMMARY" is a VerbosityLM value; setVerbosity() takes the other enum and
    // would silently fall through to SILENT.
    params.setVerbosityLM("SUMMARY");
    LevenbergMarquardtOptimizer optimizer(graph, initial, params);

    const int kMaxIterations = 30;
    double cost = chi2(init);
    msg() << "Iteration 0: chi2 = " << cost << endl;
    viz.poseGraphIteration(0, vizPoses(init), cost, edges);

    int iterations = 0;
    for (int it = 1; it <= kMaxIterations; ++it) {
        optimizer.iterate();
        const auto poses = current(optimizer.values());
        const double next = chi2(poses);
        ++iterations;
        viz.poseGraphIteration(it, vizPoses(poses), next, edges);
        msg() << "Iteration " << it << ": chi2 = " << next << endl;
        const bool converged = (cost - next) <= 1e-6 * max(1.0, cost);
        cost = next;
        if (converged) break;
    }

    const auto opt = current(optimizer.values());
    msg() << "\nchi2: " << chi2(init) << " -> " << cost << "  (" << iterations
         << " LM iterations)" << endl;

    msg() << "\nPose | ground truth        | optimized           | position error"
         << endl;
    msg() << string(68, '-') << endl;
    double max_err = 0.0;
    for (int i = 0; i < kN; ++i) {
        const double err = hypot(opt[i].x() - gt[i].x(), opt[i].y() - gt[i].y());
        max_err = max(max_err, err);
        printf("  x%d | (%5.2f,%5.2f,%5.2f) | (%5.2f,%5.2f,%5.2f) | %.6f\n", i,
               gt[i].x(), gt[i].y(), gt[i].theta(), opt[i].x(), opt[i].y(),
               opt[i].theta(), err);
    }
    msg() << "\nMax position error: " << max_err << " m" << endl;

    return 0;
}
