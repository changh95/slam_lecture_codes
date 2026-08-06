/**
 * @file rpgo_basics.cpp
 * @brief Kimera-RPGO basics: build a 3D pose graph and optimize it robustly.
 *
 * What this demo shows:
 *   1. Configuring a RobustSolver (PCM-Simple 3D outlier rejection).
 *   2. Adding a prior, odometry factors and one loop closure.
 *   3. That the optimization actually does something: the odometry is
 *      corrupted with noise (fixed seed 7) so the square does not close, the
 *      loop closure is exact, and the graph error before/after plus a
 *      ground-truth / initial / optimized table make the correction visible.
 *
 * Kimera-RPGO optimizes inside update() - there is no per-iteration callback -
 * so the viewer shows the initial guess and the converged result, not an
 * iteration sweep.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <KimeraRPGO/RobustSolver.h>
#include <KimeraRPGO/SolverParams.h>

#include "rerun_viz.hpp"

using gtsam::symbol_shorthand::X;  // pose keys X(0), X(1), ...

namespace {

/// Number of poses on the square: 4 sides of 1 m, one turn at every corner.
constexpr int kNumPoses = 8;

/// Fixed seed so every run of the demo produces the same numbers.
constexpr unsigned kSeed = 7;

/// Ground-truth odometry of the closed square: drive 1 m, then turn 90 deg at
/// every second step, so pose 7 sits back on pose 0 rotated by -90 deg.
std::vector<gtsam::Pose3> groundTruthOdometry() {
    const gtsam::Pose3 forward(gtsam::Rot3::Identity(), gtsam::Point3(1.0, 0.0, 0.0));
    const gtsam::Pose3 turn(gtsam::Rot3::Rz(M_PI / 2), gtsam::Point3(0.0, 0.0, 0.0));
    return {forward, turn, forward, turn, forward, turn, forward};
}

/// Chain relative measurements into absolute poses starting from the origin.
std::vector<gtsam::Pose3> chain(const std::vector<gtsam::Pose3>& relative) {
    std::vector<gtsam::Pose3> poses{gtsam::Pose3::Identity()};
    for (const auto& r : relative) {
        poses.push_back(poses.back() * r);
    }
    return poses;
}

/// Project a planar SE(3) pose to the (x, y, yaw) the 2D viewer takes.
///
/// Both demos in this chapter build graphs that lie entirely in the z = 0
/// plane, so this projection loses nothing. It matters because a rerun 3D view
/// will not frame a graph whose every pose sits at z = 0 - the bounding box is
/// degenerate, so the camera ends up staring past the trajectory - while the 2D
/// view draws the path, the per-pose heading and the loop closures legibly.
std::vector<part3viz::Pose2> planar(const std::vector<gtsam::Pose3>& poses) {
    std::vector<part3viz::Pose2> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back({p.translation().x(), p.translation().y(), p.rotation().yaw()});
    }
    return out;
}

std::vector<part3viz::Pose2> planar(const gtsam::Values& values, int n) {
    std::vector<part3viz::Pose2> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const gtsam::Pose3 p = values.at<gtsam::Pose3>(X(i));
        out.push_back({p.translation().x(), p.translation().y(), p.rotation().yaw()});
    }
    return out;
}

/// RMS translation error against ground truth, over all poses.
double translationRmse(const std::vector<gtsam::Pose3>& gt, const gtsam::Values& values) {
    double sum = 0.0;
    for (std::size_t i = 0; i < gt.size(); ++i) {
        const gtsam::Pose3 p = values.at<gtsam::Pose3>(X(static_cast<int>(i)));
        sum += (p.translation() - gt[i].translation()).squaredNorm();
    }
    return std::sqrt(sum / static_cast<double>(gt.size()));
}

}  // namespace

int main() {
    std::cout << "=== Kimera-RPGO Basics ===" << std::endl;

    part3viz::Viz viz("part3_rpgo_basics", "kimera_rpgo");

    // =========================================
    // 1. RobustSolver parameters
    // =========================================
    std::cout << "\n1. Setting up RobustSolver parameters..." << std::endl;

    // PCM-Simple 3D thresholds are per-node drift allowances:
    // - translation_threshold: expected drift in metres per node
    // - rotation_threshold: expected drift in radians per node
    const double translation_threshold = 0.5;  // 50 cm per node
    const double rotation_threshold = 0.1;     // ~6 deg per node

    KimeraRPGO::RobustSolverParams params;
    // Verbosity::UPDATE keeps the solver's own logs and silences PCM's
    // per-loop-closure diagnostics; Verbosity::VERBOSE turns those on as well
    // (the outlier-rejection demo uses VERBOSE for exactly that reason).
    params.setPcmSimple3DParams(translation_threshold, rotation_threshold,
                                KimeraRPGO::Verbosity::UPDATE);

    KimeraRPGO::RobustSolver solver(params);

    std::cout << "   Translation threshold: " << translation_threshold << " m per node"
              << std::endl;
    std::cout << "   Rotation threshold: " << rotation_threshold << " rad per node"
              << std::endl;

    // =========================================
    // 2. Noise models
    // =========================================
    std::cout << "\n2. Defining noise models..." << std::endl;

    // NOTE on the sigma ordering: GTSAM's Pose3 tangent space is
    // (rotation, translation) - the first three entries are rx, ry, rz in
    // radians and the last three are tx, ty, tz in metres. That reads
    // backwards from how the problem is usually described, and getting it
    // wrong silently swaps the rotation and translation weights.
    const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
    const auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.02, 0.02, 0.02, 0.05, 0.05, 0.05).finished());
    const auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.02, 0.02, 0.02).finished());

    std::cout << "   Odometry sigma: 0.02 rad (rotation), 0.05 m (translation)"
              << std::endl;
    std::cout << "   Loop closure sigma: 0.01 rad (rotation), 0.02 m (translation)"
              << std::endl;

    // =========================================
    // 3. Ground truth and corrupted odometry
    // =========================================
    std::cout << "\n3. Building the pose graph (seed " << kSeed << ")..." << std::endl;

    const std::vector<gtsam::Pose3> gt_odometry = groundTruthOdometry();
    const std::vector<gtsam::Pose3> gt_poses = chain(gt_odometry);

    // Corrupt every odometry measurement, so the chained initial guess drifts
    // and the square does not close.
    std::mt19937 rng(kSeed);
    std::normal_distribution<double> trans_noise(0.0, 0.05);  // m
    std::normal_distribution<double> rot_noise(0.0, 0.02);    // rad

    std::vector<gtsam::Pose3> noisy_odometry;
    noisy_odometry.reserve(gt_odometry.size());
    for (const auto& t : gt_odometry) {
        const gtsam::Vector6 delta =
            (gtsam::Vector(6) << rot_noise(rng), rot_noise(rng), rot_noise(rng),
             trans_noise(rng), trans_noise(rng), trans_noise(rng))
                .finished();
        noisy_odometry.push_back(t * gtsam::Pose3::Expmap(delta));
    }

    const std::vector<gtsam::Pose3> init_poses = chain(noisy_odometry);

    gtsam::NonlinearFactorGraph odom_factors;
    gtsam::Values init_values;
    odom_factors.addPrior(X(0), gt_poses.front(), prior_noise);
    init_values.insert(X(0), init_poses.front());
    for (std::size_t i = 0; i < noisy_odometry.size(); ++i) {
        odom_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(static_cast<int>(i)), X(static_cast<int>(i + 1)), noisy_odometry[i],
            odom_noise);
        init_values.insert(X(static_cast<int>(i + 1)), init_poses[i + 1]);
    }

    std::cout << "   Poses: " << init_values.size() << std::endl;
    std::cout << "   Odometry factors: " << noisy_odometry.size() << std::endl;
    std::cout << "   Prior factors: 1 (gauge anchor on X(0))" << std::endl;

    // The loop closure is exact: pose 7 is pose 0 turned by another 90 deg.
    const gtsam::Pose3 loop_measurement = gt_poses.back().between(gt_poses.front());
    gtsam::NonlinearFactorGraph loop_factors;
    loop_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(kNumPoses - 1), X(0), loop_measurement, loop_noise);
    std::cout << "   Loop closures: 1  (X(" << kNumPoses - 1 << ") -> X(0), exact)"
              << std::endl;

    // The full graph, used only to score the estimates with one formula.
    gtsam::NonlinearFactorGraph full_graph = odom_factors;
    full_graph.add(loop_factors);

    // =========================================
    // 4. Optimize
    // =========================================
    std::cout << "\n4. Running the RobustSolver..." << std::endl;

    // update() runs outlier rejection and then optimizes, so the odometry and
    // the loop closure are handed over in two calls exactly as an incremental
    // SLAM front end would.
    solver.update(odom_factors, init_values);
    const std::size_t factors_after_odom = solver.getFactorsUnsafe().size();
    solver.update(loop_factors, gtsam::Values());

    const gtsam::Values result = solver.calculateEstimate();

    const double error_before = full_graph.error(init_values);
    const double error_after = full_graph.error(result);
    const double rmse_before = translationRmse(gt_poses, init_values);
    const double rmse_after = translationRmse(gt_poses, result);

    std::cout << "   Factors in the solver after odometry: " << factors_after_odom
              << std::endl;
    std::cout << "   Factors in the solver after the loop closure: "
              << solver.getFactorsUnsafe().size() << std::endl;
    std::cout << "   Loop closures seen: " << solver.getNumLC()
              << ", kept as inliers: " << solver.getNumLCInliers() << std::endl;

    // =========================================
    // 5. Results
    // =========================================
    std::cout << "\n5. Results (graph error is GTSAM's 0.5 x chi-squared):"
              << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "   Graph error before: " << error_before << std::endl;
    std::cout << "   Graph error after:  " << error_after << std::endl;
    std::cout << "   Translation RMSE vs ground truth before: " << rmse_before << " m"
              << std::endl;
    std::cout << "   Translation RMSE vs ground truth after:  " << rmse_after << " m"
              << std::endl;

    std::cout << "\n   pose |        ground truth |       initial guess |"
                 "           optimized"
              << std::endl;
    for (int i = 0; i < kNumPoses; ++i) {
        const gtsam::Point3 g = gt_poses[i].translation();
        const gtsam::Point3 n = init_poses[i].translation();
        const gtsam::Point3 o = result.at<gtsam::Pose3>(X(i)).translation();
        std::cout << "   X(" << i << ") | " << std::setw(6) << g.x() << " "
                  << std::setw(6) << g.y() << " " << std::setw(5) << g.z() << " | "
                  << std::setw(6) << n.x() << " " << std::setw(6) << n.y() << " "
                  << std::setw(5) << n.z() << " | " << std::setw(6) << o.x() << " "
                  << std::setw(6) << o.y() << " " << std::setw(5) << o.z() << std::endl;
    }

    // The square closes only if the loop closure was accepted: pose 7 and pose
    // 0 must end up on the same spot.
    const double closure_gap = (result.at<gtsam::Pose3>(X(kNumPoses - 1)).translation() -
                                result.at<gtsam::Pose3>(X(0)).translation())
                                   .norm();
    std::cout << "\n   Loop closure gap |t7 - t0| : initial "
              << (init_poses.back().translation() - init_poses.front().translation())
                     .norm()
              << " m -> optimized " << closure_gap << " m" << std::endl;

    // =========================================
    // 6. Stream to the viewer
    // =========================================
    std::vector<part3viz::Edge> edges;
    for (int i = 0; i + 1 < kNumPoses; ++i) {
        edges.push_back({i, i + 1, part3viz::EdgeKind::Odometry});
    }
    edges.push_back({kNumPoses - 1, 0, part3viz::EdgeKind::Loop});

    // Three static graphs under graph3d/kimera_rpgo/: the square as it should
    // be, the drifted odometry chain, and the optimized result. Everything is
    // logged as static because the solver exposes no per-iteration hook.
    // The colours come from the shared palette, which is plain RGB and defined
    // whether or not the rerun SDK is present, so no #ifdef is needed here.
    viz.poseGraph2D("ground_truth", planar(gt_poses), edges,
                    part3viz::kGroundTruth, true);
    viz.poseGraph2D("initial", planar(init_poses), edges,
                    part3viz::kInitial, true);
    viz.poseGraph2D("optimized", planar(result, kNumPoses), edges,
                    part3viz::kOptimized, true);

    std::cout << "\n=== RPGO Basics Complete ===" << std::endl;
    return 0;
}
