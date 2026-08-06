/**
 * @file rpgo_outlier_rejection.cpp
 * @brief What Kimera-RPGO is for: three solvers on one graph with bad loops.
 *
 * The trajectory drives twice around the same 3 m square, so pose 12 revisits
 * pose 0, pose 15 revisits pose 3, and so on - those are genuine revisits, the
 * only situation in which a loop closure means anything. Four correct loop
 * closures tie the second lap to the first. Three outlier loop closures are
 * then added between poses that are far apart along the trajectory: they are
 * wrong by metres, but because PCM's odometry check allows a per-node drift
 * budget, a wrong loop between poses 18 nodes apart still slips through that
 * gate. They only fall when the pairwise consistency check compares them
 * against each other - which is exactly the case PCM was invented for.
 *
 * Three solvers get the identical graph:
 *   1. no rejection  - every loop closure trusted (setNoRejection)
 *   2. PCM           - odometry check + pairwise-consistency max clique
 *   3. GNC           - graduated non-convexity re-weighting
 *
 * A note on the GNC solver's setup, because it is easy to get wrong:
 * RobustSolverParams defaults to OutlierRemovalMethod::PCM3D, and
 * setGncInlierCostThresholdsAtProbability() only flips use_gnc_ on. Left like
 * that, PCM deletes the outliers before GNC ever sees them and GNC provably
 * does nothing (all weights come back 1.0). Disabling PCM's two checks with
 * negative thresholds - setPcm3DParams(-1, -1, ...) - keeps the outlier-removal
 * object alive (RobustSolver::optimize() gates GNC on
 * `use_gnc_ && outlier_removal_`) while letting every loop closure reach GNC.
 * setNoRejection() would null the object out and silently disable GNC too.
 */

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <KimeraRPGO/RobustSolver.h>
#include <KimeraRPGO/SolverParams.h>

#include "rerun_viz.hpp"

using gtsam::symbol_shorthand::X;

namespace {

/// Two laps of 12 steps around a 3 m square.
constexpr int kStepsPerLap = 12;
constexpr int kNumPoses = 2 * kStepsPerLap;  // 24
constexpr unsigned kSeed = 7;

using LoopKey = std::pair<int, int>;

/// Ground-truth odometry: drive 1 m per step and turn 90 deg at every corner
/// (every third step), so 12 steps close one lap exactly.
std::vector<gtsam::Pose3> groundTruthOdometry(int num_steps) {
    const gtsam::Rot3 turn = gtsam::Rot3::Rz(M_PI / 2);
    std::vector<gtsam::Pose3> odom;
    odom.reserve(static_cast<std::size_t>(num_steps));
    for (int k = 0; k < num_steps; ++k) {
        const bool corner = (k % 3) == 2;
        odom.emplace_back(corner ? turn : gtsam::Rot3::Identity(),
                          gtsam::Point3(1.0, 0.0, 0.0));
    }
    return odom;
}

std::vector<gtsam::Pose3> chain(const std::vector<gtsam::Pose3>& relative) {
    std::vector<gtsam::Pose3> poses{gtsam::Pose3::Identity()};
    for (const auto& r : relative) poses.push_back(poses.back() * r);
    return poses;
}

std::vector<part3viz::Vec3> translations(const std::vector<gtsam::Pose3>& poses) {
    std::vector<part3viz::Vec3> out;
    out.reserve(poses.size());
    for (const auto& p : poses) {
        out.push_back({p.translation().x(), p.translation().y(), p.translation().z()});
    }
    return out;
}

std::vector<part3viz::Vec3> translations(const gtsam::Values& values, int n) {
    std::vector<part3viz::Vec3> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const gtsam::Pose3 p = values.at<gtsam::Pose3>(X(i));
        out.push_back({p.translation().x(), p.translation().y(), p.translation().z()});
    }
    return out;
}

double translationRmse(const std::vector<gtsam::Pose3>& gt, const gtsam::Values& values) {
    double sum = 0.0;
    for (std::size_t i = 0; i < gt.size(); ++i) {
        const gtsam::Pose3 p = values.at<gtsam::Pose3>(X(static_cast<int>(i)));
        sum += (p.translation() - gt[i].translation()).squaredNorm();
    }
    return std::sqrt(sum / static_cast<double>(gt.size()));
}

/// The non-sequential BetweenFactors still present in a solver's graph, i.e.
/// the loop closures that survived outlier rejection.
std::set<LoopKey> survivingLoops(const gtsam::NonlinearFactorGraph& graph) {
    std::set<LoopKey> kept;
    for (const auto& factor : graph) {
        if (!factor) continue;
        // dynamic_cast on the raw pointer, so the code does not depend on
        // whether this GTSAM uses std:: or boost:: shared pointers.
        const auto* between =
            dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3>*>(factor.get());
        if (!between || between->keys().size() != 2) continue;
        const int i = static_cast<int>(gtsam::Symbol(between->keys().front()).index());
        const int j = static_cast<int>(gtsam::Symbol(between->keys().back()).index());
        if (std::abs(i - j) == 1) continue;  // odometry
        kept.emplace(i, j);
    }
    return kept;
}

void reportSolver(const std::string& name, const std::vector<gtsam::Pose3>& gt,
                  const gtsam::Values& result, std::size_t graph_size,
                  std::size_t loops_kept, std::size_t loops_total) {
    const gtsam::Pose3 final_pose = result.at<gtsam::Pose3>(X(kNumPoses - 1));
    const double final_err = (final_pose.translation() - gt.back().translation()).norm();
    // printf rather than iostream formatting, so no precision state leaks into
    // the solver's own logging.
    std::printf(
        "   %-13s loops kept %zu/%zu | factors %2zu | translation RMSE %7.3f m |"
        " final pose error %7.3f m\n",
        name.c_str(), loops_kept, loops_total, graph_size,
        translationRmse(gt, result), final_err);
}

}  // namespace

int main() {
    std::cout << "=== Kimera-RPGO Outlier Rejection Demo ===" << std::endl;

    part3viz::Viz viz("part3_rpgo_outliers", "kimera_rpgo");

    // =========================================
    // 1. Ground truth and odometry
    // =========================================
    std::cout << "\n1. Building the trajectory (two laps of a 3 m square, seed "
              << kSeed << ")..." << std::endl;

    const std::vector<gtsam::Pose3> gt_odom = groundTruthOdometry(kNumPoses - 1);
    const std::vector<gtsam::Pose3> gt_poses = chain(gt_odom);

    // GTSAM's Pose3 tangent order is (rotation, translation): the first three
    // sigmas are radians, the last three metres.
    const auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
    const auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.005, 0.005, 0.005, 0.02, 0.02, 0.02).finished());
    const auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());

    std::mt19937 rng(kSeed);
    std::normal_distribution<double> rot_noise(0.0, 0.003);   // rad per step
    std::normal_distribution<double> trans_noise(0.0, 0.01);  // m per step

    std::vector<gtsam::Pose3> noisy_odom;
    noisy_odom.reserve(gt_odom.size());
    for (const auto& t : gt_odom) {
        const gtsam::Vector6 delta =
            (gtsam::Vector(6) << rot_noise(rng), rot_noise(rng), rot_noise(rng),
             trans_noise(rng), trans_noise(rng), trans_noise(rng))
                .finished();
        noisy_odom.push_back(t * gtsam::Pose3::Expmap(delta));
    }
    const std::vector<gtsam::Pose3> init_poses = chain(noisy_odom);

    gtsam::NonlinearFactorGraph base_factors;
    gtsam::Values base_values;
    base_factors.addPrior(X(0), gt_poses.front(), prior_noise);
    base_values.insert(X(0), init_poses.front());
    for (std::size_t i = 0; i < noisy_odom.size(); ++i) {
        base_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(static_cast<int>(i)), X(static_cast<int>(i + 1)), noisy_odom[i],
            odom_noise);
        base_values.insert(X(static_cast<int>(i + 1)), init_poses[i + 1]);
    }
    std::cout << "   " << kNumPoses << " poses, " << noisy_odom.size()
              << " odometry factors, 1 prior on X(0)" << std::endl;

    // =========================================
    // 2. Correct loop closures at real revisits
    // =========================================
    std::cout << "\n2. Adding correct loop closures at genuine revisits..." << std::endl;

    const std::vector<LoopKey> valid_keys{{12, 0}, {15, 3}, {18, 6}, {21, 9}};
    gtsam::NonlinearFactorGraph valid_loops;
    for (const auto& [a, b] : valid_keys) {
        valid_loops.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(a), X(b), gt_poses[a].between(gt_poses[b]), loop_noise);
        std::cout << "   X(" << a << ") -> X(" << b
                  << ")  (same place, one lap apart)" << std::endl;
    }

    // =========================================
    // 3. Outlier loop closures
    // =========================================
    std::cout << "\n3. Adding OUTLIER loop closures..." << std::endl;

    // Each outlier is the true relative pose composed with a large, wrong
    // perturbation. The pose pairs are far apart along the trajectory, so the
    // error per node stays under PCM's odometry-drift budget and the outliers
    // reach the pairwise consistency stage.
    const std::vector<LoopKey> outlier_keys{{22, 4}, {23, 10}, {20, 2}};
    std::uniform_real_distribution<double> dir(-1.0, 1.0);
    std::uniform_real_distribution<double> mag(3.0, 5.0);
    std::uniform_real_distribution<double> rot(-0.4, 0.4);

    gtsam::NonlinearFactorGraph outlier_loops;
    for (const auto& [a, b] : outlier_keys) {
        // Draw into named locals first: deviates taken inside a function-call
        // argument list are evaluated in an unspecified order, which silently
        // changes the problem between compilers.
        const double ox = dir(rng);
        const double oy = dir(rng);
        gtsam::Point3 offset(ox, oy, 0.0);
        offset = offset / std::max(1e-9, offset.norm()) * mag(rng);
        const gtsam::Pose3 wrong_delta(gtsam::Rot3::Rz(rot(rng)), offset);
        outlier_loops.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(a), X(b), gt_poses[a].between(gt_poses[b]) * wrong_delta, loop_noise);
        std::printf("   X(%d) -> X(%d)  wrong by %.2f m over %d nodes\n", a, b,
                    offset.norm(), std::abs(a - b));
    }

    const std::size_t num_loops = valid_keys.size() + outlier_keys.size();
    std::cout << "   " << valid_keys.size() << " correct + " << outlier_keys.size()
              << " outlier loop closures = " << num_loops << " total" << std::endl;

    // =========================================
    // 4. Three solvers, one graph
    // =========================================
    std::cout << "\n4. Solver 1/3: no outlier rejection (baseline)..." << std::endl;

    // Verbosity::QUIET is not a style choice here, it is mandatory: with
    // rejection off, RobustSolver's outlier-removal object is null, and every
    // other verbosity leaves the solver's logging enabled - whose log line
    // dereferences that null object and segfaults. Upstream even admits it
    // ("TODO(yun) this seg faults we disable outlier removal"). So the
    // no-rejection baseline runs silently.
    KimeraRPGO::RobustSolverParams none_params;
    none_params.setNoRejection(KimeraRPGO::Verbosity::QUIET);
    KimeraRPGO::RobustSolver none_solver(none_params);
    none_solver.update(base_factors, base_values);
    none_solver.update(valid_loops, gtsam::Values());
    none_solver.update(outlier_loops, gtsam::Values());
    const gtsam::Values none_result = none_solver.calculateEstimate();
    const auto none_kept = survivingLoops(none_solver.getFactorsUnsafe());
    // getNumLC()/getNumLCInliers() dereference the outlier-removal object,
    // which setNoRejection() leaves null - do not call them on this solver.

    std::cout << "\n5. Solver 2/3: PCM..." << std::endl;

    // 4-argument form: a loose per-node budget for the odometry check (so the
    // outliers get past it) and a tight budget for the pairwise consistency
    // check that actually catches them.
    KimeraRPGO::RobustSolverParams pcm_params;
    pcm_params.setPcmSimple3DParams(0.5,   // odom check: m per node
                                    0.1,   // odom check: rad per node
                                    0.05,  // pairwise check: m per node
                                    0.02,  // pairwise check: rad per node
                                    KimeraRPGO::Verbosity::VERBOSE);
    KimeraRPGO::RobustSolver pcm_solver(pcm_params);
    pcm_solver.update(base_factors, base_values);
    const std::size_t pcm_factors_before_loops = pcm_solver.getFactorsUnsafe().size();
    pcm_solver.update(valid_loops, gtsam::Values());
    pcm_solver.update(outlier_loops, gtsam::Values());
    const gtsam::Values pcm_result = pcm_solver.calculateEstimate();
    const auto pcm_kept = survivingLoops(pcm_solver.getFactorsUnsafe());

    std::cout << "\n6. Solver 3/3: GNC..." << std::endl;

    KimeraRPGO::RobustSolverParams gnc_params;
    // Negative thresholds switch PCM's odometry and pairwise checks off while
    // keeping the outlier-removal object alive, which RobustSolver requires
    // before it will run GNC at all.
    gnc_params.setPcm3DParams(-1.0, -1.0, KimeraRPGO::Verbosity::VERBOSE);
    gnc_params.setGncInlierCostThresholdsAtProbability(0.99);
    KimeraRPGO::RobustSolver gnc_solver(gnc_params);
    gnc_solver.update(base_factors, base_values);
    gnc_solver.update(valid_loops, gtsam::Values());
    gnc_solver.update(outlier_loops, gtsam::Values());
    const gtsam::Values gnc_result = gnc_solver.calculateEstimate();
    const gtsam::Vector gnc_weights = gnc_solver.getGncWeights();

    // The loop closures are the tail of the solver's factor graph (odometry and
    // special factors first, then the loop closures in the order they were
    // added), so the last num_loops weights line up with the loop list.
    std::set<LoopKey> gnc_kept;
    std::vector<LoopKey> loop_order(valid_keys);
    loop_order.insert(loop_order.end(), outlier_keys.begin(), outlier_keys.end());
    if (gnc_weights.size() >= static_cast<int>(num_loops)) {
        const int offset = static_cast<int>(gnc_weights.size() - num_loops);
        for (std::size_t k = 0; k < num_loops; ++k) {
            if (gnc_weights(offset + static_cast<int>(k)) > 0.5) {
                gnc_kept.insert(loop_order[k]);
            }
        }
    }

    // =========================================
    // 5. Report
    // =========================================
    std::cout << "\n7. Inlier / outlier split:" << std::endl;
    std::cout << "   factors before any loop closure : " << pcm_factors_before_loops
              << std::endl;
    std::cout << "   PCM  getNumLC()        = " << pcm_solver.getNumLC()
              << "   (loop closures that reached the adjacency matrix)" << std::endl;
    std::cout << "   PCM  getNumLCInliers() = " << pcm_solver.getNumLCInliers()
              << "   (max-clique survivors)" << std::endl;
    std::cout << "   GNC  getNumLC()        = " << gnc_solver.getNumLC() << std::endl;
    std::cout << "   GNC  getNumLCInliers() = " << gnc_solver.getNumLCInliers()
              << std::endl;
    std::cout << "   Subtlety worth remembering: a loop closure killed by PCM's"
              << std::endl;
    std::cout << "   odometry check never enters the adjacency matrix, so it is not"
              << std::endl;
    std::cout << "   counted by getNumLC() either." << std::endl;

    std::cout << "\n   GNC weights for the " << num_loops << " loop closures:"
              << std::endl;
    if (gnc_weights.size() >= static_cast<int>(num_loops)) {
        const int offset = static_cast<int>(gnc_weights.size() - num_loops);
        for (std::size_t k = 0; k < num_loops; ++k) {
            const double w = gnc_weights(offset + static_cast<int>(k));
            std::printf("     X(%2d) -> X(%2d)  weight %5.3f   %-9s %s\n",
                        loop_order[k].first, loop_order[k].second, w,
                        k < valid_keys.size() ? "[correct]" : "[outlier]",
                        w > 0.5 ? "kept" : "rejected");
        }
    } else {
        std::cout << "     (GNC returned " << gnc_weights.size()
                  << " weights - GNC did not run)" << std::endl;
    }

    std::cout << "\n8. Accuracy of the three solvers:" << std::endl;
    reportSolver("no rejection", gt_poses, none_result,
                 none_solver.getFactorsUnsafe().size(), none_kept.size(), num_loops);
    reportSolver("PCM", gt_poses, pcm_result, pcm_solver.getFactorsUnsafe().size(),
                 pcm_kept.size(), num_loops);
    reportSolver("GNC", gt_poses, gnc_result, gnc_solver.getFactorsUnsafe().size(),
                 gnc_kept.size(), num_loops);
    std::printf("   %-13s %31s %7.3f m\n", "odometry only", "translation RMSE",
                translationRmse(gt_poses, base_values));

    // =========================================
    // 6. Stream to the viewer
    // =========================================
    std::vector<part3viz::Edge> odom_edges;
    for (int i = 0; i + 1 < kNumPoses; ++i) {
        odom_edges.push_back({i, i + 1, part3viz::EdgeKind::Odometry});
    }

    auto edgesFor = [&](const std::set<LoopKey>& kept) {
        std::vector<part3viz::Edge> e = odom_edges;
        for (const auto& lk : loop_order) {
            const bool alive = kept.count(lk) > 0;
            e.push_back({lk.first, lk.second,
                         alive ? part3viz::EdgeKind::Loop
                               : part3viz::EdgeKind::LoopRejected});
        }
        return e;
    };

    std::vector<part3viz::Edge> gt_edges = odom_edges;
    for (const auto& lk : valid_keys) {
        gt_edges.push_back({lk.first, lk.second, part3viz::EdgeKind::Loop});
    }
    for (const auto& lk : outlier_keys) {
        gt_edges.push_back({lk.first, lk.second, part3viz::EdgeKind::LoopRejected});
    }

    // Palette colours are plain RGB triples defined with or without the rerun
    // SDK, so these calls compile either way and need no #ifdef.
    const part3viz::Color3 kNoneColor{230, 150, 40};  // orange
    viz.poseGraph3D("ground_truth", translations(gt_poses), gt_edges,
                    part3viz::kGroundTruth, true);
    viz.poseGraph3D("initial", translations(init_poses), odom_edges,
                    part3viz::kInitial, true);
    viz.poseGraph3D("no_rejection", translations(none_result, kNumPoses),
                    edgesFor(none_kept), kNoneColor, true);
    viz.poseGraph3D("pcm", translations(pcm_result, kNumPoses), edgesFor(pcm_kept),
                    part3viz::kOptimized, true);
    viz.poseGraph3D("gnc", translations(gnc_result, kNumPoses), edgesFor(gnc_kept),
                    part3viz::kLoop, true);

    std::cout << "\n=== Outlier Rejection Demo Complete ===" << std::endl;
    return 0;
}
