/**
 * GTSAM Tutorial: 2D Pose-Graph Optimization (PGO)
 *
 * A robot drives a square loop. We build a factor graph with a prior on x0,
 * BetweenFactor<Pose2> odometry derived from ground truth, and one loop-closure
 * factor (x4 -> x0); then optimize from a noisy initial estimate. Dumps
 * `pose_graph.txt` for viz/plot_pose_graph.py.
 */

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

using namespace std;
using namespace gtsam;

int main() {
    cout << "=== GTSAM Tutorial: 2D Pose-Graph Optimization ===\n" << endl;

    const int N = 5;
    array<Pose2, N> gt = {Pose2(0, 0, 0), Pose2(1, 0, 0), Pose2(1, 1, M_PI / 2),
                          Pose2(0, 1, M_PI), Pose2(0, 0, -M_PI / 2)};

    // Edges: 4 odometry + 1 loop closure (i, j, type).
    vector<tuple<int, int, int>> edges = {
        {0, 1, 0}, {1, 2, 0}, {2, 3, 0}, {3, 4, 0}, {4, 0, 1}};

    NonlinearFactorGraph graph;
    auto prior_noise = noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.005));
    auto odo_noise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));

    graph.addPrior(Symbol('x', 0), gt[0], prior_noise);  // anchor
    for (auto& e : edges) {
        int i = get<0>(e), j = get<1>(e);
        graph.add(BetweenFactor<Pose2>(Symbol('x', i), Symbol('x', j),
                                       gt[i].between(gt[j]), odo_noise));
    }

    // Noisy initial estimate (keep a copy for visualization).
    mt19937 rng(7);
    normal_distribution<double> nxy(0.0, 0.15), nth(0.0, 0.08);
    array<Pose2, N> init;
    Values initial;
    for (int i = 0; i < N; ++i) {
        init[i] = (i == 0) ? gt[0]
                           : Pose2(gt[i].x() + nxy(rng), gt[i].y() + nxy(rng),
                                   gt[i].theta() + nth(rng));
        initial.insert(Symbol('x', i), init[i]);
    }

    cout << "Initial error: " << graph.error(initial) << endl;
    LevenbergMarquardtParams params;
    params.setVerbosity("SUMMARY");
    Values result = LevenbergMarquardtOptimizer(graph, initial, params).optimize();
    cout << "Final error:   " << graph.error(result) << endl;

    cout << "\nPose | ground truth        | optimized           | error" << endl;
    cout << string(63, '-') << endl;
    array<Pose2, N> opt;
    for (int i = 0; i < N; ++i) {
        opt[i] = result.at<Pose2>(Symbol('x', i));
        double err = hypot(opt[i].x() - gt[i].x(), opt[i].y() - gt[i].y());
        printf("  x%d | (%5.2f,%5.2f,%5.2f) | (%5.2f,%5.2f,%5.2f) | %.4f\n", i,
               gt[i].x(), gt[i].y(), gt[i].theta(), opt[i].x(), opt[i].y(),
               opt[i].theta(), err);
    }

    auto dump = [](ofstream& o, const Pose2& p) {
        o << p.x() << " " << p.y() << " " << p.theta();
    };
    ofstream out("pose_graph.txt");
    out << "nodes " << N << "\n";
    for (int i = 0; i < N; ++i) {
        out << i << " ";
        dump(out, gt[i]); out << " ";
        dump(out, init[i]); out << " ";
        dump(out, opt[i]); out << "\n";
    }
    out << "edges " << edges.size() << "\n";
    for (auto& e : edges)
        out << get<0>(e) << " " << get<1>(e) << " " << get<2>(e) << "\n";
    out.close();
    cout << "\nWrote pose_graph.txt -> visualize with viz/plot_pose_graph.py" << endl;

    return 0;
}
