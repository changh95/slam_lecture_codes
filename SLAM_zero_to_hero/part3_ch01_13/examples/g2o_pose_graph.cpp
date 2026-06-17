/**
 * g2o Tutorial: 2D Pose-Graph Optimization (PGO)
 *
 * A robot drives a square loop. We use g2o's built-in VertexSE2 / EdgeSE2,
 * synthesize *consistent* odometry from the ground-truth poses plus one loop
 * closure (x4 -> x0), corrupt the initial estimate with noise, and optimize.
 * Dumps `pose_graph.txt` for viz/plot_pose_graph.py.
 */

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam2d/types_slam2d.h>

using namespace std;
using namespace g2o;

int main() {
    cout << "=== g2o Tutorial: 2D Pose-Graph Optimization ===\n" << endl;

    // Ground-truth square trajectory.
    const int N = 5;
    array<SE2, N> gt = {SE2(0, 0, 0), SE2(1, 0, 0), SE2(1, 1, M_PI / 2),
                        SE2(0, 1, M_PI), SE2(0, 0, -M_PI / 2)};

    // Edges: 4 odometry + 1 loop closure (i, j, type), measurement from GT.
    vector<tuple<int, int, int>> edges = {
        {0, 1, 0}, {1, 2, 0}, {2, 3, 0}, {3, 4, 0}, {4, 0, 1}};

    // Solver: variable block sizes, dense, Levenberg-Marquardt.
    using BlockSolverType = BlockSolver<BlockSolverTraits<-1, -1>>;
    using LinearSolverType = LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto solver = new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Noisy initial estimate (keep a copy for visualization).
    mt19937 rng(7);
    normal_distribution<double> nxy(0.0, 0.15), nth(0.0, 0.08);
    array<SE2, N> init;
    for (int i = 0; i < N; ++i) {
        auto* v = new VertexSE2();
        v->setId(i);
        if (i == 0) {
            v->setEstimate(gt[0]);
            v->setFixed(true);  // anchor
            init[0] = gt[0];
        } else {
            SE2 noisy(gt[i].translation().x() + nxy(rng),
                      gt[i].translation().y() + nxy(rng),
                      gt[i].rotation().angle() + nth(rng));
            v->setEstimate(noisy);
            init[i] = noisy;
        }
        optimizer.addVertex(v);
    }

    Eigen::Matrix3d information = Eigen::Matrix3d::Identity() * 100.0;
    for (auto& e : edges) {
        int i = get<0>(e), j = get<1>(e);
        auto* edge = new EdgeSE2();
        edge->setVertex(0, optimizer.vertex(i));
        edge->setVertex(1, optimizer.vertex(j));
        edge->setMeasurement(gt[i].inverse() * gt[j]);  // consistent measurement
        edge->setInformation(information);
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "\nPose | ground truth        | optimized           | error" << endl;
    cout << string(63, '-') << endl;
    array<SE2, N> opt;
    for (int i = 0; i < N; ++i) {
        opt[i] = static_cast<VertexSE2*>(optimizer.vertex(i))->estimate();
        double err = hypot(opt[i].translation().x() - gt[i].translation().x(),
                           opt[i].translation().y() - gt[i].translation().y());
        printf("  x%d | (%5.2f,%5.2f,%5.2f) | (%5.2f,%5.2f,%5.2f) | %.4f\n", i,
               gt[i].translation().x(), gt[i].translation().y(), gt[i].rotation().angle(),
               opt[i].translation().x(), opt[i].translation().y(), opt[i].rotation().angle(),
               err);
    }

    auto dump = [](ofstream& o, const SE2& p) {
        o << p.translation().x() << " " << p.translation().y() << " "
          << p.rotation().angle();
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
