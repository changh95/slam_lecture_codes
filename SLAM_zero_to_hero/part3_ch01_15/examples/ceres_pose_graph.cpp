/**
 * Ceres-Solver Tutorial: 2D Pose-Graph Optimization (PGO)
 *
 * A robot drives a square loop. We synthesize *consistent* odometry from the
 * ground-truth poses (relative transform in the local frame) plus one loop
 * closure (x4 -> x0), corrupt the initial estimate with noise, then let Ceres
 * pull the trajectory back onto the ground truth. Dumps `pose_graph.txt` for
 * viz/plot_pose_graph.py.
 */

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <ceres/ceres.h>

using namespace std;
using Pose = array<double, 3>;  // [x, y, theta]

template <typename T>
T NormalizeAngle(const T& a) {
    T two_pi = T(2.0 * M_PI);
    return a - two_pi * floor((a + T(M_PI)) / two_pi);
}

// Relative transform b expressed in a's local frame: a^{-1} * b.
Pose Relative(const Pose& a, const Pose& b) {
    double c = cos(a[2]), s = sin(a[2]);
    double dx = b[0] - a[0], dy = b[1] - a[1];
    return {c * dx + s * dy, -s * dx + c * dy, NormalizeAngle(b[2] - a[2])};
}

// Odometry / loop constraint between two 2D poses.
struct RelativeMotion {
    RelativeMotion(double dx, double dy, double dth, double w_xy, double w_th)
        : dx_(dx), dy_(dy), dth_(dth), w_xy_(w_xy), w_th_(w_th) {}

    template <typename T>
    bool operator()(const T* const pi, const T* const pj, T* r) const {
        T c = cos(pi[2]), s = sin(pi[2]);
        T dx = pj[0] - pi[0], dy = pj[1] - pi[1];
        r[0] = w_xy_ * (c * dx + s * dy - T(dx_));
        r[1] = w_xy_ * (-s * dx + c * dy - T(dy_));
        r[2] = w_th_ * NormalizeAngle(NormalizeAngle(pj[2] - pi[2]) - T(dth_));
        return true;
    }

    static ceres::CostFunction* Create(double dx, double dy, double dth,
                                       double w_xy, double w_th) {
        return new ceres::AutoDiffCostFunction<RelativeMotion, 3, 3, 3>(
            new RelativeMotion(dx, dy, dth, w_xy, w_th));
    }

private:
    const double dx_, dy_, dth_, w_xy_, w_th_;
};

int main() {
    cout << "=== Ceres Tutorial: 2D Pose-Graph Optimization ===\n" << endl;

    // Ground-truth square trajectory.
    const int N = 5;
    array<Pose, N> gt = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, M_PI / 2},
        {0.0, 1.0, M_PI},
        {0.0, 0.0, -M_PI / 2},
    }};

    // Edges: 4 odometry + 1 loop closure, measurements derived from GT.
    vector<tuple<int, int, int>> edges = {
        {0, 1, 0}, {1, 2, 0}, {2, 3, 0}, {3, 4, 0}, {4, 0, 1}};  // type 1 = loop

    // Noisy initial estimate (keep a copy for visualization).
    mt19937 rng(7);
    normal_distribution<double> nxy(0.0, 0.15), nth(0.0, 0.08);
    array<Pose, N> init, poses;
    for (int i = 0; i < N; ++i) {
        init[i] = {gt[i][0] + nxy(rng), gt[i][1] + nxy(rng), gt[i][2] + nth(rng)};
        poses[i] = init[i];
    }
    poses[0] = gt[0];  // anchor first pose at ground truth

    ceres::Problem problem;
    for (auto& e : edges) {
        int i = get<0>(e), j = get<1>(e);
        Pose m = Relative(gt[i], gt[j]);
        problem.AddResidualBlock(
            RelativeMotion::Create(m[0], m[1], m[2], 10.0, 5.0), nullptr,
            poses[i].data(), poses[j].data());
    }
    problem.SetParameterBlockConstant(poses[0].data());  // fix gauge

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << "\n" << summary.BriefReport() << endl;

    cout << "\nPose | ground truth        | optimized           | error" << endl;
    cout << string(63, '-') << endl;
    for (int i = 0; i < N; ++i) {
        double err = hypot(poses[i][0] - gt[i][0], poses[i][1] - gt[i][1]);
        printf("  x%d | (%5.2f,%5.2f,%5.2f) | (%5.2f,%5.2f,%5.2f) | %.4f\n", i,
               gt[i][0], gt[i][1], gt[i][2], poses[i][0], poses[i][1], poses[i][2], err);
    }

    // Dump result for visualization.
    ofstream out("pose_graph.txt");
    out << "nodes " << N << "\n";
    for (int i = 0; i < N; ++i) {
        out << i << " " << gt[i][0] << " " << gt[i][1] << " " << gt[i][2] << " "
            << init[i][0] << " " << init[i][1] << " " << init[i][2] << " "
            << poses[i][0] << " " << poses[i][1] << " " << poses[i][2] << "\n";
    }
    out << "edges " << edges.size() << "\n";
    for (auto& e : edges)
        out << get<0>(e) << " " << get<1>(e) << " " << get<2>(e) << "\n";
    out.close();
    cout << "\nWrote pose_graph.txt -> visualize with viz/plot_pose_graph.py" << endl;

    return 0;
}
