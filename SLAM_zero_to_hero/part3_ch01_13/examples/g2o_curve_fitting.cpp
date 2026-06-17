/**
 * g2o Tutorial: Curve Fitting
 *
 * Fits y = exp(a*x^2 + b*x + c) to noisy data using a custom g2o vertex
 * (the 3 parameters) and a custom unary edge (one per observation) with an
 * analytic Jacobian. Dumps `curve_fitting.txt` for viz/plot_curve_fitting.py.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;

// Vertex: the 3 curve parameters (a, b, c).
class CurveVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate << 0, 0, 0; }
    void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }
    bool read(istream&) override { return false; }
    bool write(ostream&) const override { return false; }
};

// Unary edge: residual of a single (x, y) observation.
class CurveEdge : public g2o::BaseUnaryEdge<1, double, CurveVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit CurveEdge(double x) : x_(x) {}

    void computeError() override {
        const auto* v = static_cast<const CurveVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
    }

    void linearizeOplus() override {
        const auto* v = static_cast<const CurveVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
        _jacobianOplusXi[0] = -x_ * x_ * y;
        _jacobianOplusXi[1] = -x_ * y;
        _jacobianOplusXi[2] = -y;
    }

    bool read(istream&) override { return false; }
    bool write(ostream&) const override { return false; }

private:
    double x_;
};

int main() {
    cout << "=== g2o Tutorial: Curve Fitting ===\n" << endl;

    const double gt[3] = {1.0, 2.0, 1.0};
    const double init[3] = {2.0, -1.0, 5.0};

    const int N = 100;
    const double sigma = 0.2;
    mt19937 rng(42);
    normal_distribution<double> noise(0.0, sigma);
    vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N;
        xs[i] = x;
        ys[i] = exp(gt[0] * x * x + gt[1] * x + gt[2]) + noise(rng);
    }

    // Solver: 3-DoF block, dense linear solver, Gauss-Newton.
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    auto* v = new CurveVertex();
    v->setEstimate(Eigen::Vector3d(init[0], init[1], init[2]));
    v->setId(0);
    optimizer.addVertex(v);

    const double inv_sigma = 1.0 / sigma;
    for (int i = 0; i < N; ++i) {
        auto* e = new CurveEdge(xs[i]);
        e->setId(i);
        e->setVertex(0, v);
        e->setMeasurement(ys[i]);
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * inv_sigma * inv_sigma);
        optimizer.addEdge(e);
    }

    cout << "Ground truth : a=" << gt[0] << " b=" << gt[1] << " c=" << gt[2] << endl;
    cout << "Initial guess: a=" << init[0] << " b=" << init[1] << " c=" << init[2] << endl;

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    Eigen::Vector3d abc = v->estimate();
    cout << "Estimated    : a=" << abc[0] << " b=" << abc[1] << " c=" << abc[2] << endl;

    ofstream out("curve_fitting.txt");
    out << "model exp(a*x^2+b*x+c)\n";
    out << "gt " << gt[0] << " " << gt[1] << " " << gt[2] << "\n";
    out << "init " << init[0] << " " << init[1] << " " << init[2] << "\n";
    out << "est " << abc[0] << " " << abc[1] << " " << abc[2] << "\n";
    out << "data " << N << "\n";
    for (int i = 0; i < N; ++i) out << xs[i] << " " << ys[i] << "\n";
    out.close();
    cout << "\nWrote curve_fitting.txt -> visualize with viz/plot_curve_fitting.py" << endl;

    return 0;
}
