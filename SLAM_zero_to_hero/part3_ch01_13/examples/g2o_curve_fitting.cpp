/**
 * g2o Tutorial: Curve Fitting
 *
 * Fits y = exp(a*x^2 + b*x + c) to noisy samples with a custom g2o vertex (the
 * 3 parameters) and a custom unary edge (one per observation) carrying an
 * analytic Jacobian.
 *
 * This is the shared curve-fitting exercise of part3 chapter 1: the same data,
 * the same noise model, the same solver family and the same reported cost as
 * the GTSAM / Ceres / SymForce chapters, so the four runs are directly
 * comparable in one rerun viewer.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include "rerun_viz.hpp"

namespace {

// Shared problem definition - identical in every chapter of this series.
constexpr int kNumSamples = 100;
constexpr double kSigma = 0.2;  // observation noise -> information 1/sigma^2 = 25
constexpr int kMaxIterations = 30;
constexpr unsigned kSeed = 42;
// Relative chi2 decrease below which the problem counts as converged.
constexpr double kGainThreshold = 1e-6;

const part3viz::Abc kGroundTruth{1.0, 2.0, 1.0};
const part3viz::Abc kInitialGuess{2.0, -1.0, 5.0};

/**
 * The one shared cost formula:
 *   chi2 = sum_i ((y_i - exp(a x_i^2 + b x_i + c)) / sigma)^2
 *
 * Computed here rather than taken from the solver on purpose. Every library in
 * this series defines its own "error"/"cost" number slightly differently (some
 * carry a factor of 0.5), so the graphs would not be comparable.
 */
double curveChi2(const std::vector<double>& xs, const std::vector<double>& ys,
                 const part3viz::Abc& abc) {
    double sum = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double r = (ys[i] - part3viz::curveModel(abc, xs[i])) / kSigma;
        sum += r * r;
    }
    return sum;
}

// Vertex: the 3 curve parameters (a, b, c).
class CurveVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate << 0, 0, 0; }
    void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// Unary edge: residual of a single (x, y) observation, analytic Jacobian.
class CurveEdge : public g2o::BaseUnaryEdge<1, double, CurveVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit CurveEdge(double x) : x_(x) {}

    void computeError() override {
        const auto* v = static_cast<const CurveVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) =
            _measurement - std::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
    }

    void linearizeOplus() override {
        const auto* v = static_cast<const CurveVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        const double y = std::exp(abc[0] * x_ * x_ + abc[1] * x_ + abc[2]);
        _jacobianOplusXi[0] = -x_ * x_ * y;
        _jacobianOplusXi[1] = -x_ * y;
        _jacobianOplusXi[2] = -y;
    }

    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    double x_;
};

/**
 * Post-iteration hook: streams the current curve to the viewer and stops the
 * optimizer once the relative chi2 decrease drops below kGainThreshold.
 *
 * g2o also ships g2o::SparseOptimizerTerminateAction for exactly this kind of
 * gain-based stop; the criterion is spelled out by hand here so that it is
 * bit-for-bit the same test the sibling chapters use.
 */
struct IterationRecorder : public g2o::HyperGraphAction {
    IterationRecorder(const CurveVertex* v, const std::vector<double>& xs,
                      const std::vector<double>& ys, part3viz::Viz* viz, bool* stop)
        : v_(v), xs_(xs), ys_(ys), viz_(viz), stop_(stop) {}

    /// Log one frame; returns the chi2 of the state just logged.
    double record(int iter) {
        const Eigen::Vector3d e = v_->estimate();
        const part3viz::Abc abc{e[0], e[1], e[2]};
        const double chi2 = curveChi2(xs_, ys_, abc);
        viz_->curveIteration(iter, abc, chi2);
        history.push_back({abc, chi2});
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
        part3viz::Abc abc;
        double chi2;
    };
    std::vector<Frame> history;

private:
    const CurveVertex* v_;
    const std::vector<double>& xs_;
    const std::vector<double>& ys_;
    part3viz::Viz* viz_;
    bool* stop_;
};

}  // namespace

int main() {
    std::cout << "=== g2o Tutorial: Curve Fitting ===\n" << std::endl;

    // Samples: y = exp(a x^2 + b x + c) + N(0, sigma^2), fixed seed.
    std::mt19937 rng(kSeed);
    std::normal_distribution<double> noise(0.0, kSigma);
    std::vector<double> xs(kNumSamples), ys(kNumSamples);
    for (int i = 0; i < kNumSamples; ++i) {
        const double x = static_cast<double>(i) / kNumSamples;
        xs[i] = x;
        ys[i] = part3viz::curveModel(kGroundTruth, x) + noise(rng);
    }

    part3viz::Viz viz(part3viz::kCurveFittingRecording, "g2o");
    viz.curveSetup(xs, ys, kGroundTruth, kInitialGuess);

    // Solver: 3-DoF block, dense linear solver, Levenberg-Marquardt.
    // g2o also ships OptimizationAlgorithmGaussNewton (and ...Dogleg); LM is
    // used here so the iteration counts match the sibling chapters.
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    auto* v = new CurveVertex();
    v->setEstimate(
        Eigen::Vector3d(kInitialGuess[0], kInitialGuess[1], kInitialGuess[2]));
    v->setId(0);
    optimizer.addVertex(v);

    const double information = 1.0 / (kSigma * kSigma);  // 25
    for (int i = 0; i < kNumSamples; ++i) {
        auto* e = new CurveEdge(xs[i]);
        e->setId(i);
        e->setVertex(0, v);
        e->setMeasurement(ys[i]);
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * information);
        optimizer.addEdge(e);
    }

    std::cout << "Samples      : " << kNumSamples << "  sigma=" << kSigma
              << "  information=" << information << std::endl;
    std::cout << "Ground truth : a=" << kGroundTruth[0] << " b=" << kGroundTruth[1]
              << " c=" << kGroundTruth[2] << std::endl;
    std::cout << "Initial guess: a=" << kInitialGuess[0] << " b=" << kInitialGuess[1]
              << " c=" << kInitialGuess[2] << std::endl;
    std::cout << std::endl;

    bool stop = false;
    IterationRecorder recorder(v, xs, ys, &viz, &stop);
    optimizer.setForceStopFlag(&stop);
    optimizer.addPostIterationAction(&recorder);

    optimizer.initializeOptimization();
    recorder.record(0);  // frame 0: the initial state
    optimizer.optimize(kMaxIterations);
    optimizer.setForceStopFlag(nullptr);

    const int iterations = static_cast<int>(recorder.history.size()) - 1;
    const double chi2_initial = recorder.history.front().chi2;
    const double chi2_final = recorder.history.back().chi2;
    const part3viz::Abc abc = recorder.history.back().abc;

    std::cout << "\nIter |      a       b       c |      chi2" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    for (std::size_t i = 0; i < recorder.history.size(); ++i) {
        const auto& f = recorder.history[i];
        std::printf("%4zu | %7.4f %7.4f %7.4f | %9.4f\n", i, f.abc[0], f.abc[1],
                    f.abc[2], f.chi2);
    }

    std::cout << "\nEstimated    : a=" << abc[0] << " b=" << abc[1] << " c=" << abc[2]
              << std::endl;
    std::printf("chi2         : %.4f -> %.4f  (%.4f%% lower)\n", chi2_initial,
                chi2_final, (1.0 - chi2_final / chi2_initial) * 100.0);
    std::printf("Iterations   : %d of %d%s\n", iterations, kMaxIterations,
                iterations < kMaxIterations ? " (converged early)" : "");

    return 0;
}
