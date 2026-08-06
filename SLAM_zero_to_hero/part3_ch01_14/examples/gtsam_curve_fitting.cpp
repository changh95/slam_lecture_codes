/**
 * GTSAM: curve fitting
 *
 * Fits y = exp(a*x^2 + b*x + c) to noisy samples with a custom GTSAM factor
 * over a single Vector3 variable (a, b, c), supplying the analytic Jacobian.
 * Solved with Levenberg-Marquardt, one LM step at a time so every iteration can
 * be streamed to a live rerun viewer.
 *
 * Shared exercise setup (identical in the g2o / Ceres / SymForce chapters):
 *   ground truth (a, b, c) = (1, 2, 1), initial guess (2, -1, 5),
 *   N = 100 samples at x_i = i/N, Gaussian noise sigma = 0.2, seed 42.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "rerun_viz.hpp"

using namespace std;
using namespace gtsam;

// GTSAM's VerbosityLM output rewrites cout's precision while it prints, so this
// puts it back before each of our own lines.
static ostream& msg() { return cout << setprecision(6); }

// Factor: residual exp(a x^2 + b x + c) - y for one observation.
class CurveFactor : public NoiseModelFactorN<Vector3> {
public:
    CurveFactor(Key key, double x, double y, const SharedNoiseModel& model)
        : NoiseModelFactorN<Vector3>(model, key), x_(x), y_(y) {}

    Vector evaluateError(const Vector3& abc, OptionalMatrixType H) const override {
        const double e = exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
        if (H) *H = (Matrix(1, 3) << x_ * x_ * e, x_ * e, e).finished();
        return (Vector(1) << e - y_).finished();
    }

    // The base-class clone() is a stub that throws. Every custom factor needs
    // this override: the moment the factor is copied - which is what ISAM2 and
    // NonlinearFactorGraph::clone() do internally - the stub is what runs.
    // GTSAM 4.3 uses std::shared_ptr; older builds with boost features enabled
    // want boost::static_pointer_cast here instead.
    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new CurveFactor(*this)));
    }

private:
    double x_, y_;
};

int main() {
    msg() << "=== GTSAM: curve fitting  y = exp(a x^2 + b x + c) ===\n" << endl;

    part3viz::Viz viz(part3viz::kCurveFittingRecording, "gtsam");

    const Vector3 gt(1.0, 2.0, 1.0);
    const Vector3 init(2.0, -1.0, 5.0);

    const int N = 100;
    const double sigma = 0.2;
    mt19937 rng(42);  // fixed seed: the same samples in every chapter
    normal_distribution<double> noise(0.0, sigma);
    vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        const double x = static_cast<double>(i) / N;
        xs[i] = x;
        ys[i] = exp(gt(0) * x * x + gt(1) * x + gt(2)) + noise(rng);
    }

    // Chi-squared, computed here rather than taken from the library: GTSAM's
    // NonlinearFactorGraph::error() is 0.5 * chi2, g2o reports chi2, Ceres
    // reports 0.5 * chi2. Computing it explicitly keeps the streamed cost curve
    // comparable across the four chapters.
    const auto chi2 = [&](const Vector3& abc) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            const double r = (ys[i] - exp(abc(0) * xs[i] * xs[i] + abc(1) * xs[i] +
                                          abc(2))) /
                             sigma;
            sum += r * r;
        }
        return sum;
    };

    NonlinearFactorGraph graph;
    auto model = noiseModel::Isotropic::Sigma(1, sigma);  // information 1/sigma^2 = 25
    const Key k = Symbol('p', 0);
    for (int i = 0; i < N; ++i) {
        graph.emplace_shared<CurveFactor>(k, xs[i], ys[i], model);
    }

    Values initial;
    initial.insert(k, init);

    msg() << "Ground truth : a=" << gt(0) << " b=" << gt(1) << " c=" << gt(2) << endl;
    msg() << "Initial guess: a=" << init(0) << " b=" << init(1) << " c=" << init(2)
         << endl;

    viz.curveSetup(xs, ys, {gt(0), gt(1), gt(2)}, {init(0), init(1), init(2)});

    LevenbergMarquardtParams params;
    // Two separate knobs: setVerbosity() takes NonlinearOptimizerParams::Verbosity
    // (SILENT/TERMINATION/ERROR/VALUES/DELTA/LINEAR) while "SUMMARY" is a
    // LevenbergMarquardtParams::VerbosityLM value. Passing "SUMMARY" to
    // setVerbosity() silently leaves the optimizer at SILENT.
    params.setVerbosityLM("SUMMARY");
    LevenbergMarquardtOptimizer optimizer(graph, initial, params);

    const int kMaxIterations = 30;
    double cost = chi2(init);
    msg() << "\nIteration 0: chi2 = " << cost << endl;
    viz.curveIteration(0, {init(0), init(1), init(2)}, cost);

    int iterations = 0;
    for (int it = 1; it <= kMaxIterations; ++it) {
        optimizer.iterate();
        const Vector3 abc = optimizer.values().at<Vector3>(k);
        const double next = chi2(abc);
        ++iterations;
        viz.curveIteration(it, {abc(0), abc(1), abc(2)}, next);
        msg() << "Iteration " << it << ": chi2 = " << next << "  a=" << abc(0)
             << " b=" << abc(1) << " c=" << abc(2) << endl;
        const bool converged = (cost - next) <= 1e-6 * max(1.0, cost);
        cost = next;
        if (converged) break;  // do not burn the remaining budget
    }

    const Vector3 abc = optimizer.values().at<Vector3>(k);
    msg() << "\nEstimated    : a=" << abc(0) << " b=" << abc(1) << " c=" << abc(2)
         << endl;
    msg() << "chi2         : " << chi2(init) << " -> " << cost << "  (" << iterations
         << " LM iterations)" << endl;

    return 0;
}
