/**
 * Ceres-Solver Tutorial: Curve Fitting
 *
 * Fits the classic nonlinear model  y = exp(a*x^2 + b*x + c)  to noisy data
 * using Ceres automatic differentiation, and streams every solver iteration to
 * a rerun viewer while it runs.
 *
 * This is the shared curve-fitting exercise of part3 chapter 1: the same data,
 * the same initial guess, the same Levenberg-Marquardt budget and the same
 * chi-squared definition as the g2o / GTSAM / SymForce chapters, so the four
 * cost curves can be overlaid in one viewer.
 */

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include "rerun_viz.hpp"

using namespace std;

// Shared problem definition (identical in all chapters of this series).
static constexpr int kNumSamples = 100;
static constexpr double kSigma = 0.2;          // measurement noise std-dev
static constexpr int kMaxIterations = 30;
static const part3viz::Abc kGroundTruth = {1.0, 2.0, 1.0};
static const part3viz::Abc kInitialGuess = {2.0, -1.0, 5.0};

// Residual: r = y - exp(a*x^2 + b*x + c), with abc packed in one block.
// The residual is scaled by 1/sigma so Ceres minimizes the same chi-squared
// the other chapters report (up to Ceres' own factor of 0.5).
struct CurveResidual {
    CurveResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        const T model = exp(abc[0] * T(x_) * T(x_) + abc[1] * T(x_) + abc[2]);
        residual[0] = (T(y_) - model) / T(kSigma);
        return true;
    }

    static ceres::CostFunction* Create(double x, double y) {
        return new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
            new CurveResidual(x, y));
    }

private:
    const double x_, y_;
};

// The one shared chi-squared formula:  sum_i ((y_i - model(x_i)) / sigma)^2.
// Computed here rather than read from the solver, because every library in this
// series defines its internal "cost" slightly differently (Ceres, for one,
// carries a factor of 0.5) and the graphs have to be comparable.
static double Chi2(const part3viz::Abc& abc, const vector<double>& xs,
                   const vector<double>& ys) {
    double chi2 = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double r = (ys[i] - part3viz::curveModel(abc, xs[i])) / kSigma;
        chi2 += r * r;
    }
    return chi2;
}

// Streams the state after every accepted step. Ceres calls this once with
// iteration 0 for the initial state, then once per iteration.
struct IterationStreamer : public ceres::IterationCallback {
    IterationStreamer(part3viz::Viz& viz, const double* abc, const vector<double>& xs,
                      const vector<double>& ys)
        : viz_(viz), abc_(abc), xs_(xs), ys_(ys) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override {
        const part3viz::Abc abc = {abc_[0], abc_[1], abc_[2]};
        const double chi2 = Chi2(abc, xs_, ys_);
        viz_.curveIteration(s.iteration, abc, chi2);
        steps_ = s.iteration;
        return ceres::SOLVER_CONTINUE;
    }

    part3viz::Viz& viz_;
    const double* abc_;
    const vector<double>& xs_;
    const vector<double>& ys_;
    int steps_ = 0;
};

int main(int /*argc*/, char** argv) {
    google::InitGoogleLogging(argv[0]);
    cout << "=== Ceres Tutorial: Curve Fitting ===\n" << endl;

    part3viz::Viz viz(part3viz::kCurveFittingRecording, "ceres");

    // Generate N noisy samples of y = exp(a*x^2 + b*x + c) over x in [0, 1).
    // Fixed seed 42, shared with every other chapter in the series.
    mt19937 rng(42);
    normal_distribution<double> noise(0.0, kSigma);

    vector<double> xs(kNumSamples), ys(kNumSamples);
    for (int i = 0; i < kNumSamples; ++i) {
        const double x = static_cast<double>(i) / kNumSamples;
        xs[i] = x;
        ys[i] = part3viz::curveModel(kGroundTruth, x) + noise(rng);
    }

    double abc[3] = {kInitialGuess[0], kInitialGuess[1], kInitialGuess[2]};

    cout << "Ground truth : a=" << kGroundTruth[0] << " b=" << kGroundTruth[1]
         << " c=" << kGroundTruth[2] << endl;
    cout << "Initial guess: a=" << abc[0] << " b=" << abc[1] << " c=" << abc[2] << endl;
    cout << "Initial chi2 : " << Chi2(kInitialGuess, xs, ys) << endl;

    viz.curveSetup(xs, ys, kGroundTruth, kInitialGuess);

    // Build and solve.
    ceres::Problem problem;
    for (int i = 0; i < kNumSamples; ++i) {
        problem.AddResidualBlock(CurveResidual::Create(xs[i], ys[i]), nullptr, abc);
    }

    IterationStreamer streamer(viz, abc, xs, ys);

    ceres::Solver::Options options;
    // Levenberg-Marquardt (Ceres' default trust-region strategy), same as the
    // sibling chapters, so the iteration counts are comparable.
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = kMaxIterations;
    // Ceres stops as soon as its own convergence tests fire, so the run
    // normally ends well inside the 30-iteration budget.
    options.update_state_every_iteration = true;  // refresh abc before the callback
    options.callbacks.push_back(&streamer);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    const part3viz::Abc est = {abc[0], abc[1], abc[2]};

    cout << "\n" << summary.BriefReport() << endl;
    cout << "Stopped because: " << summary.message << endl;
    cout << "Estimated    : a=" << est[0] << " b=" << est[1] << " c=" << est[2] << endl;
    cout << "Final chi2   : " << Chi2(est, xs, ys) << endl;
    cout << "Iterations   : " << streamer.steps_ << " (frame 0 is the initial state)"
         << endl;

    return 0;
}
