/**
 * GTSAM Tutorial: Curve Fitting
 *
 * Fits y = exp(a*x^2 + b*x + c) to noisy data with a custom GTSAM factor over a
 * single Vector3 variable (a, b, c), providing the analytic Jacobian. Dumps
 * `curve_fitting.txt` for viz/plot_curve_fitting.py.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

using namespace std;
using namespace gtsam;

// Factor: residual exp(a x^2 + b x + c) - y for one observation.
class CurveFactor : public NoiseModelFactorN<Vector3> {
public:
    CurveFactor(Key key, double x, double y, const SharedNoiseModel& model)
        : NoiseModelFactorN<Vector3>(model, key), x_(x), y_(y) {}

    Vector evaluateError(const Vector3& abc, OptionalMatrixType H) const override {
        double e = exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
        if (H) *H = (Matrix(1, 3) << x_ * x_ * e, x_ * e, e).finished();
        return (Vector(1) << e - y_).finished();
    }

private:
    double x_, y_;
};

int main() {
    cout << "=== GTSAM Tutorial: Curve Fitting ===\n" << endl;

    const Vector3 gt(1.0, 2.0, 1.0);
    const Vector3 init(2.0, -1.0, 5.0);

    const int N = 100;
    const double sigma = 0.2;
    mt19937 rng(42);
    normal_distribution<double> noise(0.0, sigma);
    vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N;
        xs[i] = x;
        ys[i] = exp(gt(0) * x * x + gt(1) * x + gt(2)) + noise(rng);
    }

    NonlinearFactorGraph graph;
    auto model = noiseModel::Isotropic::Sigma(1, sigma);
    Key k = Symbol('p', 0);
    for (int i = 0; i < N; ++i)
        graph.emplace_shared<CurveFactor>(k, xs[i], ys[i], model);

    Values initial;
    initial.insert(k, init);

    cout << "Ground truth : a=" << gt(0) << " b=" << gt(1) << " c=" << gt(2) << endl;
    cout << "Initial guess: a=" << init(0) << " b=" << init(1) << " c=" << init(2) << endl;

    LevenbergMarquardtParams params;
    params.setVerbosity("SUMMARY");
    Values result = LevenbergMarquardtOptimizer(graph, initial, params).optimize();
    Vector3 abc = result.at<Vector3>(k);
    cout << "Estimated    : a=" << abc(0) << " b=" << abc(1) << " c=" << abc(2) << endl;

    ofstream out("curve_fitting.txt");
    out << "model exp(a*x^2+b*x+c)\n";
    out << "gt " << gt(0) << " " << gt(1) << " " << gt(2) << "\n";
    out << "init " << init(0) << " " << init(1) << " " << init(2) << "\n";
    out << "est " << abc(0) << " " << abc(1) << " " << abc(2) << "\n";
    out << "data " << N << "\n";
    for (int i = 0; i < N; ++i) out << xs[i] << " " << ys[i] << "\n";
    out.close();
    cout << "\nWrote curve_fitting.txt -> visualize with viz/plot_curve_fitting.py" << endl;

    return 0;
}
