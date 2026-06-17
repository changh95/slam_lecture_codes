/**
 * Ceres-Solver Tutorial: Curve Fitting
 *
 * Fits the classic nonlinear model  y = exp(a*x^2 + b*x + c)  to noisy data
 * using Ceres automatic differentiation, then dumps the result to
 * `curve_fitting.txt` for visualization with viz/plot_curve_fitting.py.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <ceres/ceres.h>

using namespace std;

// Residual: r = y - exp(a*x^2 + b*x + c), with abc packed in one block.
struct CurveResidual {
    CurveResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        residual[0] = T(y_) - exp(abc[0] * T(x_) * T(x_) + abc[1] * T(x_) + abc[2]);
        return true;
    }

    static ceres::CostFunction* Create(double x, double y) {
        return new ceres::AutoDiffCostFunction<CurveResidual, 1, 3>(
            new CurveResidual(x, y));
    }

private:
    const double x_, y_;
};

int main() {
    cout << "=== Ceres Tutorial: Curve Fitting ===\n" << endl;

    // Ground-truth model parameters and the (deliberately bad) initial guess.
    const double gt[3] = {1.0, 2.0, 1.0};
    double abc[3] = {2.0, -1.0, 5.0};

    // Generate N noisy samples of y = exp(a*x^2 + b*x + c) over x in [0, 1].
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

    cout << "Ground truth : a=" << gt[0] << " b=" << gt[1] << " c=" << gt[2] << endl;
    cout << "Initial guess: a=" << abc[0] << " b=" << abc[1] << " c=" << abc[2] << endl;

    // Build and solve.
    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        problem.AddResidualBlock(CurveResidual::Create(xs[i], ys[i]), nullptr, abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << "\n" << summary.BriefReport() << endl;
    cout << "Estimated    : a=" << abc[0] << " b=" << abc[1] << " c=" << abc[2] << endl;

    // Dump result for visualization.
    ofstream out("curve_fitting.txt");
    out << "model exp(a*x^2+b*x+c)\n";
    out << "gt " << gt[0] << " " << gt[1] << " " << gt[2] << "\n";
    out << "init 2.0 -1.0 5.0\n";
    out << "est " << abc[0] << " " << abc[1] << " " << abc[2] << "\n";
    out << "data " << N << "\n";
    for (int i = 0; i < N; ++i) out << xs[i] << " " << ys[i] << "\n";
    out.close();
    cout << "\nWrote curve_fitting.txt -> visualize with viz/plot_curve_fitting.py" << endl;

    return 0;
}
