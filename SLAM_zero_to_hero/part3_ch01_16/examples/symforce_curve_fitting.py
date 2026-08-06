#!/usr/bin/env python3
"""SymForce Tutorial: Curve Fitting

Fits y = exp(a*x^2 + b*x + c) to noisy data. SymForce builds the residual
symbolically and generates the Jacobian automatically, then minimizes with
Levenberg-Marquardt.

Every iteration is streamed live to a rerun viewer (curve, chi-squared, and the
three parameters) - see examples/rerun_viz.py. The run also prints its numbers,
so it is useful with no viewer at all.

Shared problem definition (identical in the g2o / GTSAM / Ceres chapters):
  ground truth (a, b, c) = (1, 2, 1), initial guess (2, -1, 5),
  N = 100 samples at x_i = i / N, Gaussian noise sigma = 0.2, seed 42,
  noise model sigma = 0.2 (information 1 / sigma^2 = 25), LM, max 30 iterations.

The noise *realization* is not identical: numpy's default_rng(42) is a different
generator from the C++ chapters' std::mt19937(42), and its Gaussian transform
differs too, so the 100 samples differ and this chapter's chi-squared and fitted
parameters come out slightly different. Same problem, different draw - see the
README section "What is identical to the C++ chapters, and what is not".
"""
import sys
from pathlib import Path

import symforce

symforce.set_epsilon_to_symbol()

import numpy as np
import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.values import Values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_viz import CURVE_FITTING_RECORDING, Viz  # noqa: E402

GT = (1.0, 2.0, 1.0)
INIT = (2.0, -1.0, 5.0)
N = 100
SIGMA = 0.2
MAX_ITERATIONS = 30


def curve_residual(
    params: sf.V3, x: sf.Scalar, y: sf.Scalar, inv_sigma: sf.Scalar, epsilon: sf.Scalar
) -> sf.V1:
    """Whitened residual (exp(a x^2 + b x + c) - y) / sigma for one observation.

    Dividing by sigma is this chapter's noise model: SymForce has no separate
    "information matrix" object, the weighting lives in the residual itself.
    """
    a, b, c = params[0], params[1], params[2]
    return sf.V1(inv_sigma * (sf.exp(a * x * x + b * x + c) - y))


def chi_squared(abc, xs, ys):
    """chi2 = sum_i ((y_i - exp(a x_i^2 + b x_i + c)) / sigma)^2.

    Computed here rather than taken from the optimizer: SymForce reports
    0.5 * sum of squares, other libraries report other conventions, and the
    chapters must plot the same number.
    """
    a, b, c = abc
    r = (ys - np.exp(a * xs * xs + b * xs + c)) / SIGMA
    return float(np.dot(r, r))


def main():
    print("=== SymForce Tutorial: Curve Fitting ===\n")

    rng = np.random.default_rng(42)
    xs = np.arange(N) / N
    ys = np.exp(GT[0] * xs * xs + GT[1] * xs + GT[2]) + rng.normal(0, SIGMA, N)

    viz = Viz(CURVE_FITTING_RECORDING, "symforce")
    viz.curve_setup(xs, ys, GT, INIT)

    values = Values()
    values["params"] = sf.V3(*INIT)
    values["inv_sigma"] = 1.0 / SIGMA
    values["epsilon"] = sf.numeric_epsilon
    factors = []
    for i in range(N):
        values[f"x{i}"] = float(xs[i])
        values[f"y{i}"] = float(ys[i])
        factors.append(
            Factor(
                residual=curve_residual,
                keys=["params", f"x{i}", f"y{i}", "inv_sigma", "epsilon"],
            )
        )

    # debug_stats=True keeps the Values of every iteration so each step can be
    # streamed; early_exit_min_reduction stops as soon as LM stops improving
    # instead of burning the whole budget.
    params = Optimizer.Params(
        iterations=MAX_ITERATIONS,
        debug_stats=True,
        verbose=False,
        early_exit_min_reduction=1e-6,
    )
    optimizer = Optimizer(factors=factors, optimized_keys=["params"], params=params)
    print("Optimizing (the first run compiles the symbolic residual)...")
    result = optimizer.optimize(values)

    # result.iterations[0] is the initial state, not a step (SymForce numbers it
    # -1); it is streamed as frame 0 so the timeline matches the sibling
    # chapters. SymForce has no per-iteration callback, so the frames are
    # replayed from the stats debug_stats kept rather than logged during the run.
    for step, it in enumerate(result.iterations):
        v = optimizer.load_iteration_values(it.values)
        abc = np.array(v["params"], dtype=float).flatten()
        viz.curve_iteration(step, abc, chi_squared(abc, xs, ys))

    est = np.array(result.optimized_values["params"], dtype=float).flatten()
    chi2_init = chi_squared(INIT, xs, ys)
    chi2_final = chi_squared(est, xs, ys)

    print(f"\nGround truth : a={GT[0]:.4f} b={GT[1]:.4f} c={GT[2]:.4f}")
    print(f"Initial guess: a={INIT[0]:.4f} b={INIT[1]:.4f} c={INIT[2]:.4f}")
    print(f"Estimated    : a={est[0]:.4f} b={est[1]:.4f} c={est[2]:.4f}")
    print(f"\nSolver: Levenberg-Marquardt, {len(result.iterations) - 1} iterations "
          f"(max {MAX_ITERATIONS}), status {result.status.name}")
    print(f"Initial chi2: {chi2_init:.4f}")
    print(f"Final chi2:   {chi2_final:.4f}")
    print(f"Reduction:    {(1 - chi2_final / chi2_init) * 100:.4f}%")


if __name__ == "__main__":
    main()
