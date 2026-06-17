#!/usr/bin/env python3
"""SymForce Tutorial: Curve Fitting

Fits y = exp(a*x^2 + b*x + c) to noisy data. SymForce builds the residual
symbolically and generates the Jacobian automatically. Dumps `curve_fitting.txt`
for viz/plot_curve_fitting.py.
"""
import symforce

symforce.set_epsilon_to_symbol()

import numpy as np
import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.values import Values


def curve_residual(
    params: sf.V3, x: sf.Scalar, y: sf.Scalar, epsilon: sf.Scalar
) -> sf.V1:
    """Residual exp(a x^2 + b x + c) - y for one observation."""
    a, b, c = params[0], params[1], params[2]
    return sf.V1(sf.exp(a * x * x + b * x + c) - y)


def main():
    print("=== SymForce Tutorial: Curve Fitting ===\n")

    gt = (1.0, 2.0, 1.0)
    init = (2.0, -1.0, 5.0)
    N, sigma = 100, 0.2
    rng = np.random.default_rng(42)
    xs = np.arange(N) / N
    ys = np.exp(gt[0] * xs * xs + gt[1] * xs + gt[2]) + rng.normal(0, sigma, N)

    values = Values()
    values["params"] = sf.V3(*init)
    values["epsilon"] = sf.numeric_epsilon
    factors = []
    for i in range(N):
        values[f"x{i}"] = float(xs[i])
        values[f"y{i}"] = float(ys[i])
        factors.append(
            Factor(residual=curve_residual, keys=["params", f"x{i}", f"y{i}", "epsilon"])
        )

    optimizer = Optimizer(factors=factors, optimized_keys=["params"])
    result = optimizer.optimize(values)

    p = np.array(result.optimized_values["params"]).flatten()
    print(f"Ground truth : a={gt[0]} b={gt[1]} c={gt[2]}")
    print(f"Initial guess: a={init[0]} b={init[1]} c={init[2]}")
    print(f"Estimated    : a={p[0]:.4f} b={p[1]:.4f} c={p[2]:.4f}")
    print(f"Initial error: {result.iterations[0].new_error:.4f}  "
          f"Final error: {result.error():.4f}")

    with open("curve_fitting.txt", "w") as f:
        f.write("model exp(a*x^2+b*x+c)\n")
        f.write(f"gt {gt[0]} {gt[1]} {gt[2]}\n")
        f.write(f"init {init[0]} {init[1]} {init[2]}\n")
        f.write(f"est {p[0]} {p[1]} {p[2]}\n")
        f.write(f"data {N}\n")
        for i in range(N):
            f.write(f"{xs[i]} {ys[i]}\n")
    print("\nWrote curve_fitting.txt -> visualize with viz/plot_curve_fitting.py")


if __name__ == "__main__":
    main()
