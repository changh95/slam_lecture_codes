#!/usr/bin/env python3
"""Visualize a curve-fitting result produced by the *_curve_fitting example.

Reads the standard ``curve_fitting.txt`` dump (see format below) and renders a
matplotlib PNG showing the noisy observations together with the ground-truth,
initial-guess and fitted curves of the model ``y = exp(a*x^2 + b*x + c)``.

Data format (whitespace separated)::

    model exp(a*x^2+b*x+c)
    gt   <a> <b> <c>
    init <a> <b> <c>
    est  <a> <b> <c>
    data <N>
    <x> <y>
    ... (N rows)
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / Docker safe
import matplotlib.pyplot as plt


def parse(path):
    params = {}
    xs, ys = [], []
    with open(path) as f:
        n_data = 0
        reading_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if tok[0] in ("gt", "init", "est"):
                params[tok[0]] = [float(v) for v in tok[1:4]]
            elif tok[0] == "data":
                n_data = int(tok[1])
                reading_data = True
            elif reading_data:
                xs.append(float(tok[0]))
                ys.append(float(tok[1]))
    return params, np.array(xs), np.array(ys), n_data


def curve(abc, x):
    a, b, c = abc
    return np.exp(a * x * x + b * x + c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="curve_fitting.txt")
    ap.add_argument("-o", "--out", default="curve_fitting.png")
    args = ap.parse_args()

    try:
        params, xs, ys, _ = parse(args.data)
    except FileNotFoundError:
        sys.exit(f"[plot_curve_fitting] cannot open {args.data}; run the "
                 f"curve_fitting example first to generate it.")

    xline = np.linspace(xs.min(), xs.max(), 400)

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=18, c="#444", alpha=0.6, label="observations")
    if "gt" in params:
        plt.plot(xline, curve(params["gt"], xline), "g-", lw=2,
                 label="ground truth %s" % params["gt"])
    if "init" in params:
        plt.plot(xline, curve(params["init"], xline), color="gray",
                 ls="--", lw=1.5, label="initial guess %s" % params["init"])
    if "est" in params:
        plt.plot(xline, curve(params["est"], xline), "r-", lw=2,
                 label="fitted %s" % [round(v, 3) for v in params["est"]])

    plt.title("Curve fitting:  y = exp(a x^2 + b x + c)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[plot_curve_fitting] wrote {args.out}")


if __name__ == "__main__":
    main()
