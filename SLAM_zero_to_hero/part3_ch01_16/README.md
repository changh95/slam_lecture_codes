# SymForce: Symbolic Computing for Robotics

[SymForce](https://github.com/symforce-org/symforce) (Skydio) builds residuals
symbolically and generates fast Jacobians / code automatically, with a built-in
factor-graph optimizer. This chapter demonstrates the three canonical SLAM
back-end problems with SymForce, each with a visualization. The examples are
pure Python.

| Example | Source | Output | Visualize with |
|---|---|---|---|
| Curve fitting | `examples/symforce_curve_fitting.py` | `curve_fitting.txt` | `viz/plot_curve_fitting.py` (matplotlib PNG) |
| Pose-graph optimization | `examples/symforce_pose_graph.py` | `pose_graph.txt` | `viz/plot_pose_graph.py` (matplotlib PNG) |
| Bundle adjustment (BAL, Trafalgar Square) | `examples/symforce_bundle_adjustment.py` | `bundle_adjustment.txt` | `viz/show_bundle_adjustment.py` (rerun 3D) |

---

## Build

Docker (recommended — `slam:base` ships SymForce + rerun; matplotlib is added):
```bash
docker build . -t slam_zero_to_hero:part3_ch01_16
docker run -it --rm slam_zero_to_hero:part3_ch01_16
```

The image lands you in `run/`, which already contains the BAL problem
`problem-21-11315-pre.txt`. (Nothing to compile — SymForce examples are Python;
`pip install symforce matplotlib` to run locally.)

---

## Run + visualize

All commands are run from `run/` (examples are at `../examples`, viz at `../viz`).

```bash
# 1) Curve fitting:  y = exp(a x^2 + b x + c)
python3 ../examples/symforce_curve_fitting.py
python3 ../viz/plot_curve_fitting.py            # -> curve_fitting.png

# 2) 2D pose-graph optimization (square loop + loop closure)
python3 ../examples/symforce_pose_graph.py
python3 ../viz/plot_pose_graph.py               # -> pose_graph.png

# 3) Bundle adjustment on the BAL dataset (full problem; takes a while to build)
python3 ../examples/symforce_bundle_adjustment.py problem-21-11315-pre.txt
python3 ../viz/show_bundle_adjustment.py        # -> bundle_adjustment.rrd (one frame per iteration)
#   open it and scrub the 'iteration' timeline:  rerun bundle_adjustment.rrd
#   or stream live to a running viewer:           python3 ../viz/show_bundle_adjustment.py --connect
```

The first SymForce run compiles the symbolic residual (a few seconds) before
optimizing. PNGs are written headlessly; copy them out with `docker cp`.

---

## Code notes

- **Residuals are symbolic Python functions.** Each `Factor` references its
  variables by key; SymForce derives the Jacobian from the symbolic residual and
  JIT-compiles it once, then reuses it for every observation.
- **Curve fitting** optimizes a single `sf.V3` of `(a, b, c)`, one factor per
  data point.
- **Pose graph** uses `sf.Pose2` with a `prior_residual` anchor and a
  `between_residual` per odometry/loop edge (measurements from ground truth);
  residuals are returned in the tangent space via `to_tangent`.
- **Bundle adjustment** models each camera as `sf.Pose3` (world-to-camera) plus
  an `sf.V3` calibration `(f, k1, k2)`, with the BAL reprojection
  `p' = f (1 + k1 r^2 + k2 r^4)(-P/P.z)`. The first camera is held fixed to
  remove gauge freedom. Camera centers are recovered via
  `pose.inverse().position()`.
  - **Full problem, no subsampling:** all cameras, points, and observations are
    used. The optimizer builds one Python factor per observation, so constructing
    the full Trafalgar problem (36455 observations) takes a while before the
    solver runs.

BAL dataset: https://grail.cs.washington.edu/projects/bal/

---

## References

- [SymForce GitHub](https://github.com/symforce-org/symforce) · [paper](https://arxiv.org/abs/2204.07889) · [docs](https://symforce.org/)
- [BAL dataset](https://grail.cs.washington.edu/projects/bal/)
