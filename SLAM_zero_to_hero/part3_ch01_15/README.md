# Ceres-Solver: Nonlinear Least Squares

[Ceres Solver](http://ceres-solver.org/) is Google's C++ library for large-scale
nonlinear least squares (auto-diff, robust losses, sparse Schur). This chapter
demonstrates the three canonical SLAM back-end problems with Ceres, each with a
visualization.

| Example | Source | Output | Visualize with |
|---|---|---|---|
| Curve fitting | `examples/ceres_curve_fitting.cpp` | `curve_fitting.txt` | `viz/plot_curve_fitting.py` (matplotlib PNG) |
| Pose-graph optimization | `examples/ceres_pose_graph.cpp` | `pose_graph.txt` | `viz/plot_pose_graph.py` (matplotlib PNG) |
| Bundle adjustment (BAL, Trafalgar Square) | `examples/ceres_bundle_adjustment.cpp` | `bundle_adjustment.txt` | `viz/show_bundle_adjustment.py` (rerun 3D) |

---

## Build

Docker (recommended — `slam:base` already ships Ceres + rerun, matplotlib is added):
```bash
docker build . -t slam_zero_to_hero:part3_ch01_15
docker run -it --rm slam_zero_to_hero:part3_ch01_15
```

Local (needs Ceres + Eigen):
```bash
mkdir build && cd build
cmake .. && make -j4
```

The image builds the three executables under `build/` and downloads the BAL
problem `problem-21-11315-pre.txt`.

---

## Run + visualize

All commands are run from `build/`. Each program writes a `*.txt` dump that the
matching `viz/` script renders.

```bash
# 1) Curve fitting:  y = exp(a x^2 + b x + c)
./ceres_curve_fitting
python3 ../viz/plot_curve_fitting.py            # -> curve_fitting.png

# 2) 2D pose-graph optimization (square loop + loop closure)
./ceres_pose_graph
python3 ../viz/plot_pose_graph.py               # -> pose_graph.png

# 3) Bundle adjustment on the BAL dataset
./ceres_bundle_adjustment problem-21-11315-pre.txt
python3 ../viz/show_bundle_adjustment.py        # -> bundle_adjustment.rrd (one frame per iteration)
#   open it and scrub the 'iteration' timeline:  rerun bundle_adjustment.rrd
#   stream live to a running viewer:              python3 ../viz/show_bundle_adjustment.py --connect
#   or open its own viewer (needs a display):     python3 ../viz/show_bundle_adjustment.py --spawn
```

PNGs are written headlessly (matplotlib `Agg`); copy them out of the container
with `docker cp` to view.

---

## Code notes

- **Auto-diff cost functions** — each example defines a templated `operator()`
  wrapped in `ceres::AutoDiffCostFunction<Functor, num_residuals, block_dims...>`;
  Ceres derives the Jacobians.
- **Curve fitting** packs `(a, b, c)` in one parameter block of size 3 and adds
  one residual per sample (`DENSE_QR`).
- **Pose graph** parameterizes each pose as `[x, y, theta]`, adds a
  `RelativeMotion` residual per odometry/loop edge, and fixes pose 0 with
  `SetParameterBlockConstant` to remove gauge freedom (`SPARSE_NORMAL_CHOLESKY`).
- **Bundle adjustment** uses the BAL 9-DoF camera (angle-axis, translation,
  `f, k1, k2`), a Huber-robust reprojection residual, and `SPARSE_SCHUR`. Camera
  centers are recovered as `C = -R^T t` for the 3D plot.

BAL format: `<#cameras> <#points> <#observations>`, then observations
`<cam> <pt> <x> <y>`, then 9 params per camera, then 3 coords per point. More
problems: https://grail.cs.washington.edu/projects/bal/

---

## References

- [Ceres Solver](http://ceres-solver.org/) · [Tutorial](http://ceres-solver.org/tutorial.html)
- [BAL dataset](https://grail.cs.washington.edu/projects/bal/)
