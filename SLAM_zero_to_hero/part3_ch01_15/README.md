# Ceres-Solver: Nonlinear Least Squares

[Ceres Solver](http://ceres-solver.org/) is Google's C++ library for large-scale
nonlinear least squares — automatic differentiation, robust losses and a sparse
Schur complement for bundle adjustment. This chapter solves the three canonical
SLAM back-end problems with Ceres, each streaming its own optimization live to a
rerun viewer.

The four solver chapters — g2o (`part3_ch01_13`), GTSAM (`part3_ch01_14`), Ceres
(this one) and SymForce (`part3_ch01_16`) — all solve the same three problems
from the same data and noise models under the same 30-iteration budget.
`part3_ch01_17` (Kimera-RPGO) contributes only the shared pose graph: it is an
outlier-rejection front end for a robust PGO backend, so it has no curve fitting
and no bundle adjustment, and it logs a 3-stage timeline rather than LM
iterations.

One caveat inside the four: chapter 16 draws its noise from numpy's PRNG instead
of `std::mt19937`, so its noise *realization* differs even though the seeds,
sigmas and budget match. Its chi-squared values are therefore close to but not
identical to the C++ chapters' — same model, same sigmas, different sample
values. The three C++ chapters (13, 14, 15) agree on the chi-squared values — 15.9973
initial for the pose graph, 84.1805 final for the curve fit — and 17 agrees on
the shared pose graph's initial 15.9973. Fitted parameters and final costs can
still differ in the last digit or two, because the libraries stop on different
tolerances: this chapter's curve fit ends at a=0.981348 c=0.994797 where g2o and
GTSAM report a=0.981351 c=0.994798, and its bundle adjustment ends at sq_error
303408 against their 303407. Same problem, one LM step apart.

Every chapter logs to the same rerun recording names under its own library name,
so running two of them against one viewer overlays their cost curves.

| Example | Source | Streams |
|---|---|---|
| Curve fitting | `examples/ceres_curve_fitting.cpp` | samples, ground-truth and fitted curve per iteration, `cost` (chi-squared) and `params/{a,b,c}` graphs |
| Pose-graph optimization | `examples/ceres_pose_graph.cpp` | ground truth / initial / optimized trajectory with heading arrows, loop-closure edge, `cost` graph |
| Bundle adjustment (BAL, Trafalgar Square) | `examples/ceres_bundle_adjustment.cpp` | landmark cloud + camera centres per iteration, `reprojection_error` and `rmse_px` graphs |

---

## Build

The base image `slam:base` must exist first — build it from the course-root
`SLAM_zero_to_hero/Dockerfile`, one level above this chapter (it ships Ceres 2.2,
Eigen, SuiteSparse and glog).

Both blocks below assume you start in this chapter's directory. Each
`docker build .` takes the *current* directory as its build context, so each
block has to `cd` to the directory holding the Dockerfile it is building.

```bash
cd ..                        # SLAM_zero_to_hero/, where the base Dockerfile lives
docker build . -t slam:base
```

Then this chapter, which adds the rerun C++ SDK 0.33.0 and downloads the BAL
problem (podman works too — substitute `podman` for `docker`):

```bash
cd part3_ch01_15
docker build . -t slam_zero_to_hero:part3_ch01_15
```

Local build (needs Ceres, Eigen, glog; the rerun C++ SDK is optional — without
it the demos build and print their numbers, they just do not stream):

```bash
mkdir build && cd build
cmake .. && make -j4
```

---

## Run

Start a viewer on the host first, then run the demos with `--network=host`:

```bash
# 1. Host: start the rerun viewer
rerun &

# 2. All three demos (all commands run from /workspace/part3_ch01_15/build,
#    which is the image's WORKDIR)
docker run -it --rm --network=host slam_zero_to_hero:part3_ch01_15 ./ceres_curve_fitting
docker run -it --rm --network=host slam_zero_to_hero:part3_ch01_15 ./ceres_pose_graph
docker run -it --rm --network=host slam_zero_to_hero:part3_ch01_15 ./ceres_bundle_adjustment
```

`--network=host` lets the container reach the viewer at `127.0.0.1:9876`. Live
gRPC streaming is version-sensitive: the container's rerun SDK **must match**
your host viewer's version (the Dockerfile pins `0.33.0` — set it to whatever
`rerun --version` prints on the host). Point the demos somewhere else with
`RERUN_URL`, e.g. `-e RERUN_URL='rerun+http://127.0.0.1:9999/proxy'`.

Streaming is automatic and never required: with no viewer listening each demo
prints a one-line note and runs exactly the same, printing all of its numbers to
stdout.

`ceres_bundle_adjustment` takes an optional BAL file path; with no argument it
uses `problem-21-11315-pre.txt`, which the image already downloaded into the
working directory.

---

## Output

Everything below is measured output from the commands above.

### Curve fitting

`y = exp(a x² + b x + c)`, ground truth `(1, 2, 1)`, initial guess `(2, -1, 5)`,
100 samples with `sigma = 0.2` noise (seed 42). Levenberg-Marquardt with
`DENSE_QR`:

```
Ground truth : a=1 b=2 c=1
Initial guess: a=2 b=-1 c=5
Initial chi2 : 8.00002e+07
Ceres Solver Report: Iterations: 8, Initial cost: 4.000008e+07, Final cost: 4.209025e+01, Termination: CONVERGENCE
Stopped because: Function tolerance reached. |cost_change|/cost: 1.311230e-10 <= 1.000000e-06
Estimated    : a=0.981348 b=2.02217 c=0.994797
Final chi2   : 84.1805
Iterations   : 7 (frame 0 is the initial state)
```

chi-squared falls from 8.0e7 to 84.18 in 7 iterations — about 0.84 per sample,
which is what 100 samples of `sigma = 0.2` noise costs at the optimum.

In the viewer: `curve/observations` and `curve/ground_truth` are static,
`curve/ceres/fitted` moves along the `iteration` timeline, and the `cost/ceres`
graph plots chi-squared per iteration next to `params/ceres/{a,b,c}`.

### Pose-graph optimization

Five poses around a unit square, four odometry edges plus the `x4 -> x0` loop
closure. The measurements are the *exact* relative transforms from ground truth
— only the initial estimate is noisy (seed 7, `sigma_xy = 0.15`,
`sigma_theta = 0.08`, poses 1..4) — so the optimum is the ground truth itself
and the demo isolates how the solver gets there:

```
Initial chi2 : 15.9973
Ceres Solver Report: Iterations: 4, Initial cost: 7.998670e+00, Final cost: 3.907016e-20, Termination: CONVERGENCE
Stopped because: Parameter tolerance reached. Relative step_norm: 9.990705e-12 <= 1.000000e-08.

Pose | ground truth        | optimized           | error
---------------------------------------------------------------
  x0 | ( 0.00, 0.00, 0.00) | ( 0.00, 0.00, 0.00) | 0.0000
  x1 | ( 1.00, 0.00, 0.00) | ( 1.00,-0.00, 0.00) | 0.0000
  x2 | ( 1.00, 1.00, 1.57) | ( 1.00, 1.00, 1.57) | 0.0000
  x3 | ( 0.00, 1.00, 3.14) | (-0.00, 1.00, 3.14) | 0.0000
  x4 | ( 0.00, 0.00,-1.57) | (-0.00,-0.00,-1.57) | 0.0000

Final chi2   : 7.81403e-20
Iterations   : 3 (frame 0 is the initial state)
```

chi-squared goes 15.997 -> 7.8e-20 in 3 iterations: with consistent
measurements the graph collapses onto ground truth to machine precision.

In the viewer: `graph/ground_truth` and `graph/ceres/initial` are static,
`graph/ceres/optimized` (positions, path and heading arrows) advances with the
`iteration` timeline, and `cost/ceres` graphs chi-squared. The heading arrows
matter here — pose 4 sits exactly on pose 0 and differs only in orientation, so
the loop closure is invisible in a position-only plot.

### Bundle adjustment

BAL Trafalgar Square, `problem-21-11315-pre.txt`: 21 cameras, 11315 points,
36455 observations. Intrinsics fixed, camera 0's pose fixed, no robust loss,
`SPARSE_SCHUR`:

```
Cameras: 21  Points: 11315  Observations: 36455
Intrinsics are FIXED (read from the dataset): camera 0 has f=2844.31 k1=-2.0201e-08 k2=2.12464e-15
Optimizing 126 pose parameters (camera 0 fixed) + 33945 point parameters
Initial sq_error: 8.82648e+06 px^2   RMSE: 15.5602 px
Ceres Solver Report: Iterations: 8, Initial cost: 4.413239e+06, Final cost: 1.517038e+05, Termination: CONVERGENCE
Stopped because: Function tolerance reached. |cost_change|/cost: 6.342519e-07 <= 1.000000e-06

Iterations : 7 (frame 0 is the initial state)
sq_error   : 8.82648e+06 -> 303408 px^2
RMSE       : 15.5602 -> 2.88493 px
Reduction  : 96.5625 % of sq_error, 81.4596 % of RMSE
```

`sq_error` is the raw sum of squared reprojection error over all observations and
`rmse_px = sqrt(sq_error / num_observations)` — per observation, not per residual
component. Both are recomputed from the parameters rather than read out of
`summary.final_cost`, so they mean the same thing in every chapter of this
series.

In the viewer: `world/initial_points` is the static starting cloud,
`world/ceres/landmarks` and `world/ceres/cameras` advance with the `iteration`
timeline (the cameras sit inside the cloud, since the centres are recovered as
`C = -Rᵀt`), and `reprojection_error/ceres` and `rmse_px/ceres` graph the two
metrics.

---

## Code notes

- **Auto-diff cost functions** — every example defines a templated `operator()`
  wrapped in `ceres::AutoDiffCostFunction<Functor, num_residuals, block_dims...>`;
  Ceres derives the Jacobians itself.
- **Weighting has no information matrix.** g2o and GTSAM take an information
  matrix or a `noiseModel`; Ceres has no such concept, so the residual is
  pre-multiplied by the square root of the information (i.e. divided by sigma)
  inside the functor. Curve fitting divides by `sigma = 0.2`; the pose graph
  scales `(x, y, theta)` by `(10, 10, 20)`, which is the `sigma = (0.1, 0.1,
  0.05)` / information `diag(100, 100, 400)` of the other chapters.
- **The reported cost is not the solver's cost.** Ceres' `summary.*_cost` is
  ½·Σr², and each library in this series scales its internal cost differently, so
  every example computes chi-squared (and, for BA, `sq_error`/`rmse_px`) from the
  parameters itself. That is what the graphs and the printed numbers use.
- **Per-iteration streaming** uses a `ceres::IterationCallback` together with
  `options.update_state_every_iteration = true` — without that flag Ceres keeps
  the parameter blocks stale until the solve finishes and the callback would log
  the same state every time. The callback fires once with `iteration == 0` for
  the initial state, so frame 0 in the viewer is the starting point, not the
  first step, and the reported iteration count is one less than
  `summary.iterations.size()`.
- **Gauge freedom** is removed with `problem.SetParameterBlockConstant(...)`:
  pose 0 in the pose graph, camera 0's pose in bundle adjustment. This is the
  cheapest of the mechanisms the sibling chapters use (g2o's `setFixed(true)` is
  the same idea; GTSAM instead adds a tight prior factor, and SymForce simply
  omits the key from `optimized_keys`) — the constant block leaves the linear
  system altogether rather than being pinned by an extra residual.
- **Intrinsics are fixed in bundle adjustment.** BAL cameras have 9 parameters
  (angle-axis, translation, `f, k1, k2`); only the first 6 are optimized. The
  three intrinsics are read from the dataset into a separate array and captured
  as constants by each observation's functor, which is why the cost function is
  `AutoDiffCostFunction<BALReprojectionError, 2, 6, 3>` and not `..., 2, 9, 3>`.
- **Scale stays free in bundle adjustment.** Fixing camera 0 removes 6 of the 7
  gauge degrees of freedom; the BAL projection is scale-invariant, so the seventh
  (global scale) remains. LM's damping term handles that singular direction —
  which is also why the run is stopped by a relative function-decrease test
  (`function_tolerance = 1e-6`, the same threshold the GTSAM chapter uses) rather
  than by a gradient test that would never fire, and why fixing a point as well
  would be wrong: that over-constrains the problem.
- **No robust loss.** Ceres ships `ceres::HuberLoss`, `CauchyLoss` and friends,
  and passing one as the second argument of `AddResidualBlock` is all it takes.
  It is deliberately off here so the reported error *is* the objective being
  minimized and the four solver chapters' numbers compare directly.
- **`google::InitGoogleLogging(argv[0])`** is called first in each example. Ceres
  is built against real glog here, so without it every internal `LOG(WARNING)`
  is prefixed with a "Logging before InitGoogleLogging()" complaint.
- **Levenberg-Marquardt everywhere**, `max_num_iterations = 30`, early stop on
  Ceres' own convergence tests — the shared budget of the four solver chapters.
  `ceres::DOGLEG` is the other trust-region strategy on offer.

BAL format: `<#cameras> <#points> <#observations>`, then observations
`<cam> <pt> <x> <y>`, then 9 parameters per camera, then 3 coordinates per point.
Projection: `P_cam = R X + t`, `p = (-P.x/P.z, -P.y/P.z)`,
`p' = f (1 + k1 r² + k2 r⁴) p`. More problems:
https://grail.cs.washington.edu/projects/bal/

---

## References

- [Ceres Solver](http://ceres-solver.org/) · [Tutorial](http://ceres-solver.org/tutorial.html)
- [BAL dataset](https://grail.cs.washington.edu/projects/bal/)
- [rerun](https://rerun.io/) — the viewer the demos stream to
