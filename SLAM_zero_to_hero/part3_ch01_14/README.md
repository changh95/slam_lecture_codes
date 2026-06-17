# GTSAM: Georgia Tech Smoothing and Mapping

[GTSAM](https://gtsam.org/) is a factor-graph optimization library widely used
in robotics and SLAM (variables = unknowns, factors = constraints; iSAM2 for
incremental solving). This chapter demonstrates the three canonical SLAM
back-end problems with GTSAM, each with a visualization.

| Example | Source | Output | Visualize with |
|---|---|---|---|
| Curve fitting | `examples/gtsam_curve_fitting.cpp` | `curve_fitting.txt` | `viz/plot_curve_fitting.py` (matplotlib PNG) |
| Pose-graph optimization | `examples/gtsam_pose_graph.cpp` | `pose_graph.txt` | `viz/plot_pose_graph.py` (matplotlib PNG) |
| Bundle adjustment (BAL, Trafalgar Square) | `examples/gtsam_bundle_adjustment.cpp` | `bundle_adjustment.txt` | `viz/show_bundle_adjustment.py` (rerun 3D) |

---

## Build

Docker (recommended — `slam:base` ships GTSAM + rerun; matplotlib is added):
```bash
docker build . -t slam_zero_to_hero:part3_ch01_14
docker run -it --rm slam_zero_to_hero:part3_ch01_14
```

Local (needs GTSAM):
```bash
mkdir build && cd build
cmake .. && make -j4
```

The image builds the three executables under `build/` and downloads the BAL
problem `problem-21-11315-pre.txt`.

---

## Run + visualize

All commands are run from `build/`.

```bash
# 1) Curve fitting:  y = exp(a x^2 + b x + c)
./gtsam_curve_fitting
python3 ../viz/plot_curve_fitting.py            # -> curve_fitting.png

# 2) 2D pose-graph optimization (square loop + loop closure)
./gtsam_pose_graph
python3 ../viz/plot_pose_graph.py               # -> pose_graph.png

# 3) Bundle adjustment on the BAL dataset
./gtsam_bundle_adjustment problem-21-11315-pre.txt
python3 ../viz/show_bundle_adjustment.py        # -> bundle_adjustment.rrd (one frame per iteration)
#   open it and scrub the 'iteration' timeline:  rerun bundle_adjustment.rrd
#   or stream live to a running viewer:           python3 ../viz/show_bundle_adjustment.py --connect
```

PNGs are written headlessly (matplotlib `Agg`); copy them out with `docker cp`.

---

## Code notes

- **Curve fitting** defines a custom `CurveFactor : NoiseModelFactorN<Vector3>`
  over a single `(a, b, c)` variable, supplying the analytic Jacobian in
  `evaluateError`; solved with Levenberg-Marquardt.
- **Pose graph** uses `Pose2` variables with a `PriorFactor` anchor and
  `BetweenFactor<Pose2>` odometry/loop measurements derived from ground truth
  (`gt[i].between(gt[j])`).
- **Bundle adjustment** loads the problem with GTSAM's own `SfmData::FromBalFile`
  (which handles the BAL camera/sign convention), then builds
  `GeneralSFMFactor<SfmCamera, Point3>` reprojection factors over
  `PinholeCamera<Cal3Bundler>` cameras (pose + `f, k1, k2`) and `Point3`
  landmarks. Priors on camera 0 and point 0 fix the gauge. GTSAM stores
  camera-to-world poses, so the camera center is `camera.translation()`.

BAL dataset: https://grail.cs.washington.edu/projects/bal/

---

## References

- [GTSAM website](https://gtsam.org/) · [intro tutorial](https://gtsam.org/tutorials/intro.html)
- [BAL dataset](https://grail.cs.washington.edu/projects/bal/)
