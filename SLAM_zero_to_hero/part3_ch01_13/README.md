# g2o: General Graph Optimization

[g2o](https://github.com/RainerKuemmerle/g2o) is a C++ framework for optimizing
graph-based nonlinear least squares (vertices = parameters, edges =
constraints), used in ORB-SLAM, LSD-SLAM and many others. This chapter
demonstrates the three canonical SLAM back-end problems with g2o, each with a
visualization.

| Example | Source | Output | Visualize with |
|---|---|---|---|
| Curve fitting | `examples/g2o_curve_fitting.cpp` | `curve_fitting.txt` | `viz/plot_curve_fitting.py` (matplotlib PNG) |
| Pose-graph optimization | `examples/g2o_pose_graph.cpp` | `pose_graph.txt` | `viz/plot_pose_graph.py` (matplotlib PNG) |
| Bundle adjustment (BAL, Trafalgar Square) | `examples/g2o_bundle_adjustment.cpp` | `bundle_adjustment.txt` | `viz/show_bundle_adjustment.py` (rerun 3D) |

---

## Build

Docker (recommended — `slam:base` ships g2o + rerun; matplotlib + an spdlog/fmt
fix are added):
```bash
docker build . -t slam_zero_to_hero:part3_ch01_13
docker run -it --rm slam_zero_to_hero:part3_ch01_13
```

Local (needs g2o + Eigen + spdlog):
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
./g2o_curve_fitting
python3 ../viz/plot_curve_fitting.py            # -> curve_fitting.png

# 2) 2D pose-graph optimization (square loop + loop closure)
./g2o_pose_graph
python3 ../viz/plot_pose_graph.py               # -> pose_graph.png

# 3) Bundle adjustment on the BAL dataset
./g2o_bundle_adjustment problem-21-11315-pre.txt
python3 ../viz/show_bundle_adjustment.py        # -> bundle_adjustment.rrd (one frame per iteration)
#   open it and scrub the 'iteration' timeline:  rerun bundle_adjustment.rrd
#   or stream live to a running viewer:           python3 ../viz/show_bundle_adjustment.py --connect
```

PNGs are written headlessly (matplotlib `Agg`); copy them out with `docker cp`.

---

## Code notes

- **Curve fitting** defines a custom `CurveVertex` (the 3 parameters) and a
  custom unary `CurveEdge` with an **analytic Jacobian** in `linearizeOplus()`;
  solved with a dense Gauss-Newton solver.
- **Pose graph** uses g2o's built-in `VertexSE2` / `EdgeSE2`. Odometry and loop
  measurements are derived from ground truth (`gt[i].inverse() * gt[j]`) so the
  graph is consistent; vertex 0 is fixed to remove gauge freedom.
- **Bundle adjustment** implements the BAL camera model with custom types
  (`VertexCamera` is 9-DoF: quaternion rotation with a proper SO(3) update in
  `oplusImpl`, translation, and `f, k1, k2`; `VertexPoint` is marginalized for
  the Schur complement). The reprojection edge uses
  `p' = f (1 + k1 r^2 + k2 r^4)(-P/P.z)`; Jacobians are numeric. Camera centers
  are recovered as `C = -R^T t`.

BAL format: `<#cameras> <#points> <#observations>`, observations
`<cam> <pt> <x> <y>`, then 9 params/camera, then 3 coords/point. More problems:
https://grail.cs.washington.edu/projects/bal/

---

## References

- [g2o GitHub](https://github.com/RainerKuemmerle/g2o)
- [BAL dataset](https://grail.cs.washington.edu/projects/bal/)
