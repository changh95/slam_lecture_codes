# GTSAM: Georgia Tech Smoothing and Mapping

This tutorial covers [GTSAM](https://gtsam.org/), a factor graph-based optimization library widely used in robotics and SLAM.

---

## What is GTSAM?

GTSAM is a C++ library that implements:

- **Factor Graphs**: Probabilistic graphical models for optimization
- **Smoothing and Mapping**: State estimation for robotics
- **iSAM2**: Incremental smoothing for real-time SLAM

Used in production systems like Boston Dynamics' Spot robot.

---

## Key Concepts

### Factor Graphs

```
Variables (X):
  - Robot poses
  - 3D landmarks
  - Calibration parameters

Factors (constraints):
  - Prior factors: Initial knowledge
  - Between factors: Odometry
  - Projection factors: Camera observations
```

### GTSAM vs g2o

| Feature | GTSAM | g2o |
|---------|-------|-----|
| API Style | Factor-based | Vertex/Edge-based |
| Incremental | iSAM2 | Not built-in |
| Documentation | Excellent | Limited |
| Python bindings | Native | Limited |

---

## How to Build

Local build:
```bash
mkdir build && cd build
cmake ..
make -j4
```

Docker build:
```bash
docker build . -t slam_zero_to_hero:3_14
```

---

## Core API

### Creating a Factor Graph

```cpp
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

NonlinearFactorGraph graph;
Values initial_estimate;
```

### Adding Factors

```cpp
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

// Prior factor (anchor)
auto noise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));
graph.addPrior(Symbol('x', 0), Pose2(0, 0, 0), noise);

// Between factor (odometry)
graph.add(BetweenFactor<Pose2>(
    Symbol('x', 0), Symbol('x', 1),
    Pose2(1.0, 0.0, 0.0),  // measurement
    noise
));
```

### Adding Variables

```cpp
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose2.h>

// Using Symbol for named keys
initial_estimate.insert(Symbol('x', 0), Pose2(0, 0, 0));
initial_estimate.insert(Symbol('x', 1), Pose2(1, 0, 0));

// Or using shorthand
using namespace gtsam::symbol_shorthand;
initial_estimate.insert(X(0), Pose2(0, 0, 0));
```

### Optimization

```cpp
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

LevenbergMarquardtParams params;
params.setVerbosity("SUMMARY");
params.setMaxIterations(100);

LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
Values result = optimizer.optimize();

// Access results
Pose2 pose = result.at<Pose2>(Symbol('x', 0));
```

### Marginal Covariances

```cpp
#include <gtsam/nonlinear/Marginals.h>

Marginals marginals(graph, result);
Matrix cov = marginals.marginalCovariance(Symbol('x', 0));
```

---

## Common Factor Types

| Factor | Description |
|--------|-------------|
| `PriorFactor<T>` | Prior knowledge on variable |
| `BetweenFactor<T>` | Relative constraint between poses |
| `GenericProjectionFactor` | 3D point to 2D image projection |
| `RangeFactor` | Range measurement |
| `BearingFactor` | Bearing measurement |

---

## Geometry Types

| Type | Description |
|------|-------------|
| `Pose2` | 2D pose (x, y, theta) |
| `Pose3` | 3D pose (SE3) |
| `Point2` | 2D point |
| `Point3` | 3D point |
| `Rot2` | 2D rotation |
| `Rot3` | 3D rotation (SO3) |

---

## Noise Models

```cpp
// Diagonal (independent noise per dimension)
auto noise = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.05));

// Isotropic (same noise all dimensions)
auto noise = noiseModel::Isotropic::Sigma(3, 0.1);

// Robust (outlier rejection)
auto huber = noiseModel::Robust::Create(
    noiseModel::mEstimator::Huber::Create(1.345),
    noiseModel::Isotropic::Sigma(2, 1.0)
);
```

---

## iSAM2 (Incremental Optimization)

```cpp
#include <gtsam/nonlinear/ISAM2.h>

// Create iSAM2 instance
ISAM2Params params;
params.relinearizeThreshold = 0.01;
ISAM2 isam(params);

// Incremental update
NonlinearFactorGraph new_factors;
Values new_values;
// ... add new factors and values ...

isam.update(new_factors, new_values);
Values current_estimate = isam.calculateEstimate();
```

---

## Bundle Adjustment Example

```cpp
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/geometry/Cal3Bundler.h>

// Camera intrinsics (f, k1, k2, u0, v0)
Cal3Bundler K(500.0, 0.0, 0.0, 320.0, 240.0);

// Projection factor
graph.emplace_shared<GeneralSFMFactor2<Cal3Bundler>>(
    Point2(100, 200),           // measured pixel
    measurement_noise,
    Symbol('c', camera_id),     // camera pose
    Symbol('p', point_id),      // 3D point
    Symbol('K', camera_id)      // intrinsics
);
```

---

## Running Examples

```bash
# 2D pose graph
./gtsam_basics

# Bundle adjustment with BAL dataset
./gtsam_bundle_adjustment problem-49-7776-pre.txt
```

Docker:
```bash
docker run -it --rm slam_zero_to_hero:3_14
```

---

## Tips

1. **Use Symbols** for readable variable names: `Symbol('x', 0)` instead of raw integers
2. **Start with priors** to anchor the optimization
3. **Check marginals** to verify optimization quality
4. **Use iSAM2** for real-time applications
5. **Leverage robust noise models** for outlier handling

---

## References

- [GTSAM Website](https://gtsam.org/)
- [GTSAM Tutorial](https://gtsam.org/tutorials/intro.html)
- [Factor Graphs for Robot Perception](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf)
