# Ceres-Solver: Nonlinear Least Squares

This tutorial covers [Ceres Solver](http://ceres-solver.org/), a powerful C++ library for nonlinear least squares optimization developed by Google.

---

## What is Ceres?

Ceres Solver is used for:

- **Bundle Adjustment**: Camera pose and 3D point optimization
- **SLAM Back-end**: Pose graph optimization
- **Sensor Calibration**: Camera intrinsics, extrinsics
- **General Optimization**: Curve fitting, parameter estimation

Used in Google Street View, Blender, and many SLAM systems.

---

## Key Features

| Feature | Description |
|---------|-------------|
| Automatic Differentiation | No manual Jacobians needed |
| Robust Loss Functions | Huber, Cauchy, etc. for outliers |
| Sparse Solvers | Efficient for large problems |
| Thread Safety | Multi-threaded optimization |

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
docker build . -t slam_zero_to_hero:3_15
```

---

## Core API

### 1. Define Cost Function

```cpp
// Using auto-differentiation (recommended)
struct MyCostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

// Create cost function
ceres::CostFunction* cost_function =
    new ceres::AutoDiffCostFunction<MyCostFunctor, 1, 1>(
        new MyCostFunctor);
// Template args: residual_dim, param1_dim, param2_dim, ...
```

### 2. Build Problem

```cpp
ceres::Problem problem;

double x = 5.0;  // Initial value

problem.AddResidualBlock(cost_function, nullptr, &x);
// nullptr = no loss function (L2 loss)
```

### 3. Configure and Solve

```cpp
ceres::Solver::Options options;
options.linear_solver_type = ceres::DENSE_QR;
options.minimizer_progress_to_stdout = true;

ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);

std::cout << summary.BriefReport() << std::endl;
```

---

## Loss Functions

For outlier robustness:

```cpp
// Huber loss
problem.AddResidualBlock(
    cost_function,
    new ceres::HuberLoss(1.0),  // delta = 1.0
    &x);

// Other options
new ceres::CauchyLoss(1.0);    // Cauchy
new ceres::SoftLOneLoss(1.0);  // Soft L1
new ceres::ArctanLoss(1.0);    // Arctan
```

---

## Linear Solvers

| Solver | Best For |
|--------|----------|
| `DENSE_QR` | Small, dense problems |
| `DENSE_SCHUR` | BA with few cameras |
| `SPARSE_SCHUR` | BA with many cameras |
| `SPARSE_NORMAL_CHOLESKY` | General sparse |
| `CGNR` | Very large problems |

```cpp
options.linear_solver_type = ceres::SPARSE_SCHUR;
options.num_threads = 4;  // Multi-threading
```

---

## Rotation Helpers

```cpp
#include <ceres/rotation.h>

// Angle-axis rotation
double angle_axis[3] = {0.1, 0.2, 0.3};
double point[3] = {1.0, 0.0, 0.0};
double result[3];

ceres::AngleAxisRotatePoint(angle_axis, point, result);

// Conversions
double quaternion[4];
ceres::AngleAxisToQuaternion(angle_axis, quaternion);
ceres::QuaternionToAngleAxis(quaternion, angle_axis);

double R[9];  // Row-major
ceres::AngleAxisToRotationMatrix(angle_axis, R);
```

---

## Bundle Adjustment Example

```cpp
struct ReprojectionError {
    ReprojectionError(double u, double v) : u_(u), v_(v) {}

    template <typename T>
    bool operator()(const T* const camera,  // 6: rotation(3) + translation(3)
                    const T* const point,   // 3: x, y, z
                    const T* const focal,   // 1: focal length
                    T* residual) const {
        // Rotate and translate
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Project
        T predicted_u = focal[0] * p[0] / p[2];
        T predicted_v = focal[0] * p[1] / p[2];

        residual[0] = predicted_u - u_;
        residual[1] = predicted_v - v_;

        return true;
    }

private:
    double u_, v_;
};
```

---

## Common Patterns

### Fix Parameters

```cpp
problem.SetParameterBlockConstant(&x);
```

### Parameter Bounds

```cpp
problem.SetParameterLowerBound(&x, 0, 0.0);  // x[0] >= 0
problem.SetParameterUpperBound(&x, 0, 1.0);  // x[0] <= 1
```

### Local Parameterization (Manifolds)

```cpp
// For quaternions (4 params, 3 DOF)
ceres::Manifold* quaternion_manifold = new ceres::QuaternionManifold;
problem.SetManifold(quaternion, quaternion_manifold);
```

---

## Running Examples

```bash
# Basic examples
./ceres_basics

# Bundle adjustment with BAL dataset
./ceres_bundle_adjustment problem-49-7776-pre.txt
```

Docker:
```bash
docker run -it --rm slam_zero_to_hero:3_15
```

---

## Comparison with Other Libraries

| Library | Strength |
|---------|----------|
| Ceres | General optimization, auto-diff |
| g2o | Graph-specific, fast for SLAM |
| GTSAM | Factor graphs, incremental |

---

## Tips

1. **Use AutoDiff** - avoids manual Jacobian errors
2. **Start with DENSE_QR** - then switch to sparse for large problems
3. **Add robust loss** - always use Huber for real data
4. **Check residuals** - before and after optimization
5. **Fix gauge freedom** - anchor at least one parameter

---

## References

- [Ceres Solver](http://ceres-solver.org/)
- [Ceres Tutorial](http://ceres-solver.org/tutorial.html)
- [BAL Dataset](https://grail.cs.washington.edu/projects/bal/)
