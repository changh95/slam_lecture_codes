# SymForce: Symbolic Computing for Robotics

This tutorial covers [SymForce](https://github.com/symforce-org/symforce), a symbolic computing library designed for robotics applications, developed by Skydio.

---

## What is SymForce?

SymForce combines:

- **Symbolic Math**: Build expressions symbolically
- **Code Generation**: Generate optimized C++/Python code
- **Optimization**: Factor graph optimization built-in
- **Geometry**: SE2, SE3, cameras, etc.

Used in Skydio's autonomous drones for real-time state estimation.

---

## Key Features

| Feature | Description |
|---------|-------------|
| Symbolic Differentiation | Automatic Jacobians |
| Code Generation | C++, Python, CUDA |
| Tangent Space | Proper manifold handling |
| Factor Graphs | Built-in optimization |

---

## Installation

```bash
pip install symforce
```

---

## Core API

### Symbolic Variables

```python
import symforce.symbolic as sf

x = sf.Symbol("x")
y = sf.Symbol("y")

expr = x**2 + y**2
print(expr.diff(x))  # 2*x
```

### Geometry Types

```python
# 2D Pose
pose2 = sf.Pose2(
    R=sf.Rot2.from_angle(sf.pi / 4),
    t=sf.V2([1.0, 2.0])
)

# 3D Pose
pose3 = sf.Pose3(
    R=sf.Rot3.from_yaw_pitch_roll(yaw, pitch, roll),
    t=sf.V3([x, y, z])
)

# Composition
result = pose_a * pose_b

# Inverse
inv_pose = pose.inverse()

# Tangent space
tangent = pose.to_tangent()
pose = sf.Pose3.from_tangent(tangent)
```

### Residual Functions

```python
def odometry_residual(
    pose_a: sf.Pose2,
    pose_b: sf.Pose2,
    measured: sf.Pose2,
    epsilon: sf.Scalar,
) -> sf.V3:
    """Compute residual for odometry constraint."""
    predicted = pose_a.inverse() * pose_b
    error = measured.inverse() * predicted
    return error.to_tangent(epsilon=epsilon)
```

### Factor Graph Optimization

```python
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.values import Values

# Create factors
factors = [
    Factor(
        residual=prior_residual,
        keys=["pose_0", "prior", "epsilon"],
    ),
    Factor(
        residual=odometry_residual,
        keys=["pose_0", "pose_1", "odom_01", "epsilon"],
    ),
]

# Initial values
initial_values = Values()
initial_values["pose_0"] = sf.Pose2.identity()
initial_values["pose_1"] = sf.Pose2(R=sf.Rot2.identity(), t=sf.V2([1.0, 0.0]))
initial_values["epsilon"] = sf.epsilon()

# Optimize
optimizer = Optimizer(
    factors=factors,
    optimized_keys=["pose_0", "pose_1"],
)

result = optimizer.optimize(initial_values)
print(f"Final error: {result.final_error}")
```

### Code Generation

```python
from symforce.codegen import Codegen, CppConfig

# Generate C++ code
codegen = Codegen.function(
    func=odometry_residual,
    config=CppConfig(),
)

# Generate to files
codegen.generate_function(output_dir="gen")
```

---

## Geometry Types

| Type | Description |
|------|-------------|
| `sf.Rot2` | 2D rotation |
| `sf.Rot3` | 3D rotation (quaternion) |
| `sf.Pose2` | 2D pose (x, y, theta) |
| `sf.Pose3` | 3D pose (SE3) |
| `sf.V2`, `sf.V3` | 2D/3D vectors |
| `sf.M33`, `sf.M44` | Matrices |

---

## Cameras

```python
# Linear camera
camera = sf.LinearCameraCal(
    focal_length=[fx, fy],
    principal_point=[cx, cy],
)

# Fisheye camera
camera = sf.EquirectangularCameraCal(
    focal_length=[fx, fy],
    principal_point=[cx, cy],
)

# Project point
pixel = camera.pixel_from_camera_point(point_3d)

# Unproject
ray = camera.camera_ray_from_pixel(pixel)
```

---

## Comparison with Other Libraries

| Feature | SymForce | Ceres | GTSAM |
|---------|----------|-------|-------|
| Symbolic | Yes | No | No |
| Code Gen | Yes | No | No |
| Python | Native | Limited | Bindings |
| Auto Diff | Symbolic | Numeric | Numeric |

---

## Running Examples

```bash
# Python example
python3 examples/symforce_basics.py
```

Docker:
```bash
docker build . -t slam_zero_to_hero:3_16
docker run -it --rm slam_zero_to_hero:3_16
```

---

## Use Cases

1. **SLAM Back-end**: Pose graph, bundle adjustment
2. **State Estimation**: EKF, UKF with symbolic Jacobians
3. **Motion Planning**: Trajectory optimization
4. **Calibration**: Camera, IMU, extrinsics

---

## Tips

1. **Use epsilon** for numerical stability near singularities
2. **Generate code** for production - symbolic is for development
3. **Leverage tangent space** for proper manifold optimization
4. **Check Values** type for organizing optimization variables

---

## References

- [SymForce GitHub](https://github.com/symforce-org/symforce)
- [SymForce Paper](https://arxiv.org/abs/2204.07889)
- [SymForce Documentation](https://symforce.org/)
