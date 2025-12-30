# Eigen + Sophus: Lie Group and Lie Algebra

This tutorial covers **Eigen** for linear algebra and **Sophus** for Lie Group operations, which are fundamental for representing rotations and rigid body transformations in SLAM.

---

## What is Sophus?

[Sophus](https://github.com/strasdat/Sophus) is a C++ library for Lie Groups commonly used in robotics:

- **SO(3)**: 3D rotation group (rotation matrices, quaternions)
- **SE(3)**: 3D rigid body transformation group (rotation + translation)
- **SO(2)**: 2D rotation group
- **SE(2)**: 2D rigid body transformation group

---

## Key Concepts

### Lie Group and Lie Algebra

| Lie Group | Lie Algebra | Description |
|-----------|-------------|-------------|
| SO(3) | so(3) | 3D rotations |
| SE(3) | se(3) | 3D rigid transformations |

### Exponential and Log Maps

```
exp: Lie Algebra → Lie Group  (tangent space → manifold)
log: Lie Group → Lie Algebra  (manifold → tangent space)
```

### Why Use Lie Groups in SLAM?

1. **Proper composition**: Rotation matrices stay valid
2. **Optimization**: Work in tangent space (vector space)
3. **Perturbation**: Small updates via exponential map
4. **Interpolation**: Proper interpolation on manifold

---

## How to Build

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:1_21
docker run -it --rm slam_zero_to_hero:1_21
```

---

## Examples

### 1. Eigen Basics (`eigen_basics`)

Basic Eigen operations for rotation and transformation:
- Rotation matrices
- Quaternions
- Euler angles
- Transformation matrices

```bash
./eigen_basics
```

### 2. SO(3) Operations (`sophus_so3`)

3D rotation with Sophus:
- Creating SO(3) from various representations
- Exponential and log maps
- Rotation composition
- Interpolation (slerp)

```bash
./sophus_so3
```

### 3. SE(3) Operations (`sophus_se3`)

3D rigid body transformation:
- Creating SE(3) from rotation and translation
- Exponential and log maps
- Transformation composition
- Inverse transformation

```bash
./sophus_se3
```

### 4. Perturbation Model (`perturbation`)

How to apply small updates in optimization:
- Left perturbation: `T_new = exp(delta) * T`
- Right perturbation: `T_new = T * exp(delta)`
- Jacobian computation

```bash
./perturbation
```

---

## Common Operations

### SO(3) Cheat Sheet

```cpp
#include <sophus/so3.hpp>

// Create from rotation matrix
Sophus::SO3d R(rotation_matrix);

// Create from quaternion
Sophus::SO3d R(quaternion);

// Create from angle-axis
Sophus::SO3d R = Sophus::SO3d::exp(angle_axis_vector);

// Get rotation matrix
Eigen::Matrix3d mat = R.matrix();

// Get quaternion
Eigen::Quaterniond q = R.unit_quaternion();

// Log map (SO3 → so3)
Eigen::Vector3d omega = R.log();

// Composition
Sophus::SO3d R_ab = R_a * R_b;

// Inverse
Sophus::SO3d R_inv = R.inverse();
```

### SE(3) Cheat Sheet

```cpp
#include <sophus/se3.hpp>

// Create from rotation and translation
Sophus::SE3d T(R, t);

// Create from 4x4 matrix
Sophus::SE3d T(matrix4x4);

// Exponential map (se3 → SE3)
Sophus::SE3d T = Sophus::SE3d::exp(twist_vector);  // 6D vector

// Log map (SE3 → se3)
Eigen::Matrix<double, 6, 1> xi = T.log();

// Get components
Sophus::SO3d R = T.so3();
Eigen::Vector3d t = T.translation();

// Transform a point
Eigen::Vector3d p_world = T * p_local;

// Composition
Sophus::SE3d T_ac = T_ab * T_bc;

// Inverse
Sophus::SE3d T_inv = T.inverse();
```

---

## References

- [Sophus GitHub](https://github.com/strasdat/Sophus)
- [A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537)
- [Lie Groups for 2D and 3D Transformations](http://ethaneade.com/lie.pdf)
