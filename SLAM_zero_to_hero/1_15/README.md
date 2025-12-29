# nanobind: Python-C++ Bindings for Robotics/SLAM

This tutorial demonstrates how to use [nanobind](https://github.com/wjakob/nanobind) to create Python bindings for C++ code, with a focus on robotics and SLAM applications.

## Why nanobind?

nanobind is the successor to pybind11, offering:
- **Smaller binaries** (~5x smaller than pybind11)
- **Faster compilation** (~2-3x faster)
- **Lower runtime overhead**
- **Zero-copy NumPy/Eigen interop** (critical for SLAM)

---

## Topics Covered

| Topic | Description |
|-------|-------------|
| Container Conversion | Python list/dict ↔ C++ vector/map |
| Eigen Zero-Copy | NumPy arrays ↔ Eigen matrices without copying |
| Class Bindings | Expose C++ classes (SE2 pose, Kalman filter) |
| Operators | Overload `*`, `[]`, `repr`, etc. |
| CMake Integration | Proper build configuration |

---

## How to Build

### Local Build

```bash
# Prerequisites: Python3, Eigen3, CMake
mkdir build && cd build
cmake ..
make -j4

# Set Python path
export PYTHONPATH=${PYTHONPATH}:$(pwd)

# Test
python3 -c "import slam_bindings; print(slam_bindings.__doc__)"
python3 ../examples/test_bindings.py
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:1_15
docker run -it --rm slam_zero_to_hero:1_15
```

---

## Key Concepts

### 1. Container Conversion (list ↔ vector)

C++:
```cpp
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>

double sum_vector(const std::vector<double>& vec) {
    double sum = 0.0;
    for (const auto& v : vec) sum += v;
    return sum;
}

NB_MODULE(my_module, m) {
    m.def("sum_vector", &sum_vector);
}
```

Python:
```python
import my_module
result = my_module.sum_vector([1.0, 2.0, 3.0, 4.0])
# result = 10.0
```

### 2. NumPy ↔ Eigen Zero-Copy

C++:
```cpp
#include <nanobind/eigen/dense.h>

// Eigen::Ref provides zero-copy access to numpy arrays
Eigen::Vector3d transform_point(
    const Eigen::Ref<const Eigen::Matrix3d>& R,
    const Eigen::Ref<const Eigen::Vector3d>& t,
    const Eigen::Ref<const Eigen::Vector3d>& p
) {
    return R * p + t;
}
```

Python:
```python
import numpy as np
import my_module

R = np.eye(3)  # Rotation matrix
t = np.array([1.0, 2.0, 3.0])  # Translation
p = np.array([0.5, 0.5, 0.5])  # Point

# No memory copy! Direct Eigen access to numpy data
result = my_module.transform_point(R, t, p)
```

**Important**: Use `dtype=np.float64` for double precision Eigen matrices.

### 3. Class Bindings

C++:
```cpp
class SE2 {
public:
    SE2(double x, double y, double theta);
    double x() const;
    SE2 compose(const SE2& other) const;
    std::string repr() const;
};

NB_MODULE(my_module, m) {
    nb::class_<SE2>(m, "SE2")
        .def(nb::init<double, double, double>())
        .def_prop_ro("x", &SE2::x)
        .def("compose", &SE2::compose)
        .def("__repr__", &SE2::repr)
        .def("__mul__", &SE2::compose);  // pose1 * pose2
}
```

Python:
```python
pose1 = my_module.SE2(1.0, 2.0, 0.5)
pose2 = my_module.SE2(0.5, 0.0, 0.1)

# Use compose method or * operator
result = pose1.compose(pose2)
result = pose1 * pose2

print(pose1)  # Uses __repr__
print(pose1.x)  # Read-only property
```

### 4. CMakeLists.txt Setup

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Fetch nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(nanobind)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Build module
nanobind_add_module(
    my_module
    STABLE_ABI
    src/bindings.cpp
)

target_include_directories(my_module PRIVATE ${EIGEN3_INCLUDE_DIR})
```

---

## Example: SE2 Pose for Mobile Robots

```python
import numpy as np
import slam_bindings as slam

# Create a robot pose (x, y, theta)
robot_pose = slam.SE2(0.0, 0.0, 0.0)

# Move forward 1m
motion = slam.SE2(1.0, 0.0, 0.0)
robot_pose = robot_pose * motion

# Turn 90 degrees
turn = slam.SE2(0.0, 0.0, np.pi/2)
robot_pose = robot_pose * turn

# Move forward again
robot_pose = robot_pose * motion

print(f"Final pose: {robot_pose}")
# SE2(x=1.000, y=1.000, theta=1.571)

# Transform a landmark from robot frame to world frame
landmark_robot = np.array([1.0, 0.0])
landmark_world = robot_pose.transform_point(landmark_robot)
```

---

## Example: Kalman Filter

```python
import slam_bindings as slam

# 1D Kalman filter for position tracking
# Initial: x=0, P=1, Q=0.1 (process noise), R=0.5 (measurement noise)
kf = slam.KalmanFilter1D(0.0, 1.0, 0.1, 0.5)

# Simulate measurements
measurements = [4.8, 5.2, 4.9, 5.1, 5.0]

for z in measurements:
    kf.predict()      # Predict next state
    kf.update(z)      # Update with measurement
    print(f"Measurement: {z:.1f}, Estimate: {kf.state:.3f}")
```

---

## Performance Tips

1. **Use Eigen::Ref for zero-copy**
   ```cpp
   // Good: zero-copy from numpy
   void process(const Eigen::Ref<const Eigen::MatrixXd>& data);

   // Bad: copies data
   void process(Eigen::MatrixXd data);
   ```

2. **Batch operations in C++**
   ```cpp
   // Process many points at once instead of one-by-one
   Eigen::MatrixXd transform_points_batch(
       const Eigen::Ref<const Eigen::Matrix3d>& R,
       const Eigen::Ref<const Eigen::Vector3d>& t,
       const Eigen::Ref<const Eigen::MatrixXd>& points  // 3xN
   );
   ```

3. **Avoid converting large containers**
   - Pass numpy arrays directly using Eigen
   - Only use std::vector for small data

---

## Common Issues

**Problem**: `TypeError: incompatible function arguments`
**Solution**: Use `dtype=np.float64` for numpy arrays

**Problem**: Module not found
**Solution**: Set PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)/build`

**Problem**: Eigen version mismatch
**Solution**: Ensure consistent Eigen version between builds

---

## References

- [nanobind Documentation](https://nanobind.readthedocs.io/)
- [Eigen-nanobind Integration](https://nanobind.readthedocs.io/en/latest/eigen.html)
- [pybind11 to nanobind Migration](https://nanobind.readthedocs.io/en/latest/migrating.html)
