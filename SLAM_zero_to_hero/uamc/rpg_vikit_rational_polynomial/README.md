# Vikit: Vision-Kit for Robotics

Vikit is a versatile collection of C++ tools and utilities designed for computer vision and robotics projects. This version has been modernized for the ROS2 ecosystem (specifically tested on Jazzy and Humble) with a focus on high-performance parameter handling, cross-node compatibility, and extended camera model support including the OpenCV rational polynomial distortion model.

---

## 🔒 Disclaimer & Acknowledgments

### Usage Policy
This software is provided for **educational and research purposes only**. It shall **not be used for any commercial purposes**.

### Acknowledgments
We extend our deepest gratitude to the original developers and contributors of the projects that served as the foundation for this kit:
*   [uzh-rpg/rpg_vikit](https://github.com/uzh-rpg/rpg_vikit)
*   [xuankuzcr/rpg_vikit](https://github.com/xuankuzcr/rpg_vikit)
*   [uavfly/vikit](https://github.com/uavfly/vikit)
*   [Robotic-Developer-Road/rpg_vikit](https://github.com/Robotic-Developer-Road/rpg_vikit.git)

---

## 🚀 Key Improvements in ROS2

### Optimized Parameter Fetching Architecture
One of the most significant challenges in migrating from ROS1 to ROS2 is the removal of the global Parameter Server. In ROS2, parameters are local to each node. Vikit addresses this through a multi-tiered fetching strategy in `params_helper.hpp`:

1.  **Native Node Access**: Direct, high-speed access to parameters owned by the current node handle.
2.  **`SyncParametersClient` (High Performance)**: For cross-node parameter access (e.g., retrieving camera intrinsics from a central `parameter_blackboard`). This utilizes optimized ROS2 Service calls to achieve microsecond-level latency, avoiding the overhead of CLI tools.
3.  **Command-Line Fallback**: A robust fallback mechanism using `popen` to interface with the ROS2 CLI (`ros2 param get`), ensuring parameter retrieval even in complex edge cases where service clients might be restricted.

This architecture ensures that vision components can load dozens of camera parameters nearly instantaneously, a critical requirement for real-time SLAM and VIO systems.

---

## 📷 OpenCV Rational Polynomial Camera Model

This fork adds a new `RationalPolynomialCamera` class implementing the OpenCV `CALIB_RATIONAL_MODEL` — a distortion model well-suited for wide-angle rectilinear lenses with complex distortion that standard Brown-Conrady cannot capture.

### Parameters

| Parameter | Count | Description |
|---|---|---|
| `fx, fy, cx, cy` | 4 | Intrinsics (focal lengths and principal point) |
| `k1, k2, k3` | 3 | Radial distortion numerator coefficients |
| `k4, k5, k6` | 3 | Radial distortion denominator coefficients |
| `p1, p2` | 2 | Tangential (decentering) distortion coefficients |

### Distortion Model

The radial distortion uses a **ratio of two 6th-order polynomials**, providing more flexibility than the numerator-only Brown-Conrady model:

```
              1 + k1*r^2 + k2*r^4 + k3*r^6
r_coeff  =  ---------------------------------
              1 + k4*r^2 + k5*r^4 + k6*r^6
```

Tangential distortion (p1, p2) is applied on top. The denominator is clamped to prevent division-by-zero.

### Projection Pipeline (3D -> Pixel)

1. **Perspective divide**: `x' = X/Z`, `y' = Y/Z`
2. **Rational radial distortion**: apply `r_coeff` scaling
3. **Tangential distortion**: add `p1/p2` decentering terms
4. **Intrinsics**: `u = fx*xd + cx`, `v = fy*yd + cy`

### Unprojection (Pixel -> Bearing Vector)

Iterative undistortion via Newton's method (10 iterations) with an analytical 2x2 Jacobian. Tested round-trip accuracy: **~7.3e-15 px** (machine epsilon).

### ROS 2 Camera Loader

Registered as `cam_model: "RationalPolynomial"` in the `camera_loader`. Example YAML:

```yaml
camera:
  cam_model: "RationalPolynomial"
  cam_width: 1920
  cam_height: 1080
  cam_fx: 1050.0
  cam_fy: 1050.0
  cam_cx: 960.0
  cam_cy: 540.0
  k1: -0.2586
  k2: 0.0623
  k3: 0.0
  k4: 0.0
  k5: 0.0
  k6: 0.0
  p1: 0.000446
  p2: -0.000270
```

### Supported Camera Models (Full List)

| Model | Class | Lens Type | Loader Name |
|---|---|---|---|
| Pinhole + Brown-Conrady | `PinholeCamera` | Standard | `"Pinhole"` |
| ATAN / FOV | `ATANCamera` | Wide-angle | `"ATAN"` |
| Equidistant (Kannala-Brandt) | `EquidistantCamera` | Fisheye | `"EquidistantCamera"` |
| Polynomial (angle-based) | `PolynomialCamera` | Fisheye | `"PolynomialCamera"` / `"FishPoly"` |
| Omnidirectional | `OmniCamera` | Omni | `"Ocam"` |
| **Rational Polynomial** | **`RationalPolynomialCamera`** | **Wide-angle rectilinear** | **`"RationalPolynomial"`** |

---

## 🔧 aarch64 / ARM Build Support

The CMake build system now correctly detects the CPU architecture via `CMAKE_SYSTEM_PROCESSOR` and applies appropriate compiler flags:

- **x86_64**: `-march=native -mmmx -msse -msse2 -msse3 -mssse3`
- **aarch64 (Jetson Orin, RPi, etc.)**: `-march=armv8-a`

The legacy `ARM_ARCHITECTURE` environment variable check has been removed. NEON SIMD is available for `halfSample` in `vision.cpp`; other routines use scalar C++ fallback on ARM.

---

## 🛠 Installation Guide

### Prerequisites: Sophus
Vikit relies on Sophus for Lie groups. It is recommended to use version `1.22.10`.

```bash
git clone https://github.com/strasdat/Sophus.git -b 1.22.10
cd Sophus && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### Building `vikit_common`
`vikit_common` is a pure CMake package and can be installed globally.

```bash
cd vikit_common
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### Building `vikit_ros`
`vikit_ros` is integrated into the ROS2 workspace and should be built using `colcon`.

```bash
# Move vikit_ros to your workspace src directory
cd ~/ros2_ws
colcon build --symlink-install --packages-select vikit_ros
```

---

## 📅 Maintenance Info
*   **Last Update**: March 2026
*   **Target Systems**: Ubuntu 22.04 (Humble) / 24.04 (Jazzy)
*   **Target Platforms**: x86_64, aarch64 (NVIDIA Jetson Orin, Raspberry Pi 4/5)
*   **Compiler**: C++17 compliant (GCC 9+)
