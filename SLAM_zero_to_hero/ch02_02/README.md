# Building C++ Libraries for SLAM

This tutorial covers how to build and use C++ libraries - an essential skill for SLAM development where you'll work with Eigen, OpenCV, PCL, g2o, GTSAM, and more.

## How to build

Dependencies: None (pure C++17)

Local build:
```bash
mkdir build
cd build
cmake ..
make -j
```

Docker build:
```bash
docker build . -t slam_zero_to_hero:1_10
```

## How to run

Local:
```bash
./build/use_static_lib
./build/use_shared_lib
./build/use_header_only
```

Docker:
```bash
docker run -it --rm slam_zero_to_hero:1_10
```

---

## Topics Covered

### 1. Library Types

| Type | Extension | Linking | Use Case |
|------|-----------|---------|----------|
| **Static** | `.a` (Linux), `.lib` (Windows) | Compile-time | Eigen, small libs |
| **Shared** | `.so` (Linux), `.dll` (Windows) | Run-time | OpenCV, PCL |
| **Header-only** | `.hpp` / `.h` | Include only | Eigen, Sophus |

### 2. CMake Concepts

- `add_library(name STATIC/SHARED sources...)` - Create a library
- `target_include_directories()` - Specify header locations
- `target_link_libraries()` - Link libraries together
- `find_package()` - Find installed libraries
- `PUBLIC/PRIVATE/INTERFACE` - Visibility of dependencies

### 3. Project Structure

```
project/
├── CMakeLists.txt
├── include/
│   └── mylib/
│       └── mylib.hpp      # Public headers
├── src/
│   └── mylib.cpp          # Implementation
└── examples/
    └── main.cpp           # Usage example
```

---

## Why This Matters for SLAM

- **Eigen**: Header-only library, just include and use
- **OpenCV**: Shared library, needs `find_package(OpenCV)` and linking
- **g2o/GTSAM**: Can be static or shared, need proper CMake setup
- **Your own SLAM modules**: Often organized as libraries for reusability
