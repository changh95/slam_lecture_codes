# Basic C++ Programming for SLAM

This tutorial covers fundamental C++ programming concepts essential for SLAM development.

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
docker build . -t slam_zero_to_hero:1_9
```

## How to run

Local:
```bash
./build/for_loop
./build/while_loop
./build/vector
./build/map
./build/unordered_map
./build/template_function
./build/template_class
./build/smart_pointers
```

Docker:
```bash
docker run -it --rm slam_zero_to_hero:1_9
```

---

## Topics Covered

### 1. Control Flow
- **for_loop**: Different styles of for loops (index-based, range-based)
- **while_loop**: While and do-while loops

### 2. STL Containers
- **vector**: Dynamic arrays with push_back, pop_back, clear
- **map**: Ordered key-value pairs (sorted by key)
- **unordered_map**: Hash-based key-value pairs (faster lookup)

### 3. Templates
- **template_function**: Generic functions that work with any type
- **template_class**: Generic classes (commonly used in Eigen, OpenCV)

### 4. Memory Management
- **smart_pointers**: unique_ptr, shared_ptr, weak_ptr (essential for SLAM)

---

## Why These Matter for SLAM

- **Containers**: Store keypoints, descriptors, map points, keyframes
- **Templates**: Eigen matrices, OpenCV Mat operations
- **Smart Pointers**: Memory management for map points, keyframes in ORB-SLAM, etc.
- **Range-based loops**: Efficient iteration over point clouds, features
