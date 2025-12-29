#!/usr/bin/env python3
"""
Test nanobind SLAM bindings

This script demonstrates:
1. Python list/dict <-> C++ vector/map conversions
2. NumPy <-> Eigen zero-copy operations
3. Using exposed robotics classes (SE2, KalmanFilter)
"""

import numpy as np
import slam_bindings as slam

print("=" * 60)
print("nanobind SLAM Bindings - Test Examples")
print("=" * 60)

# =============================================================================
# Part 1: Container Conversions (Python <-> C++)
# =============================================================================
print("\n=== Part 1: Container Conversions ===\n")

# List to vector
data = [1.0, 2.0, 3.0, 4.0, 5.0]
result = slam.sum_vector(data)
print(f"sum_vector({data}) = {result}")

# Vector operations
scaled = slam.scale_vector(data, 2.0)
print(f"scale_vector({data}, 2.0) = {scaled}")

# Dictionary to map
config1 = {"speed": 2.0, "threshold": 0.5}
config2 = {"threshold": 0.8, "timeout": 10.0}
merged = slam.merge_configs(config1, config2)
print(f"\nmerge_configs:")
print(f"  config1 = {config1}")
print(f"  config2 = {config2}")
print(f"  merged  = {merged}")

# Complex nested structure
readings = [
    {"timestamp": 0.0, "distance": 1.5, "intensity": 0.8},
    {"timestamp": 0.1, "distance": 2.3, "intensity": 0.3},
    {"timestamp": 0.2, "distance": 0.8, "intensity": 0.9},
    {"timestamp": 0.3, "distance": 3.1, "intensity": 0.7},
]
filtered = slam.filter_readings(readings, "distance", 1.0)
print(f"\nfilter_readings (distance > 1.0):")
print(f"  input:  {len(readings)} readings")
print(f"  output: {len(filtered)} readings")
for r in filtered:
    print(f"    t={r['timestamp']:.1f}, d={r['distance']:.1f}")

# =============================================================================
# Part 2: NumPy <-> Eigen Zero-Copy Operations
# =============================================================================
print("\n=== Part 2: NumPy <-> Eigen Zero-Copy ===\n")

# Single point transformation
R = np.array([
    [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
    [np.sin(np.pi/4),  np.cos(np.pi/4), 0],
    [0,                0,               1]
], dtype=np.float64)

t = np.array([1.0, 2.0, 0.0], dtype=np.float64)
p = np.array([1.0, 0.0, 0.0], dtype=np.float64)

transformed = slam.transform_point(R, t, p)
print(f"transform_point:")
print(f"  R (45° rotation) = [[{R[0,0]:.3f}, {R[0,1]:.3f}], ...]")
print(f"  t = {t}")
print(f"  p = {p}")
print(f"  R @ p + t = {transformed}")

# Batch transformation (efficient with Eigen)
points = np.random.rand(3, 1000).astype(np.float64)
transformed_batch = slam.transform_points_batch(R, t, points)
print(f"\ntransform_points_batch:")
print(f"  Input:  {points.shape} points")
print(f"  Output: {transformed_batch.shape} points")
print(f"  First 3 input:  {points[:, :3].T}")
print(f"  First 3 output: {transformed_batch[:, :3].T}")

# Covariance computation
data_matrix = np.random.rand(3, 100).astype(np.float64)
cov = slam.compute_covariance(data_matrix)
print(f"\ncompute_covariance:")
print(f"  Input shape: {data_matrix.shape}")
print(f"  Covariance matrix:\n{cov}")

# SVD decomposition
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
U, S, V = slam.compute_svd(A)
print(f"\ncompute_svd:")
print(f"  Input A shape: {A.shape}")
print(f"  U shape: {U.shape}")
print(f"  S (singular values): {S}")
print(f"  V shape: {V.shape}")

# =============================================================================
# Part 3: SE2 Pose Class
# =============================================================================
print("\n=== Part 3: SE2 Pose Class ===\n")

# Create poses
pose1 = slam.SE2(1.0, 2.0, np.pi/4)
pose2 = slam.SE2(0.5, 0.0, np.pi/4)

print(f"pose1 = {pose1}")
print(f"pose2 = {pose2}")
print(f"pose1.x = {pose1.x}, pose1.y = {pose1.y}, pose1.theta = {pose1.theta:.4f}")

# Pose composition
composed = pose1.compose(pose2)
print(f"\npose1.compose(pose2) = {composed}")

# Using operator overload
composed_op = pose1 * pose2
print(f"pose1 * pose2 = {composed_op}")

# Inverse
inv_pose = pose1.inverse()
print(f"\npose1.inverse() = {inv_pose}")

# Verify: pose * inverse = identity (approximately)
identity = pose1 * pose1.inverse()
print(f"pose1 * pose1.inverse() = {identity}")

# Get as numpy array (zero-copy)
pose_vec = pose1.as_vector()
print(f"\npose1.as_vector() = {pose_vec}")

# Set from numpy array
new_pose = slam.SE2()
new_pose.from_vector(np.array([5.0, 6.0, 1.0]))
print(f"SE2 from vector [5, 6, 1] = {new_pose}")

# Transform a point
point_2d = np.array([1.0, 0.0], dtype=np.float64)
transformed_2d = pose1.transform_point(point_2d)
print(f"\npose1.transform_point({point_2d}) = {transformed_2d}")

# Get transformation matrix
T = pose1.as_matrix()
print(f"\npose1.as_matrix():\n{T}")

# =============================================================================
# Part 4: Kalman Filter
# =============================================================================
print("\n=== Part 4: Kalman Filter 1D ===\n")

# Initialize filter
# x0=0, P0=1, Q=0.1 (process noise), R=0.5 (measurement noise)
kf = slam.KalmanFilter1D(0.0, 1.0, 0.1, 0.5)
print(f"Initial state: {kf.state:.4f}, variance: {kf.variance:.4f}")

# Simulate measurements with noise
true_position = 5.0
np.random.seed(42)
measurements = true_position + np.random.normal(0, 0.5, 10)

print("\nTracking position with noisy measurements:")
print(f"True position: {true_position}")
print("-" * 50)
print(f"{'Step':>4} | {'Measurement':>11} | {'Estimate':>10} | {'Variance':>10}")
print("-" * 50)

for i, z in enumerate(measurements):
    kf.predict()
    kf.update(z)
    print(f"{i+1:4d} | {z:11.4f} | {kf.state:10.4f} | {kf.variance:10.4f}")

print("-" * 50)
print(f"Final estimate: {kf.state:.4f} (true: {true_position})")
print(f"Final variance: {kf.variance:.4f}")

# Get state as numpy vector
state_vec = kf.get_state_vector()
print(f"\nState vector [estimate, variance]: {state_vec}")

# =============================================================================
# Performance Comparison
# =============================================================================
print("\n=== Performance: NumPy vs C++ (via nanobind) ===\n")

import time

# Generate test data
N = 100000
data_list = np.random.rand(N).tolist()
data_array = np.array(data_list)

# Python sum
start = time.perf_counter()
py_sum = sum(data_list)
py_time = time.perf_counter() - start

# NumPy sum
start = time.perf_counter()
np_sum = np.sum(data_array)
np_time = time.perf_counter() - start

# C++ sum (via nanobind)
start = time.perf_counter()
cpp_sum = slam.sum_vector(data_list)
cpp_time = time.perf_counter() - start

print(f"Summing {N:,} elements:")
print(f"  Python sum():      {py_time*1000:.2f} ms (result: {py_sum:.6f})")
print(f"  NumPy sum():       {np_time*1000:.2f} ms (result: {np_sum:.6f})")
print(f"  C++ sum (nanobind):{cpp_time*1000:.2f} ms (result: {cpp_sum:.6f})")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
