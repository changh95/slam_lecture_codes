#!/usr/bin/env python3
"""
SymForce Tutorial: Symbolic Computing for Robotics

This example demonstrates:
1. Symbolic math with SymForce
2. Code generation for optimization
3. Factor graph optimization
"""

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import typing as T
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
import numpy as np

print("=== SymForce Tutorial ===\n")

# ============================================================================
# Example 1: Symbolic Math Basics
# ============================================================================
print("=== Example 1: Symbolic Math ===\n")

# Create symbolic variables
x = sf.Symbol("x")
y = sf.Symbol("y")

# Symbolic expression
expr = x**2 + 2*x*y + y**2
print(f"Expression: {expr}")
print(f"Simplified: {sf.simplify(expr)}")

# Differentiation
dx = expr.diff(x)
dy = expr.diff(y)
print(f"d/dx: {dx}")
print(f"d/dy: {dy}")

# Substitution
result = expr.subs(x, 2).subs(y, 3)
print(f"Evaluated at x=2, y=3: {result}")

# ============================================================================
# Example 2: Geometry Types
# ============================================================================
print("\n=== Example 2: Geometry Types ===\n")

# 2D Rotation
theta = sf.Symbol("theta")
rot2 = sf.Rot2.from_angle(theta)
print(f"Rot2 from angle {theta}: {rot2}")

# 2D Pose
pose2 = sf.Pose2(
    R=sf.Rot2.from_angle(sf.pi / 4),
    t=sf.V2([1.0, 2.0])
)
print(f"Pose2: {pose2}")

# 3D Rotation (quaternion)
rot3 = sf.Rot3.from_yaw_pitch_roll(
    yaw=sf.Symbol("yaw"),
    pitch=sf.Symbol("pitch"),
    roll=sf.Symbol("roll")
)
print(f"Rot3 from ypr: {rot3}")

# 3D Pose
pose3 = sf.Pose3(
    R=sf.Rot3.identity(),
    t=sf.V3([1.0, 2.0, 3.0])
)
print(f"Pose3: {pose3}")

# Composition
pose3_b = sf.Pose3(
    R=sf.Rot3.from_yaw_pitch_roll(0.1, 0.0, 0.0),
    t=sf.V3([0.5, 0.0, 0.0])
)
composed = pose3 * pose3_b
print(f"Pose composition: translation = {composed.t}")

# ============================================================================
# Example 3: Residual Functions
# ============================================================================
print("\n=== Example 3: Residual Functions ===\n")


def pose2_between_residual(
    pose_a: sf.Pose2,
    pose_b: sf.Pose2,
    measured: sf.Pose2,
    epsilon: sf.Scalar,
) -> sf.V3:
    """
    Residual for relative pose constraint.
    Error = measured^{-1} * (pose_a^{-1} * pose_b)
    """
    predicted = pose_a.inverse() * pose_b
    error = measured.inverse() * predicted
    return error.to_tangent(epsilon=epsilon)


# Test the residual
pose_a = sf.Pose2.identity()
pose_b = sf.Pose2(R=sf.Rot2.identity(), t=sf.V2([1.0, 0.0]))
measured = sf.Pose2(R=sf.Rot2.identity(), t=sf.V2([1.0, 0.0]))

residual = pose2_between_residual(pose_a, pose_b, measured, sf.epsilon())
print(f"Residual (perfect measurement): {residual}")

# With error
measured_noisy = sf.Pose2(R=sf.Rot2.identity(), t=sf.V2([1.1, 0.05]))
residual_noisy = pose2_between_residual(pose_a, pose_b, measured_noisy, sf.epsilon())
print(f"Residual (noisy measurement): {residual_noisy}")

# ============================================================================
# Example 4: Factor Graph Optimization
# ============================================================================
print("\n=== Example 4: Factor Graph Optimization ===\n")


def prior_residual(
    pose: sf.Pose2,
    prior: sf.Pose2,
    epsilon: sf.Scalar,
) -> sf.V3:
    """Prior factor residual."""
    return (prior.inverse() * pose).to_tangent(epsilon=epsilon)


def odometry_residual(
    pose_a: sf.Pose2,
    pose_b: sf.Pose2,
    odometry: sf.Pose2,
    epsilon: sf.Scalar,
) -> sf.V3:
    """Odometry factor residual."""
    predicted = pose_a.inverse() * pose_b
    return (odometry.inverse() * predicted).to_tangent(epsilon=epsilon)


# Ground truth: square trajectory
ground_truth = [
    sf.Pose2(R=sf.Rot2.from_angle(0.0), t=sf.V2([0.0, 0.0])),
    sf.Pose2(R=sf.Rot2.from_angle(0.0), t=sf.V2([1.0, 0.0])),
    sf.Pose2(R=sf.Rot2.from_angle(np.pi/2), t=sf.V2([1.0, 1.0])),
    sf.Pose2(R=sf.Rot2.from_angle(np.pi), t=sf.V2([0.0, 1.0])),
]

# Odometry measurements
odometry_measurements = [
    sf.Pose2(R=sf.Rot2.from_angle(0.0), t=sf.V2([1.0, 0.0])),
    sf.Pose2(R=sf.Rot2.from_angle(np.pi/2), t=sf.V2([1.0, 0.0])),
    sf.Pose2(R=sf.Rot2.from_angle(np.pi/2), t=sf.V2([1.0, 0.0])),
]

# Initial values (with noise)
np.random.seed(42)
initial_values = Values()
for i in range(4):
    noise_x = np.random.normal(0, 0.1)
    noise_y = np.random.normal(0, 0.1)
    noise_theta = np.random.normal(0, 0.05)

    noisy_pose = sf.Pose2(
        R=sf.Rot2.from_angle(float(ground_truth[i].rotation().to_tangent()[0]) + noise_theta),
        t=sf.V2([float(ground_truth[i].t[0]) + noise_x,
                 float(ground_truth[i].t[1]) + noise_y])
    )
    initial_values[f"pose_{i}"] = noisy_pose

initial_values["epsilon"] = sf.epsilon()

print("Initial poses (with noise):")
for i in range(4):
    pose = initial_values[f"pose_{i}"]
    print(f"  pose_{i}: t=({pose.t[0]:.3f}, {pose.t[1]:.3f})")

# Create factors
factors = []

# Prior on first pose
factors.append(Factor(
    residual=prior_residual,
    keys=["pose_0", "prior_pose_0", "epsilon"],
))
initial_values["prior_pose_0"] = ground_truth[0]

# Odometry factors
for i in range(3):
    factors.append(Factor(
        residual=odometry_residual,
        keys=[f"pose_{i}", f"pose_{i+1}", f"odom_{i}", "epsilon"],
    ))
    initial_values[f"odom_{i}"] = odometry_measurements[i]

# Create optimizer
optimizer = Optimizer(
    factors=factors,
    optimized_keys=[f"pose_{i}" for i in range(4)],
    debug_stats=True,
)

# Optimize
result = optimizer.optimize(initial_values)

print(f"\nOptimization completed in {result.iterations} iterations")
print(f"Initial error: {result.initial_error:.6f}")
print(f"Final error: {result.final_error:.6f}")

print("\nOptimized poses:")
for i in range(4):
    pose = result.optimized_values[f"pose_{i}"]
    gt = ground_truth[i]
    error = np.sqrt((float(pose.t[0]) - float(gt.t[0]))**2 +
                    (float(pose.t[1]) - float(gt.t[1]))**2)
    print(f"  pose_{i}: t=({float(pose.t[0]):.3f}, {float(pose.t[1]):.3f}), error={error:.4f}")

# ============================================================================
# Example 5: Code Generation (Preview)
# ============================================================================
print("\n=== Example 5: Code Generation ===\n")

from symforce.codegen import Codegen, CppConfig

# Generate C++ code for the residual function
codegen = Codegen.function(
    func=pose2_between_residual,
    config=CppConfig(),
)

print("Generated C++ function signature:")
print(f"  {codegen.function.name}(")
for arg in codegen.inputs.keys():
    print(f"    {arg},")
print("  )")

print("\nTo generate files, use:")
print("  codegen.generate_function(output_dir='gen')")

print("\n=== Complete ===")
