#!/usr/bin/env python3
"""
02_forward_kinematics.py - Forward Kinematics

Forward Kinematics (FK) calculates the end-effector pose given joint angles.

Given: q = [q1, q2, ..., qn] (joint angles)
Find: T = SE(3) (end-effector pose)

FK always has a unique solution (unlike inverse kinematics).
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

def main():
    print("=" * 60)
    print("Forward Kinematics Demo")
    print("=" * 60)

    # =========================================================================
    # 1. Simple 2-link planar robot
    # =========================================================================
    print("\n--- 1. Two-Link Planar Robot FK ---")

    # Create 2-link planar robot (each link 1m long)
    robot = rtb.DHRobot([
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
    ], name="2-Link Planar")

    # Calculate FK for different configurations
    configurations = [
        [0, 0],                           # Fully extended
        [np.pi/4, 0],                     # First joint at 45 deg
        [np.pi/4, -np.pi/2],              # L-shape
        [np.pi/2, np.pi/2],               # Folded back
    ]

    print(f"\nRobot: {robot.name}, Link lengths: 1m, 1m")
    print("-" * 50)

    for i, q in enumerate(configurations):
        T = robot.fkine(q)  # Forward kinematics

        # Extract position (translation)
        x, y, z = T.t

        # Extract rotation (as angle around z for planar robot)
        theta = np.arctan2(T.R[1, 0], T.R[0, 0])

        print(f"\nConfiguration {i+1}: q = [{np.rad2deg(q[0]):.1f}, {np.rad2deg(q[1]):.1f}] deg")
        print(f"  End-effector position: x = {x:.3f}, y = {y:.3f}")
        print(f"  End-effector angle: theta = {np.rad2deg(theta):.1f} deg")

    # =========================================================================
    # 2. Puma 560 FK
    # =========================================================================
    print("\n--- 2. Puma 560 Forward Kinematics ---")

    puma = rtb.models.DH.Puma560()

    # Home configuration (all zeros)
    q_home = np.zeros(6)
    T_home = puma.fkine(q_home)

    print(f"\nHome configuration q = {q_home}")
    print(f"End-effector pose:\n{T_home}")

    # Named configuration
    q_ready = puma.qr  # Ready position
    T_ready = puma.fkine(q_ready)

    print(f"\nReady configuration q = {np.rad2deg(q_ready)} deg")
    print(f"End-effector position: {T_ready.t}")

    # =========================================================================
    # 3. Step-by-step FK calculation
    # =========================================================================
    print("\n--- 3. Step-by-Step FK Calculation ---")

    # For the 2-link planar robot, show transformation matrices
    q = [np.pi/4, -np.pi/4]  # 45 deg, -45 deg

    print(f"\nConfiguration: q = {np.rad2deg(q)} deg")

    # Individual link transformations
    T01 = robot.links[0].A(q[0])  # Transform from frame 0 to frame 1
    T12 = robot.links[1].A(q[1])  # Transform from frame 1 to frame 2

    print(f"\nT01 (Base to Joint 1):\n{T01}")
    print(f"\nT12 (Joint 1 to Joint 2):\n{T12}")

    # Total transformation
    T02 = T01 @ T12
    print(f"\nT02 = T01 * T12 (Total transformation):\n{T02}")

    # Verify with fkine
    T_fkine = robot.fkine(q)
    print(f"\nVerification with fkine():\n{T_fkine}")

    # =========================================================================
    # 4. FK for a trajectory
    # =========================================================================
    print("\n--- 4. FK Along a Trajectory ---")

    # Joint trajectory from q1 to q2
    q1 = [0, 0]
    q2 = [np.pi/2, -np.pi/2]

    print(f"Trajectory from q1={np.rad2deg(q1)} to q2={np.rad2deg(q2)} deg")
    print("-" * 40)

    n_steps = 5
    for i in range(n_steps + 1):
        # Linear interpolation
        alpha = i / n_steps
        q = [q1[0] + alpha * (q2[0] - q1[0]),
             q1[1] + alpha * (q2[1] - q1[1])]

        T = robot.fkine(q)

        print(f"Step {i}: q=[{np.rad2deg(q[0]):6.1f}, {np.rad2deg(q[1]):6.1f}] deg "
              f"-> pos=({T.t[0]:6.3f}, {T.t[1]:6.3f})")

    # =========================================================================
    # 5. Workspace visualization (conceptual)
    # =========================================================================
    print("\n--- 5. Workspace Analysis ---")

    # Sample the workspace
    print("\nSampling workspace of 2-link planar robot...")

    q1_range = np.linspace(-np.pi, np.pi, 20)
    q2_range = np.linspace(-np.pi, np.pi, 20)

    x_positions = []
    y_positions = []

    for q1 in q1_range:
        for q2 in q2_range:
            T = robot.fkine([q1, q2])
            x_positions.append(T.t[0])
            y_positions.append(T.t[1])

    x_min, x_max = min(x_positions), max(x_positions)
    y_min, y_max = min(y_positions), max(y_positions)

    print(f"Workspace bounds:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Max reach: {max(np.sqrt(np.array(x_positions)**2 + np.array(y_positions)**2)):.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - FK: joint angles -> end-effector pose")
    print("  - Use robot.fkine(q) for direct calculation")
    print("  - FK is the product of individual link transforms")
    print("  - T = T01 * T12 * ... * T(n-1)n")
    print("  - FK always has a unique solution")
    print("=" * 60)


if __name__ == "__main__":
    main()
