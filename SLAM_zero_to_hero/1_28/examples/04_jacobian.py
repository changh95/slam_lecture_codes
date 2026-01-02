#!/usr/bin/env python3
"""
04_jacobian.py - Jacobian Matrix and Velocity Kinematics

The Jacobian matrix relates joint velocities to end-effector velocities:

    v = J(q) * q_dot

where:
    v = [linear_velocity; angular_velocity] (6x1)
    J = Jacobian matrix (6xn)
    q_dot = joint velocities (nx1)

The Jacobian is also crucial for:
- Singularity analysis
- Force/torque relationships
- Inverse kinematics algorithms
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

def main():
    print("=" * 60)
    print("Jacobian Matrix and Velocity Kinematics")
    print("=" * 60)

    # =========================================================================
    # 1. Jacobian of a 2-link planar robot
    # =========================================================================
    print("\n--- 1. Two-Link Planar Robot Jacobian ---")

    robot = rtb.DHRobot([
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
    ], name="2-Link Planar")

    # Configuration
    q = [np.pi/4, np.pi/4]  # 45 deg each

    # Compute Jacobian in base frame
    J = robot.jacob0(q)

    print(f"\nConfiguration: q = {np.rad2deg(q)} deg")
    print(f"\nJacobian (base frame) - 6x2 matrix:")
    print(f"{J}")

    # For planar robot, only rows 0, 1 (linear x, y) and 5 (angular z) are relevant
    print(f"\nRelevant components (x, y, theta_z):")
    print(f"  J[0,:] (dx/dq): {J[0,:]}")
    print(f"  J[1,:] (dy/dq): {J[1,:]}")
    print(f"  J[5,:] (dtheta_z/dq): {J[5,:]}")

    # =========================================================================
    # 2. Velocity kinematics
    # =========================================================================
    print("\n--- 2. Velocity Kinematics ---")

    # Joint velocities
    q_dot = np.array([0.5, 0.3])  # rad/s

    # End-effector velocity
    v = J @ q_dot

    print(f"\nJoint velocities: {q_dot} rad/s")
    print(f"\nEnd-effector velocity:")
    print(f"  Linear (x, y, z): {v[:3]} m/s")
    print(f"  Angular (wx, wy, wz): {v[3:]} rad/s")

    # For planar robot
    print(f"\nPlanar end-effector velocity:")
    print(f"  vx = {v[0]:.4f} m/s")
    print(f"  vy = {v[1]:.4f} m/s")
    print(f"  omega = {v[5]:.4f} rad/s")

    # =========================================================================
    # 3. Jacobian in different frames
    # =========================================================================
    print("\n--- 3. Jacobian in Different Frames ---")

    puma = rtb.models.DH.Puma560()
    q = np.zeros(6)

    # Jacobian in base frame (world frame)
    J0 = puma.jacob0(q)

    # Jacobian in end-effector frame
    Je = puma.jacobe(q)

    print(f"\nConfiguration: q = {q}")

    print(f"\nJacobian in base frame J0 (shape {J0.shape}):")
    print(f"  First row: {J0[0,:]}")

    print(f"\nJacobian in end-effector frame Je (shape {Je.shape}):")
    print(f"  First row: {Je[0,:]}")

    # =========================================================================
    # 4. Singularity analysis
    # =========================================================================
    print("\n--- 4. Singularity Analysis ---")

    print("\nSingularities occur when the Jacobian loses rank.")
    print("At singularity, the robot loses ability to move in some directions.")

    # 2-link robot singularities
    print("\n2-Link Planar Robot:")

    configurations = [
        [0, 0],           # Fully extended (singular!)
        [0, np.pi],       # Folded back (singular!)
        [np.pi/4, np.pi/4],  # Non-singular
        [np.pi/2, np.pi/2],  # Non-singular
    ]

    for q in configurations:
        J = robot.jacob0(q)
        J_reduced = J[[0, 1, 5], :]  # Only x, y, theta_z

        # Manipulability measure (sqrt of det(J*J^T))
        manipulability = np.sqrt(np.linalg.det(J_reduced @ J_reduced.T))

        # Condition number (higher = closer to singular)
        cond = np.linalg.cond(J_reduced)

        print(f"\nq = {np.rad2deg(q)} deg:")
        print(f"  Manipulability: {manipulability:.4f}")
        print(f"  Condition number: {cond:.2f}")

        if manipulability < 0.01:
            print(f"  WARNING: Near singularity!")

    # =========================================================================
    # 5. Inverse velocity kinematics
    # =========================================================================
    print("\n--- 5. Inverse Velocity Kinematics ---")

    q = [np.pi/4, np.pi/4]
    J = robot.jacob0(q)
    J_reduced = J[[0, 1, 5], :]

    # Desired end-effector velocity
    v_desired = np.array([0.1, 0.2, 0.0])  # vx, vy, omega

    # Solve for joint velocities: q_dot = J^(-1) * v
    # Use pseudoinverse for numerical stability
    J_pinv = np.linalg.pinv(J_reduced)
    q_dot = J_pinv @ v_desired

    print(f"\nConfiguration: q = {np.rad2deg(q)} deg")
    print(f"Desired velocity: vx={v_desired[0]}, vy={v_desired[1]}, omega={v_desired[2]}")
    print(f"\nRequired joint velocities: {q_dot} rad/s")

    # Verify
    v_result = J_reduced @ q_dot
    print(f"Verification: v = {v_result}")

    # =========================================================================
    # 6. Manipulability ellipsoid (conceptual)
    # =========================================================================
    print("\n--- 6. Manipulability Analysis ---")

    print("\nManipulability measures how easily the robot can move in different directions.")
    print("Higher manipulability = more dexterous configuration")

    q_test = np.linspace(-np.pi, np.pi, 12)

    print("\nManipulability at different configurations:")
    print("-" * 40)

    for q1 in [0, np.pi/4, np.pi/2]:
        for q2 in [0, np.pi/4, np.pi/2]:
            q = [q1, q2]
            J = robot.jacob0(q)
            J_reduced = J[[0, 1], :]  # Just linear velocity

            # Yoshikawa's manipulability
            w = np.sqrt(np.linalg.det(J_reduced @ J_reduced.T))

            print(f"q = [{np.rad2deg(q1):6.1f}, {np.rad2deg(q2):6.1f}] deg: w = {w:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Jacobian relates joint velocities to end-effector velocities")
    print("  - v = J(q) * q_dot")
    print("  - jacob0: Jacobian in base frame")
    print("  - jacobe: Jacobian in end-effector frame")
    print("  - Singularity: Jacobian loses rank (det = 0)")
    print("  - Manipulability: measure of dexterity at configuration")
    print("=" * 60)


if __name__ == "__main__":
    main()
