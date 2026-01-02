#!/usr/bin/env python3
"""
03_inverse_kinematics.py - Inverse Kinematics

Inverse Kinematics (IK) calculates joint angles given a target end-effector pose.

Given: T = SE(3) (desired end-effector pose)
Find: q = [q1, q2, ..., qn] (joint angles)

IK can have:
- No solution (target outside workspace)
- One solution
- Multiple solutions (most common)
- Infinite solutions (redundant robots)
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

def main():
    print("=" * 60)
    print("Inverse Kinematics Demo")
    print("=" * 60)

    # =========================================================================
    # 1. Simple 2-link planar robot IK
    # =========================================================================
    print("\n--- 1. Two-Link Planar Robot IK ---")

    # Create 2-link planar robot
    robot = rtb.DHRobot([
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
    ], name="2-Link Planar")

    # Target position
    target_x, target_y = 1.5, 0.5

    # Create target pose (for 2D, rotation is around Z)
    T_target = SE3.Tx(target_x) * SE3.Ty(target_y)

    print(f"\nTarget position: ({target_x}, {target_y})")

    # Solve IK using Levenberg-Marquardt
    solution = robot.ikine_LM(T_target)

    if solution.success:
        q_solution = solution.q
        print(f"IK Solution: q = {np.rad2deg(q_solution)} deg")

        # Verify with FK
        T_result = robot.fkine(q_solution)
        print(f"Verification (FK): pos = ({T_result.t[0]:.4f}, {T_result.t[1]:.4f})")

        # Calculate error
        error = np.linalg.norm(T_target.t[:2] - T_result.t[:2])
        print(f"Position error: {error:.6f}")
    else:
        print("IK failed to find a solution!")

    # =========================================================================
    # 2. Multiple solutions
    # =========================================================================
    print("\n--- 2. Multiple IK Solutions ---")

    # For many robots, there are multiple ways to reach the same pose
    print("\nTwo-link robot can reach the same point with 'elbow up' or 'elbow down':")

    # Analytical solution for 2-link planar robot
    L1, L2 = 1.0, 1.0  # Link lengths

    # For target (x, y), the two solutions are:
    x, y = 1.2, 0.8
    d = np.sqrt(x**2 + y**2)

    if d <= L1 + L2:  # Check if reachable
        # Using law of cosines
        cos_q2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_q2 = np.clip(cos_q2, -1, 1)  # Numerical safety

        # Elbow down (positive q2)
        q2_down = np.arccos(cos_q2)
        q1_down = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2_down), L1 + L2 * np.cos(q2_down))

        # Elbow up (negative q2)
        q2_up = -np.arccos(cos_q2)
        q1_up = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2_up), L1 + L2 * np.cos(q2_up))

        print(f"\nTarget: ({x}, {y})")
        print(f"Solution 1 (elbow down): q = [{np.rad2deg(q1_down):.1f}, {np.rad2deg(q2_down):.1f}] deg")
        print(f"Solution 2 (elbow up):   q = [{np.rad2deg(q1_up):.1f}, {np.rad2deg(q2_up):.1f}] deg")

        # Verify both
        T1 = robot.fkine([q1_down, q2_down])
        T2 = robot.fkine([q1_up, q2_up])
        print(f"\nVerification:")
        print(f"  Solution 1 -> ({T1.t[0]:.4f}, {T1.t[1]:.4f})")
        print(f"  Solution 2 -> ({T2.t[0]:.4f}, {T2.t[1]:.4f})")

    # =========================================================================
    # 3. IK with initial guess
    # =========================================================================
    print("\n--- 3. IK with Initial Guess ---")

    puma = rtb.models.DH.Puma560()

    # Target pose
    T_target = SE3.Tx(0.5) * SE3.Ty(0.3) * SE3.Tz(0.4)

    # Different initial guesses can lead to different solutions
    initial_guesses = [
        np.zeros(6),
        np.array([0.5, 0.5, 0.5, 0, 0, 0]),
        np.array([-0.5, 0.5, -0.5, 0, 0, 0]),
    ]

    print(f"\nTarget pose: t = {T_target.t}")

    for i, q0 in enumerate(initial_guesses):
        solution = puma.ikine_LM(T_target, q0=q0)

        if solution.success:
            # Verify
            T_result = puma.fkine(solution.q)
            error = np.linalg.norm(T_target.t - T_result.t)

            print(f"\nInitial guess {i+1}: q0 = {np.rad2deg(q0)[:3]}... deg")
            print(f"  Solution: q = {np.rad2deg(solution.q)} deg")
            print(f"  Position error: {error:.6f}")

    # =========================================================================
    # 4. Unreachable targets
    # =========================================================================
    print("\n--- 4. Unreachable Targets ---")

    # Target outside workspace
    T_unreachable = SE3.Tx(10)  # Way too far

    print(f"\nTarget: ({T_unreachable.t[0]}, {T_unreachable.t[1]}, {T_unreachable.t[2]})")
    print("This is outside the robot's workspace!")

    solution = robot.ikine_LM(T_unreachable, ilimit=100)

    if solution.success:
        print(f"Solution found (might be approximate): {solution.q}")
    else:
        print(f"IK failed: Target is unreachable")
        print(f"  Reason: {solution.reason}")

    # =========================================================================
    # 5. Different IK methods
    # =========================================================================
    print("\n--- 5. Different IK Methods ---")

    T_target = SE3.Tx(0.4) * SE3.Ty(0.2) * SE3.Tz(0.3)

    print(f"\nTarget: {T_target.t}")

    # Levenberg-Marquardt
    sol_lm = puma.ikine_LM(T_target)
    print(f"\nikine_LM (Levenberg-Marquardt):")
    print(f"  Success: {sol_lm.success}")
    if sol_lm.success:
        print(f"  Solution: {np.rad2deg(sol_lm.q)[:3]}... deg")

    # Newton-Raphson
    sol_nr = puma.ikine_NR(T_target)
    print(f"\nikine_NR (Newton-Raphson):")
    print(f"  Success: {sol_nr.success}")
    if sol_nr.success:
        print(f"  Solution: {np.rad2deg(sol_nr.q)[:3]}... deg")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - IK: end-effector pose -> joint angles")
    print("  - Can have 0, 1, or multiple solutions")
    print("  - Initial guess affects which solution is found")
    print("  - Methods: ikine_LM (robust), ikine_NR (fast)")
    print("  - Always verify IK solution with FK")
    print("=" * 60)


if __name__ == "__main__":
    main()
