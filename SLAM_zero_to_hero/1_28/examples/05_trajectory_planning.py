#!/usr/bin/env python3
"""
05_trajectory_planning.py - Trajectory Planning

This example demonstrates trajectory planning in:
1. Joint space - interpolate between joint configurations
2. Cartesian space - interpolate between end-effector poses

Trajectory planning is important for smooth, collision-free robot motion.
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

def main():
    print("=" * 60)
    print("Trajectory Planning Demo")
    print("=" * 60)

    # =========================================================================
    # 1. Joint space trajectory (jtraj)
    # =========================================================================
    print("\n--- 1. Joint Space Trajectory ---")

    # Create a simple 2-link robot
    robot = rtb.DHRobot([
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
        rtb.RevoluteDH(d=0, a=1.0, alpha=0),
    ], name="2-Link Planar")

    # Start and end configurations
    q_start = np.array([0, 0])
    q_end = np.array([np.pi/2, -np.pi/2])

    # Generate trajectory
    n_steps = 10
    traj = rtb.jtraj(q_start, q_end, n_steps)

    print(f"\nFrom q_start = {np.rad2deg(q_start)} deg")
    print(f"To   q_end   = {np.rad2deg(q_end)} deg")
    print(f"Steps: {n_steps}")

    print("\nJoint trajectory:")
    print("-" * 50)
    print(f"{'Step':>4} {'q1 (deg)':>10} {'q2 (deg)':>10} {'End-effector (x, y)':>20}")
    print("-" * 50)

    for i, q in enumerate(traj.q):
        T = robot.fkine(q)
        print(f"{i:4d} {np.rad2deg(q[0]):10.2f} {np.rad2deg(q[1]):10.2f} "
              f"({T.t[0]:8.3f}, {T.t[1]:8.3f})")

    # =========================================================================
    # 2. Quintic polynomial properties
    # =========================================================================
    print("\n--- 2. Quintic Polynomial Properties ---")

    print("\njtraj uses quintic (5th order) polynomial:")
    print("  - Position, velocity, acceleration are smooth")
    print("  - Starts and ends with zero velocity")
    print("  - Starts and ends with zero acceleration")

    print("\nVelocity profile:")
    print("-" * 40)

    for i, (q, qd, qdd) in enumerate(zip(traj.q, traj.qd, traj.qdd)):
        if i % 2 == 0:  # Print every other step
            print(f"Step {i}: vel = [{np.rad2deg(qd[0]):7.2f}, {np.rad2deg(qd[1]):7.2f}] deg/s")

    # =========================================================================
    # 3. Cartesian space trajectory
    # =========================================================================
    print("\n--- 3. Cartesian Space Trajectory ---")

    puma = rtb.models.DH.Puma560()

    # Start and end poses
    T_start = SE3.Tx(0.5) * SE3.Ty(0.2) * SE3.Tz(0.3)
    T_end = SE3.Tx(0.4) * SE3.Ty(-0.3) * SE3.Tz(0.5)

    print(f"\nStart pose: {T_start.t}")
    print(f"End pose:   {T_end.t}")

    # Cartesian trajectory
    n_steps = 5
    Ts = rtb.ctraj(T_start, T_end, n_steps)

    print("\nCartesian trajectory (positions):")
    for i, T in enumerate(Ts):
        print(f"  Step {i}: ({T.t[0]:.3f}, {T.t[1]:.3f}, {T.t[2]:.3f})")

    # =========================================================================
    # 4. Straight line vs joint space
    # =========================================================================
    print("\n--- 4. Straight Line vs Joint Space ---")

    print("\nJoint space trajectory:")
    print("  + Smooth joint motion")
    print("  + Guaranteed no joint limit violations (if endpoints are OK)")
    print("  - End-effector path is NOT a straight line")

    print("\nCartesian space trajectory:")
    print("  + End-effector follows straight line")
    print("  + Predictable path")
    print("  - Requires IK at each step")
    print("  - May have singularity issues")

    # Demonstrate the difference
    q_start = np.array([0, np.pi/4])
    q_end = np.array([np.pi/2, -np.pi/4])

    n_steps = 6
    jtraj_result = rtb.jtraj(q_start, q_end, n_steps)

    print("\n2-Link Robot Path Comparison:")
    print("-" * 50)

    # Joint space path
    joint_path_x = []
    joint_path_y = []

    print("\nJoint space path:")
    for i, q in enumerate(jtraj_result.q):
        T = robot.fkine(q)
        joint_path_x.append(T.t[0])
        joint_path_y.append(T.t[1])
        print(f"  Step {i}: ({T.t[0]:.3f}, {T.t[1]:.3f})")

    # Cartesian space path (straight line)
    T_start = robot.fkine(q_start)
    T_end = robot.fkine(q_end)

    print("\nCartesian (straight line) path:")
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        x = T_start.t[0] + alpha * (T_end.t[0] - T_start.t[0])
        y = T_start.t[1] + alpha * (T_end.t[1] - T_start.t[1])
        print(f"  Step {i}: ({x:.3f}, {y:.3f})")

    # =========================================================================
    # 5. Trajectory with via points
    # =========================================================================
    print("\n--- 5. Trajectory Through Via Points ---")

    # Define via points
    via_points = np.array([
        [0, 0],                    # Start
        [np.pi/4, np.pi/4],        # Via point 1
        [np.pi/2, 0],              # Via point 2
        [np.pi/2, -np.pi/2],       # End
    ])

    print("Via points (joint space):")
    for i, q in enumerate(via_points):
        T = robot.fkine(q)
        print(f"  Point {i}: q = {np.rad2deg(q)} deg -> ({T.t[0]:.3f}, {T.t[1]:.3f})")

    # Generate trajectory through via points
    print("\nTrajectory through via points:")
    all_q = []

    for i in range(len(via_points) - 1):
        traj_segment = rtb.jtraj(via_points[i], via_points[i+1], 5)
        all_q.extend(traj_segment.q[:-1] if i < len(via_points) - 2 else traj_segment.q)

    print(f"Total trajectory points: {len(all_q)}")

    # =========================================================================
    # 6. Time-parameterized trajectory
    # =========================================================================
    print("\n--- 6. Time-Parameterized Trajectory ---")

    # With specific time duration
    t_duration = 2.0  # seconds
    n_steps = 11
    t = np.linspace(0, t_duration, n_steps)

    traj = rtb.jtraj(q_start, q_end, t)

    print(f"\nTrajectory duration: {t_duration} seconds")
    print(f"Time steps: {n_steps}")

    print("\nTime-parameterized trajectory:")
    for i in range(0, n_steps, 2):
        print(f"  t = {t[i]:.2f}s: q = {np.rad2deg(traj.q[i])} deg")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - jtraj: Joint space trajectory (quintic polynomial)")
    print("  - ctraj: Cartesian space trajectory (straight line)")
    print("  - Joint space: smooth joints, curved path")
    print("  - Cartesian space: straight path, needs IK")
    print("  - Can specify time duration for real-time control")
    print("=" * 60)


if __name__ == "__main__":
    main()
