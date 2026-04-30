#!/usr/bin/env python3
"""
01_robot_creation.py - Creating Robot Models

This example demonstrates how to create robot models using:
1. Denavit-Hartenberg (DH) parameters
2. Predefined robot models (Puma 560, UR5, Panda, etc.)

Robotics Toolbox for Python: https://github.com/petercorke/robotics-toolbox-python
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

def main():
    print("=" * 60)
    print("Robot Creation with Robotics Toolbox")
    print("=" * 60)

    # =========================================================================
    # 1. Create a simple 2-link planar robot using DH parameters
    # =========================================================================
    print("\n--- 1. Two-Link Planar Robot (DH Parameters) ---")

    # DH parameters: [theta, d, a, alpha]
    # theta: joint angle (for revolute joints, this is variable)
    # d: link offset along z-axis
    # a: link length along x-axis
    # alpha: twist angle

    # Define links using RevoluteDH (revolute joints)
    link1 = rtb.RevoluteDH(d=0, a=1.0, alpha=0)  # Link 1: length 1m
    link2 = rtb.RevoluteDH(d=0, a=1.0, alpha=0)  # Link 2: length 1m

    # Create the robot
    planar_robot = rtb.DHRobot([link1, link2], name="2-Link Planar")

    print(f"Robot name: {planar_robot.name}")
    print(f"Number of joints: {planar_robot.n}")
    print(f"DH table:\n{planar_robot}")

    # =========================================================================
    # 2. Using predefined robot models
    # =========================================================================
    print("\n--- 2. Predefined Robot Models ---")

    # Puma 560 - Classic industrial robot
    puma = rtb.models.DH.Puma560()
    print(f"\nPuma 560:")
    print(f"  Joints: {puma.n}")
    print(f"  Joint types: {[link.isrevolute for link in puma.links]}")

    # UR5 - Universal Robots collaborative arm
    # Note: UR5 uses modified DH, but we can create it
    print("\nUR5 (Universal Robots):")
    print("  6-DOF collaborative robot")
    print("  Commonly used in research and industry")

    # Panda - Franka Emika robot arm
    panda = rtb.models.DH.Panda()
    print(f"\nFranka Panda:")
    print(f"  Joints: {panda.n}")
    print(f"  Popular 7-DOF collaborative robot")

    # =========================================================================
    # 3. Create a custom 3-DOF robot
    # =========================================================================
    print("\n--- 3. Custom 3-DOF Robot ---")

    # 3-DOF articulated robot (like a simplified industrial arm)
    # Joint 1: Rotation around vertical axis (base)
    # Joint 2: Shoulder
    # Joint 3: Elbow

    robot_3dof = rtb.DHRobot([
        rtb.RevoluteDH(d=0.5, a=0, alpha=np.pi/2),    # Base rotation
        rtb.RevoluteDH(d=0, a=0.5, alpha=0),           # Shoulder
        rtb.RevoluteDH(d=0, a=0.5, alpha=0),           # Elbow
    ], name="Custom 3-DOF")

    print(f"Robot: {robot_3dof.name}")
    print(f"DH table:\n{robot_3dof}")

    # =========================================================================
    # 4. Robot properties and limits
    # =========================================================================
    print("\n--- 4. Robot Properties ---")

    # Using Puma as example
    print(f"\nPuma 560 Properties:")
    print(f"  Joint limits (lower): {np.rad2deg(puma.qlim[0])} deg")
    print(f"  Joint limits (upper): {np.rad2deg(puma.qlim[1])} deg")

    # Set a configuration
    q = [0, np.pi/4, -np.pi/4, 0, np.pi/4, 0]
    print(f"\n  Configuration q = {np.rad2deg(q)} deg")

    # =========================================================================
    # 5. Visualize robot (if display available)
    # =========================================================================
    print("\n--- 5. Robot Visualization ---")

    try:
        # Try to show the robot
        print("Attempting to display robot...")

        # Simple text-based representation
        print(f"\nPlanar robot at q=[0, 0]:")
        T = planar_robot.fkine([0, 0])
        print(f"  End-effector position: x={T.t[0]:.2f}, y={T.t[1]:.2f}")

        print(f"\nPlanar robot at q=[45, -45] degrees:")
        T = planar_robot.fkine([np.deg2rad(45), np.deg2rad(-45)])
        print(f"  End-effector position: x={T.t[0]:.2f}, y={T.t[1]:.2f}")

    except Exception as e:
        print(f"Visualization not available: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Robots can be created using DH parameters")
    print("  - Predefined models: Puma560, Panda, UR5, etc.")
    print("  - Each link defined by: d, a, alpha, theta")
    print("  - RevoluteDH for revolute joints")
    print("  - PrismaticDH for prismatic joints")
    print("=" * 60)


if __name__ == "__main__":
    main()
