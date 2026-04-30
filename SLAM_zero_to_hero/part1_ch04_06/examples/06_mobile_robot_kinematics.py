#!/usr/bin/env python3
"""
06_mobile_robot_kinematics.py - Mobile Robot Kinematics

This example covers kinematics models for mobile robots:
1. Differential drive (two-wheel robot)
2. Ackermann steering (car-like robot)
3. Unicycle model (simplified differential drive)

These models are fundamental for wheel odometry in SLAM.
"""

import numpy as np
import matplotlib.pyplot as plt

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def main():
    print("=" * 60)
    print("Mobile Robot Kinematics")
    print("=" * 60)

    # =========================================================================
    # 1. Unicycle Model
    # =========================================================================
    print("\n--- 1. Unicycle Model ---")

    print("""
    The unicycle model is a simplified representation:

    State: [x, y, theta]
    Input: [v, omega] (linear velocity, angular velocity)

    Kinematics:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = omega
    """)

    # Simulate unicycle motion
    def unicycle_step(state, v, omega, dt):
        x, y, theta = state
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = normalize_angle(theta + omega * dt)
        return np.array([x_new, y_new, theta_new])

    # Initial state
    state = np.array([0.0, 0.0, 0.0])  # x, y, theta

    # Simulate circular motion
    v = 1.0        # 1 m/s forward
    omega = 0.5    # 0.5 rad/s turning
    dt = 0.1       # 100ms timestep
    n_steps = 50

    print(f"\nSimulation: v = {v} m/s, omega = {omega} rad/s")
    print(f"Expected radius of curvature: r = v/omega = {v/omega:.2f} m")

    trajectory = [state.copy()]
    for _ in range(n_steps):
        state = unicycle_step(state, v, omega, dt)
        trajectory.append(state.copy())

    trajectory = np.array(trajectory)

    print(f"\nInitial state: {trajectory[0]}")
    print(f"Final state: {trajectory[-1]}")
    print(f"Distance traveled: {np.sqrt(trajectory[-1,0]**2 + trajectory[-1,1]**2):.2f} m")

    # =========================================================================
    # 2. Differential Drive Model
    # =========================================================================
    print("\n--- 2. Differential Drive Model ---")

    print("""
    Differential drive has two wheels:

    Parameters:
        r: wheel radius
        L: wheel separation (track width)

    Input: [omega_l, omega_r] (left/right wheel angular velocities)

    Conversion to unicycle:
        v = r * (omega_r + omega_l) / 2
        omega = r * (omega_r - omega_l) / L
    """)

    class DifferentialDrive:
        def __init__(self, wheel_radius, wheel_separation):
            self.r = wheel_radius
            self.L = wheel_separation

        def forward_kinematics(self, omega_l, omega_r):
            """Convert wheel velocities to robot velocities"""
            v = self.r * (omega_r + omega_l) / 2
            omega = self.r * (omega_r - omega_l) / self.L
            return v, omega

        def inverse_kinematics(self, v, omega):
            """Convert robot velocities to wheel velocities"""
            omega_r = (2 * v + omega * self.L) / (2 * self.r)
            omega_l = (2 * v - omega * self.L) / (2 * self.r)
            return omega_l, omega_r

        def step(self, state, omega_l, omega_r, dt):
            """Simulate one timestep"""
            v, omega = self.forward_kinematics(omega_l, omega_r)
            return unicycle_step(state, v, omega, dt)

    # Create robot
    robot = DifferentialDrive(wheel_radius=0.1, wheel_separation=0.5)

    print(f"\nRobot parameters:")
    print(f"  Wheel radius: {robot.r} m")
    print(f"  Wheel separation: {robot.L} m")

    # Test forward kinematics
    test_cases = [
        (10.0, 10.0, "Straight forward"),
        (-10.0, -10.0, "Straight backward"),
        (5.0, 10.0, "Turn left"),
        (10.0, 5.0, "Turn right"),
        (-5.0, 5.0, "Rotate in place (CCW)"),
        (5.0, -5.0, "Rotate in place (CW)"),
    ]

    print("\nForward kinematics examples:")
    print("-" * 60)
    for omega_l, omega_r, description in test_cases:
        v, omega = robot.forward_kinematics(omega_l, omega_r)
        print(f"{description:25s}: omega_l={omega_l:5.1f}, omega_r={omega_r:5.1f} rad/s "
              f"-> v={v:5.2f} m/s, omega={omega:5.2f} rad/s")

    # Test inverse kinematics
    print("\nInverse kinematics examples:")
    print("-" * 60)
    for v, omega in [(1.0, 0.0), (0.0, 1.0), (1.0, 0.5)]:
        omega_l, omega_r = robot.inverse_kinematics(v, omega)
        print(f"v={v:.1f} m/s, omega={omega:.1f} rad/s -> "
              f"omega_l={omega_l:.2f}, omega_r={omega_r:.2f} rad/s")

    # =========================================================================
    # 3. Ackermann Steering Model
    # =========================================================================
    print("\n--- 3. Ackermann Steering (Car-like) Model ---")

    print("""
    Car-like robot with front wheel steering:

    Parameters:
        L: wheelbase (front-to-rear axle distance)

    State: [x, y, theta]
    Input: [v, delta] (velocity, steering angle)

    Kinematics (rear-wheel reference):
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = v * tan(delta) / L
    """)

    class AckermannRobot:
        def __init__(self, wheelbase):
            self.L = wheelbase

        def step(self, state, v, delta, dt):
            """Simulate one timestep with Ackermann steering"""
            x, y, theta = state

            # Ackermann kinematics
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
            theta_new = normalize_angle(theta + v * np.tan(delta) / self.L * dt)

            return np.array([x_new, y_new, theta_new])

        def turning_radius(self, delta):
            """Calculate turning radius for given steering angle"""
            if abs(delta) < 1e-6:
                return float('inf')
            return self.L / np.tan(delta)

    # Create car-like robot
    car = AckermannRobot(wheelbase=2.5)  # 2.5m wheelbase

    print(f"\nCar parameters:")
    print(f"  Wheelbase: {car.L} m")

    # Test turning radii
    print("\nTurning radius vs steering angle:")
    print("-" * 40)
    for delta_deg in [0, 5, 10, 20, 30, 45]:
        delta = np.deg2rad(delta_deg)
        r = car.turning_radius(delta)
        if r == float('inf'):
            print(f"  delta = {delta_deg:3d} deg: r = infinity (straight)")
        else:
            print(f"  delta = {delta_deg:3d} deg: r = {r:.2f} m")

    # Simulate car motion
    state = np.array([0.0, 0.0, 0.0])
    v = 5.0  # 5 m/s
    delta = np.deg2rad(15)  # 15 degree steering
    dt = 0.1
    n_steps = 50

    print(f"\nSimulation: v = {v} m/s, steering = {np.rad2deg(delta):.1f} deg")
    print(f"Turning radius: {car.turning_radius(delta):.2f} m")

    car_trajectory = [state.copy()]
    for _ in range(n_steps):
        state = car.step(state, v, delta, dt)
        car_trajectory.append(state.copy())

    car_trajectory = np.array(car_trajectory)

    print(f"\nInitial state: {car_trajectory[0]}")
    print(f"Final state: {car_trajectory[-1]}")

    # =========================================================================
    # 4. Odometry from wheel encoders
    # =========================================================================
    print("\n--- 4. Wheel Odometry ---")

    print("""
    Wheel odometry estimates robot pose from encoder counts:

    Given:
        - Left encoder ticks: delta_ticks_l
        - Right encoder ticks: delta_ticks_r
        - Ticks per revolution: ticks_per_rev
        - Wheel radius: r
        - Wheel separation: L

    Calculate:
        delta_s_l = 2 * pi * r * delta_ticks_l / ticks_per_rev
        delta_s_r = 2 * pi * r * delta_ticks_r / ticks_per_rev

        delta_s = (delta_s_r + delta_s_l) / 2
        delta_theta = (delta_s_r - delta_s_l) / L

    Update pose:
        x += delta_s * cos(theta + delta_theta/2)
        y += delta_s * sin(theta + delta_theta/2)
        theta += delta_theta
    """)

    class WheelOdometry:
        def __init__(self, wheel_radius, wheel_separation, ticks_per_rev):
            self.r = wheel_radius
            self.L = wheel_separation
            self.ticks_per_rev = ticks_per_rev

            # Current pose
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0

        def update(self, delta_ticks_l, delta_ticks_r):
            """Update pose from encoder ticks"""
            # Convert ticks to distance
            delta_s_l = 2 * np.pi * self.r * delta_ticks_l / self.ticks_per_rev
            delta_s_r = 2 * np.pi * self.r * delta_ticks_r / self.ticks_per_rev

            # Robot displacement
            delta_s = (delta_s_r + delta_s_l) / 2
            delta_theta = (delta_s_r - delta_s_l) / self.L

            # Update pose (midpoint integration)
            self.x += delta_s * np.cos(self.theta + delta_theta / 2)
            self.y += delta_s * np.sin(self.theta + delta_theta / 2)
            self.theta = normalize_angle(self.theta + delta_theta)

            return self.x, self.y, self.theta

    # Simulate odometry
    odom = WheelOdometry(wheel_radius=0.1, wheel_separation=0.5, ticks_per_rev=1000)

    print(f"\nOdometry parameters:")
    print(f"  Wheel radius: {odom.r} m")
    print(f"  Wheel separation: {odom.L} m")
    print(f"  Ticks per revolution: {odom.ticks_per_rev}")

    # Simulate encoder readings (moving forward with slight turn)
    print("\nSimulating encoder readings:")
    print("-" * 50)

    for i in range(5):
        # Simulated encoder readings (100 ticks left, 105 ticks right = slight right turn)
        delta_ticks_l = 100
        delta_ticks_r = 105

        x, y, theta = odom.update(delta_ticks_l, delta_ticks_r)
        print(f"Step {i+1}: ticks=({delta_ticks_l}, {delta_ticks_r}) "
              f"-> pose=({x:.3f}, {y:.3f}, {np.rad2deg(theta):.1f} deg)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Unicycle: simplest model [v, omega]")
    print("  - Differential drive: two wheels, FK/IK between wheel and robot velocities")
    print("  - Ackermann: car-like steering, non-holonomic constraint")
    print("  - Wheel odometry: integrate encoder ticks for pose estimation")
    print("\nThese models are essential for SLAM:")
    print("  - Provide motion model for prediction step")
    print("  - Used in wheel odometry for dead reckoning")
    print("  - Important for motion planning and control")
    print("=" * 60)


if __name__ == "__main__":
    main()
