# Forward/Inverse Kinematics with Python Robotics Toolbox

This tutorial demonstrates robot kinematics using Peter Corke's Python Robotics Toolbox.

## Topics Covered

- Denavit-Hartenberg (DH) parameters
- Forward kinematics (FK)
- Inverse kinematics (IK)
- Jacobian matrix and velocity kinematics
- Trajectory planning
- Mobile robot kinematics

## Installation

### Using Virtual Environment

```bash
cd 1_28
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Examples

```bash
cd examples
python3 01_robot_creation.py
python3 02_forward_kinematics.py
python3 03_inverse_kinematics.py
python3 04_jacobian.py
python3 05_trajectory_planning.py
python3 06_mobile_robot_kinematics.py
```

## Examples

| File | Description |
|------|-------------|
| `01_robot_creation.py` | Create robots using DH parameters and predefined models |
| `02_forward_kinematics.py` | Calculate end-effector pose from joint angles |
| `03_inverse_kinematics.py` | Solve for joint angles given target pose |
| `04_jacobian.py` | Velocity kinematics and singularity analysis |
| `05_trajectory_planning.py` | Joint and Cartesian space trajectories |
| `06_mobile_robot_kinematics.py` | Differential drive, Ackermann steering, unicycle models |

## Why Kinematics for SLAM?

Understanding robot kinematics is important for SLAM because:

1. **Wheel Odometry**: Mobile robot kinematics defines how wheel velocities translate to robot motion
2. **Manipulator SLAM**: Robot arms need FK/IK for manipulation tasks integrated with SLAM
3. **Leg Odometry**: Quadruped/humanoid robots use kinematics for leg odometry
4. **Sensor Placement**: Camera/LiDAR placement on robot arms requires forward kinematics

## References

- [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)
- [Robotics, Vision and Control - Peter Corke](https://petercorke.com/rvc/)
- [Spatial Math for Python](https://github.com/petercorke/spatialmath-python)
