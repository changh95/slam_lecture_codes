#!/usr/bin/env python3
"""
Basic Python Programming - Part 6: Modules and Virtual Environments
Topics: Imports, packages, virtual environments (venv)
"""

# =============================================================================
# 1. Import Basics
# =============================================================================
print("=== Import Basics ===")

# Import entire module
import math

print(f"math.pi = {math.pi}")
print(f"math.sqrt(16) = {math.sqrt(16)}")

# Import with alias
import numpy as np  # Common alias

arr = np.array([1, 2, 3, 4, 5])
print(f"numpy array: {arr}")

# Import specific items
from os.path import join, exists

path = join("dir", "file.txt")
print(f"Joined path: {path}")

# Import multiple items
from typing import List, Dict, Optional, Tuple

# Import all (not recommended - pollutes namespace)
# from math import *

# =============================================================================
# 2. Module Search Path
# =============================================================================
print("\n=== Module Search Path ===")

import sys

print("Python searches for modules in these locations:")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")
print("  ...")

# Current Python version
print(f"\nPython version: {sys.version}")

# =============================================================================
# 3. Package Structure (Example)
# =============================================================================
print("\n=== Package Structure ===")

print("""
A typical Python package structure:

my_robot_package/
├── __init__.py           # Makes it a package
├── core/
│   ├── __init__.py
│   ├── robot.py
│   └── sensors.py
├── algorithms/
│   ├── __init__.py
│   ├── slam.py
│   └── planning.py
├── utils/
│   ├── __init__.py
│   └── math_utils.py
├── tests/
│   ├── __init__.py
│   └── test_robot.py
├── setup.py              # Package installation config
├── pyproject.toml        # Modern package config
└── requirements.txt      # Dependencies

Import examples:
    from my_robot_package.core.robot import Robot
    from my_robot_package.algorithms import slam
    from my_robot_package.utils.math_utils import normalize_angle
""")

# =============================================================================
# 4. __init__.py Examples
# =============================================================================
print("=== __init__.py Examples ===")

print("""
# my_robot_package/__init__.py

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"

# Convenient imports (users can do: from my_robot_package import Robot)
from .core.robot import Robot
from .core.sensors import Lidar, Camera

# What gets exported with "from package import *"
__all__ = ["Robot", "Lidar", "Camera"]


# my_robot_package/core/__init__.py

from .robot import Robot
from .sensors import Lidar, Camera, IMU

__all__ = ["Robot", "Lidar", "Camera", "IMU"]
""")

# =============================================================================
# 5. Virtual Environments (venv)
# =============================================================================
print("\n=== Virtual Environments (venv) ===")

print("""
Virtual environments isolate project dependencies.
Each project can have its own Python packages without conflicts.

Commands:

# Create a virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\\Scripts\\activate

# Install packages
pip install numpy opencv-python

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate

# Remove (just delete the folder)
rm -rf .venv
""")

# =============================================================================
# 6. requirements.txt Example
# =============================================================================
print("=== requirements.txt Example ===")

print("""
# requirements.txt - Pin versions for reproducibility

# Core scientific computing
numpy==1.24.0
scipy==1.10.0

# Computer vision
opencv-python==4.7.0.72

# Visualization
matplotlib==3.7.0
rerun-sdk>=0.8.0

# Robotics utilities
transforms3d>=0.4.0

# Development tools
pytest>=7.0.0
black>=23.0.0
mypy>=1.0.0

# Version specifiers:
#   ==1.0.0    Exact version
#   >=1.0.0    Minimum version
#   <=1.0.0    Maximum version
#   ~=1.0.0    Compatible release (>=1.0.0, <2.0.0)
#   >=1.0,<2.0 Range
""")

# =============================================================================
# 7. pyproject.toml (Modern Packaging)
# =============================================================================
print("\n=== pyproject.toml Example ===")

print("""
# pyproject.toml - Modern Python packaging (replaces setup.py)

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-robot-package"
version = "1.0.0"
description = "A robotics package for SLAM"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

dependencies = [
    "numpy>=1.20.0",
    "opencv-python>=4.5.0",
    "scipy>=1.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
my-robot = "my_robot_package.cli:main"

[tool.black]
line-length = 88

[tool.mypy]
python_version = "3.10"
strict = true
""")

# =============================================================================
# 8. Common Packages for Robotics/SLAM
# =============================================================================
print("\n=== Common Robotics/SLAM Packages ===")

packages = {
    "numpy": "Numerical computing, arrays",
    "scipy": "Scientific computing, optimization",
    "opencv-python": "Computer vision",
    "matplotlib": "Plotting and visualization",
    "rerun-sdk": "3D visualization for robotics",
    "transforms3d": "3D transformations",
    "spatialmath": "Spatial math for robotics",
    "open3d": "3D data processing",
    "g2o": "Graph optimization (Python bindings)",
    "gtsam": "Factor graphs for SLAM",
}

print("Package             | Description")
print("-" * 50)
for pkg, desc in packages.items():
    print(f"{pkg:<18} | {desc}")

# =============================================================================
# 9. Checking Installed Packages
# =============================================================================
print("\n=== Checking Installed Packages ===")

# Check if package is available
def check_package(name: str) -> bool:
    """Check if a package is installed."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


packages_to_check = ["numpy", "cv2", "scipy", "matplotlib", "rerun"]
print("Package availability:")
for pkg in packages_to_check:
    status = "✓" if check_package(pkg) else "✗"
    print(f"  {status} {pkg}")

# Package versions
print("\nInstalled versions:")
for pkg in ["numpy", "cv2"]:
    if check_package(pkg):
        module = __import__(pkg)
        version = getattr(module, "__version__", "unknown")
        print(f"  {pkg}: {version}")

# =============================================================================
# 10. Best Practices
# =============================================================================
print("\n=== Best Practices ===")

print("""
1. ALWAYS use virtual environments for projects
   - Keeps dependencies isolated
   - Makes projects reproducible

2. Pin versions in requirements.txt
   - Use exact versions for production
   - Use >= for development

3. Organize code into packages
   - Use __init__.py to define public API
   - Keep related code together

4. Use relative imports within packages
   - from .module import Class
   - from ..utils import helper

5. Document dependencies
   - requirements.txt for pip
   - pyproject.toml for packaging

6. NEVER use conda (per course guidelines)
   - Use venv + pip instead
   - More universal and lightweight
""")


if __name__ == "__main__":
    print("\n=== Script Complete ===")
    print("Run: python3 06_modules_venv.py")
