#!/usr/bin/env python3
"""
Basic Python Programming - Part 4: Classes and OOP
Topics: Classes, inheritance, magic methods, dataclasses
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import math

# =============================================================================
# 1. Basic Classes
# =============================================================================
print("=== Basic Classes ===")


class Robot:
    """A simple robot class."""

    # Class attribute (shared by all instances)
    robot_count = 0

    def __init__(self, name: str, x: float = 0.0, y: float = 0.0):
        """Initialize robot with name and position."""
        self.name = name        # Instance attribute
        self.x = x
        self.y = y
        self._battery = 100.0   # Convention: private (single underscore)
        self.__secret = "hidden"  # Name mangling (double underscore)

        Robot.robot_count += 1

    def move(self, dx: float, dy: float) -> None:
        """Move robot by delta."""
        self.x += dx
        self.y += dy
        self._battery -= 1.0

    def get_position(self) -> tuple:
        """Get current position."""
        return (self.x, self.y)

    def __str__(self) -> str:
        """String representation for print()."""
        return f"Robot({self.name} at ({self.x:.1f}, {self.y:.1f}))"

    def __repr__(self) -> str:
        """Technical representation for debugging."""
        return f"Robot(name='{self.name}', x={self.x}, y={self.y})"


# Create and use robots
robot1 = Robot("Turtlebot", 0, 0)
robot2 = Robot("Drone", 5, 10)

print(f"robot1: {robot1}")
print(f"robot2: {robot2}")
print(f"Total robots: {Robot.robot_count}")

robot1.move(1, 2)
print(f"After move: {robot1}")
print(f"Position: {robot1.get_position()}")

# Accessing "private" attributes (Python doesn't enforce privacy)
print(f"Battery (convention private): {robot1._battery}")
# print(f"Secret: {robot1.__secret}")  # Would raise AttributeError
print(f"Secret (name mangled): {robot1._Robot__secret}")  # Works but don't do this!

# =============================================================================
# 2. Properties
# =============================================================================
print("\n=== Properties ===")


class Sensor:
    """Sensor class with properties."""

    def __init__(self, name: str, max_range: float):
        self.name = name
        self._max_range = max_range
        self._enabled = False

    @property
    def max_range(self) -> float:
        """Read-only property."""
        return self._max_range

    @property
    def enabled(self) -> bool:
        """Getter for enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Setter for enabled with validation."""
        if not isinstance(value, bool):
            raise ValueError("enabled must be a boolean")
        self._enabled = value
        print(f"  Sensor {self.name} {'enabled' if value else 'disabled'}")


sensor = Sensor("Lidar", 100.0)
print(f"Max range: {sensor.max_range}")

sensor.enabled = True
sensor.enabled = False

# sensor.max_range = 200.0  # Would raise AttributeError (read-only)

# =============================================================================
# 3. Inheritance
# =============================================================================
print("\n=== Inheritance ===")


class Vehicle:
    """Base vehicle class."""

    def __init__(self, name: str):
        self.name = name

    def start(self) -> str:
        return f"{self.name} starting..."

    def stop(self) -> str:
        return f"{self.name} stopping..."


class GroundRobot(Vehicle):
    """Ground robot inheriting from Vehicle."""

    def __init__(self, name: str, wheels: int):
        super().__init__(name)  # Call parent constructor
        self.wheels = wheels

    def start(self) -> str:
        # Override parent method
        return f"{self.name} (ground robot with {self.wheels} wheels) starting..."

    def navigate(self, x: float, y: float) -> str:
        return f"{self.name} navigating to ({x}, {y})"


class FlyingRobot(Vehicle):
    """Flying robot inheriting from Vehicle."""

    def __init__(self, name: str, rotors: int):
        super().__init__(name)
        self.rotors = rotors
        self.altitude = 0.0

    def start(self) -> str:
        return f"{self.name} (drone with {self.rotors} rotors) taking off..."

    def fly_to(self, x: float, y: float, z: float) -> str:
        self.altitude = z
        return f"{self.name} flying to ({x}, {y}, {z})"


ground = GroundRobot("Turtlebot", 2)
drone = FlyingRobot("Quadcopter", 4)

print(ground.start())
print(ground.navigate(10, 20))
print()
print(drone.start())
print(drone.fly_to(5, 5, 10))

# isinstance and issubclass
print(f"\nisinstance(ground, Vehicle): {isinstance(ground, Vehicle)}")
print(f"isinstance(drone, GroundRobot): {isinstance(drone, GroundRobot)}")
print(f"issubclass(FlyingRobot, Vehicle): {issubclass(FlyingRobot, Vehicle)}")

# =============================================================================
# 4. Abstract Base Classes
# =============================================================================
print("\n=== Abstract Base Classes ===")


class SensorInterface(ABC):
    """Abstract base class for sensors."""

    @abstractmethod
    def read(self) -> float:
        """Read sensor value (must be implemented)."""
        pass

    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate sensor (must be implemented)."""
        pass

    def status(self) -> str:
        """Common method (can be inherited)."""
        return "Sensor operational"


class DistanceSensor(SensorInterface):
    """Concrete distance sensor implementation."""

    def __init__(self, max_distance: float):
        self.max_distance = max_distance
        self._calibrated = False

    def read(self) -> float:
        import random
        return random.uniform(0, self.max_distance)

    def calibrate(self) -> bool:
        self._calibrated = True
        return True


# Cannot instantiate abstract class
# sensor = SensorInterface()  # Would raise TypeError

distance_sensor = DistanceSensor(10.0)
distance_sensor.calibrate()
print(f"Distance reading: {distance_sensor.read():.2f}")
print(f"Status: {distance_sensor.status()}")

# =============================================================================
# 5. Magic Methods (Dunder Methods)
# =============================================================================
print("\n=== Magic Methods ===")


class Vector3:
    """3D vector with magic methods."""

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __eq__(self, other: 'Vector3') -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        raise IndexError("Vector3 index out of range")

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __str__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


v1 = Vector3(1, 2, 3)
v2 = Vector3(4, 5, 6)

print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")
print(f"v1 * 2 = {v1 * 2}")
print(f"v1 == v2: {v1 == v2}")
print(f"len(v1): {len(v1)}")
print(f"v1[0]: {v1[0]}")
print(f"Unpack: {list(v1)}")
print(f"Magnitude: {v1.magnitude():.3f}")

# =============================================================================
# 6. Dataclasses (Python 3.7+)
# =============================================================================
print("\n=== Dataclasses ===")


@dataclass
class Point:
    """Simple point using dataclass."""
    x: float
    y: float
    z: float = 0.0  # Default value


@dataclass
class MapPoint:
    """Map point with more features."""
    id: int
    position: Point
    descriptor: List[float] = field(default_factory=list)
    observations: int = 0

    def __post_init__(self):
        """Called after __init__."""
        if not self.descriptor:
            self.descriptor = [0.0] * 128


# Dataclasses automatically generate __init__, __repr__, __eq__
p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0, 3.0)

print(f"p1 = {p1}")
print(f"p2 = {p2}")
print(f"p1 == Point(1.0, 2.0): {p1 == Point(1.0, 2.0)}")

mp = MapPoint(id=1, position=Point(5.0, 10.0, 1.0))
print(f"\nMapPoint: {mp}")
print(f"Descriptor length: {len(mp.descriptor)}")


# Frozen dataclass (immutable)
@dataclass(frozen=True)
class ImmutableConfig:
    """Immutable configuration."""
    name: str
    version: str


config = ImmutableConfig("SLAM", "1.0")
print(f"\nImmutable config: {config}")
# config.name = "new"  # Would raise FrozenInstanceError

# =============================================================================
# 7. Class Methods and Static Methods
# =============================================================================
print("\n=== Class Methods and Static Methods ===")


class Pose:
    """Robot pose with class and static methods."""

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    @classmethod
    def from_tuple(cls, pose_tuple: tuple) -> 'Pose':
        """Alternative constructor from tuple."""
        return cls(*pose_tuple)

    @classmethod
    def identity(cls) -> 'Pose':
        """Create identity pose."""
        return cls(0.0, 0.0, 0.0)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi] (no instance needed)."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def __str__(self) -> str:
        return f"Pose(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f})"


# Regular constructor
pose1 = Pose(1.0, 2.0, 0.5)
print(f"Regular: {pose1}")

# Class method constructors
pose2 = Pose.from_tuple((3.0, 4.0, 1.0))
print(f"From tuple: {pose2}")

pose3 = Pose.identity()
print(f"Identity: {pose3}")

# Static method (can be called without instance)
angle = Pose.normalize_angle(4.0)
print(f"Normalized 4.0 rad: {angle:.3f}")


if __name__ == "__main__":
    print("\n=== Script Complete ===")
    print("Run: python3 04_classes.py")
