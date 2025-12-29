#!/usr/bin/env python3
"""
Writing Python Like It's Rust

This module demonstrates Rust-inspired Python patterns for more robust code:
1. Comprehensive type hints
2. Dataclasses over dicts/tuples
3. Union types (Algebraic Data Types)
4. Newtypes for domain specificity
5. Named constructors
6. Runtime invariant verification

Reference: https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Literal, NewType, TypeAlias, Optional
from enum import Enum, auto
import math

# =============================================================================
# 1. Comprehensive Type Hints
# =============================================================================
print("=== 1. Comprehensive Type Hints ===\n")


# Bad: No types - what does this return?
def process_data_bad(items, threshold):
    return [x for x in items if x > threshold]


# Good: Clear types - API is self-documenting
def process_sensor_readings(
    readings: list[float],
    threshold: float,
    *,  # Force keyword arguments after this
    include_zeros: bool = False,
) -> list[float]:
    """Filter sensor readings above threshold."""
    if include_zeros:
        return [r for r in readings if r >= threshold or r == 0.0]
    return [r for r in readings if r > threshold]


readings = [0.1, 0.5, 0.0, 1.2, 0.3]
filtered = process_sensor_readings(readings, threshold=0.4, include_zeros=True)
print(f"Filtered readings: {filtered}")

# =============================================================================
# 2. Dataclasses Over Dicts/Tuples
# =============================================================================
print("\n=== 2. Dataclasses Over Dicts/Tuples ===\n")


# Bad: What are the fields? Easy to mix up order
def get_robot_state_bad():
    return (1.5, 2.3, 0.5, True)  # What is what?


# Good: Named, typed fields
@dataclass(frozen=True)  # frozen = immutable like Rust
class RobotState:
    x: float
    y: float
    theta: float
    is_active: bool

    @property
    def position(self) -> tuple[float, float]:
        """Get position as tuple."""
        return (self.x, self.y)

    def distance_to(self, other: RobotState) -> float:
        """Euclidean distance to another robot."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


state = RobotState(x=1.5, y=2.3, theta=0.5, is_active=True)
print(f"Robot state: {state}")
print(f"Position: {state.position}")

# IDE autocomplete works, refactoring is safe
# state.z  # Would show error in IDE

# =============================================================================
# 3. Union Types (Algebraic Data Types)
# =============================================================================
print("\n=== 3. Union Types (ADTs) ===\n")


# Model mutually exclusive states as separate types
@dataclass(frozen=True)
class SensorOk:
    value: float
    timestamp: float


@dataclass(frozen=True)
class SensorError:
    error_code: int
    message: str


@dataclass(frozen=True)
class SensorTimeout:
    last_seen: float


# Union type - sensor reading is one of these
SensorReading: TypeAlias = Union[SensorOk, SensorError, SensorTimeout]


def process_reading(reading: SensorReading) -> str:
    """Process sensor reading - exhaustive matching."""
    match reading:
        case SensorOk(value=v, timestamp=t):
            return f"OK: {v:.2f} at t={t:.1f}"
        case SensorError(error_code=code, message=msg):
            return f"ERROR {code}: {msg}"
        case SensorTimeout(last_seen=t):
            return f"TIMEOUT: last seen at t={t:.1f}"


# Usage
readings: list[SensorReading] = [
    SensorOk(value=23.5, timestamp=100.0),
    SensorError(error_code=42, message="Calibration failed"),
    SensorTimeout(last_seen=95.0),
]

for r in readings:
    print(f"  {process_reading(r)}")

# =============================================================================
# 4. Newtypes for Domain Specificity
# =============================================================================
print("\n=== 4. Newtypes ===\n")

# Problem: Easy to mix up IDs
# def assign_driver(car_id: int, driver_id: int): ...
# assign_driver(driver_id, car_id)  # Bug! Swapped arguments

# Solution: Newtypes
CarId = NewType("CarId", int)
DriverId = NewType("DriverId", int)


def assign_driver(car_id: CarId, driver_id: DriverId) -> None:
    print(f"Assigned driver {driver_id} to car {car_id}")


car = CarId(1)
driver = DriverId(42)

assign_driver(car, driver)  # OK
# assign_driver(driver, car)  # Type checker will warn!


# More examples for robotics
FrameId = NewType("FrameId", int)
LandmarkId = NewType("LandmarkId", int)
Timestamp = NewType("Timestamp", float)


@dataclass(frozen=True)
class Observation:
    frame_id: FrameId
    landmark_id: LandmarkId
    timestamp: Timestamp
    pixel_x: float
    pixel_y: float


obs = Observation(
    frame_id=FrameId(10),
    landmark_id=LandmarkId(42),
    timestamp=Timestamp(1.5),
    pixel_x=320.0,
    pixel_y=240.0,
)
print(f"Observation: {obs}")

# =============================================================================
# 5. Named Constructors
# =============================================================================
print("\n=== 5. Named Constructors ===\n")


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be < x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be < y_max ({self.y_max})")

    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
        """Create from two corner points (order doesn't matter)."""
        return cls(
            x_min=min(x1, x2),
            y_min=min(y1, y2),
            x_max=max(x1, x2),
            y_max=max(y1, y2),
        )

    @classmethod
    def from_center_size(
        cls, cx: float, cy: float, width: float, height: float
    ) -> BoundingBox:
        """Create from center point and size."""
        return cls(
            x_min=cx - width / 2,
            y_min=cy - height / 2,
            x_max=cx + width / 2,
            y_max=cy + height / 2,
        )

    @classmethod
    def from_yolo(
        cls, cx: float, cy: float, w: float, h: float, img_width: int, img_height: int
    ) -> BoundingBox:
        """Create from YOLO format (normalized center + size)."""
        return cls.from_center_size(
            cx * img_width,
            cy * img_height,
            w * img_width,
            h * img_height,
        )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


# Different ways to create the same box
box1 = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70)
box2 = BoundingBox.from_corners(110, 70, 10, 20)  # Order doesn't matter
box3 = BoundingBox.from_center_size(60, 45, 100, 50)
box4 = BoundingBox.from_yolo(0.3, 0.225, 0.5, 0.25, 200, 200)

print(f"From min/max:    {box1}")
print(f"From corners:    {box2}")
print(f"From center:     {box3}")
print(f"From YOLO:       {box4}")

# =============================================================================
# 6. Enums for State Machines
# =============================================================================
print("\n=== 6. Enums for State Machines ===\n")


class RobotMode(Enum):
    IDLE = auto()
    MAPPING = auto()
    LOCALIZING = auto()
    NAVIGATING = auto()
    EMERGENCY_STOP = auto()


@dataclass
class Robot:
    name: str
    mode: RobotMode = RobotMode.IDLE

    def start_mapping(self) -> None:
        if self.mode not in (RobotMode.IDLE, RobotMode.LOCALIZING):
            raise RuntimeError(f"Cannot start mapping from {self.mode}")
        self.mode = RobotMode.MAPPING
        print(f"{self.name}: Started mapping")

    def start_navigation(self, target: tuple[float, float]) -> None:
        if self.mode != RobotMode.LOCALIZING:
            raise RuntimeError(f"Must be localizing to navigate, currently {self.mode}")
        self.mode = RobotMode.NAVIGATING
        print(f"{self.name}: Navigating to {target}")

    def emergency_stop(self) -> None:
        self.mode = RobotMode.EMERGENCY_STOP
        print(f"{self.name}: EMERGENCY STOP!")


robot = Robot("Turtlebot")
print(f"Initial mode: {robot.mode}")

robot.start_mapping()
print(f"Current mode: {robot.mode}")

# =============================================================================
# 7. Optional with Default
# =============================================================================
print("\n=== 7. Optional Handling ===\n")


@dataclass
class SensorConfig:
    name: str
    frequency_hz: float
    timeout_ms: Optional[float] = None  # None means no timeout

    @property
    def timeout_or_default(self) -> float:
        """Get timeout with sensible default."""
        return self.timeout_ms if self.timeout_ms is not None else 1000.0


config1 = SensorConfig(name="lidar", frequency_hz=10.0, timeout_ms=500.0)
config2 = SensorConfig(name="camera", frequency_hz=30.0)  # No timeout

print(f"Lidar timeout: {config1.timeout_or_default} ms")
print(f"Camera timeout: {config2.timeout_or_default} ms (default)")

# =============================================================================
# 8. Result Type Pattern
# =============================================================================
print("\n=== 8. Result Type Pattern ===\n")


@dataclass(frozen=True)
class Ok[T]:
    value: T


@dataclass(frozen=True)
class Err[E]:
    error: E


Result: TypeAlias = Union[Ok[any], Err[any]]


def divide(a: float, b: float) -> Result:
    """Division that returns Result instead of raising."""
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)


def sqrt_safe(x: float) -> Result:
    """Square root that handles negatives."""
    if x < 0:
        return Err(f"Cannot take sqrt of negative number: {x}")
    return Ok(math.sqrt(x))


# Usage
results = [divide(10, 2), divide(10, 0), sqrt_safe(16), sqrt_safe(-4)]

for r in results:
    match r:
        case Ok(value=v):
            print(f"  Success: {v}")
        case Err(error=e):
            print(f"  Error: {e}")

# =============================================================================
# 9. Invariant Validation in __post_init__
# =============================================================================
print("\n=== 9. Invariant Validation ===\n")


@dataclass
class NormalizedBoundingBox:
    """Bounding box with coordinates in [0, 1]."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name, value in [
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        if self.x + self.width > 1.0:
            raise ValueError(f"x + width ({self.x + self.width}) exceeds 1.0")
        if self.y + self.height > 1.0:
            raise ValueError(f"y + height ({self.y + self.height}) exceeds 1.0")


# Valid box
box = NormalizedBoundingBox(x=0.1, y=0.2, width=0.5, height=0.3)
print(f"Valid box: {box}")

# Invalid box - caught immediately
try:
    invalid_box = NormalizedBoundingBox(x=0.8, y=0.2, width=0.5, height=0.3)
except ValueError as e:
    print(f"Caught invalid box: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n=== Summary: Rust-like Python Patterns ===")
print("""
1. Type everything - function signatures, variables, return types
2. Use dataclasses instead of dicts/tuples
3. Model states as Union types (ADTs)
4. Use NewType for domain-specific IDs
5. Create named constructors (from_x, from_y)
6. Use Enums for state machines
7. Handle Optional explicitly
8. Consider Result type for error handling
9. Validate invariants in __post_init__

Benefits:
- Earlier error detection (type checker finds bugs)
- Better IDE support (autocomplete, refactoring)
- Self-documenting code
- Safer refactoring
""")


if __name__ == "__main__":
    print("Run: python3 07_rustlike_python.py")
