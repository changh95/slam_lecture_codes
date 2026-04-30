#!/usr/bin/env python3
"""
Basic Python Programming - Part 5: File I/O and Exceptions
Topics: Reading/writing files, JSON, exceptions, context managers
"""

import json
import csv
import os
from pathlib import Path
import tempfile

# Create a temporary directory for our examples
TEMP_DIR = tempfile.mkdtemp()
print(f"Using temp directory: {TEMP_DIR}")

# =============================================================================
# 1. Basic File Operations
# =============================================================================
print("\n=== Basic File Operations ===")

# Writing to a file
file_path = os.path.join(TEMP_DIR, "example.txt")

with open(file_path, 'w') as f:
    f.write("Line 1: Hello SLAM!\n")
    f.write("Line 2: This is a test file.\n")
    f.writelines(["Line 3: More content\n", "Line 4: Last line\n"])

print(f"Written to: {file_path}")

# Reading entire file
with open(file_path, 'r') as f:
    content = f.read()
print(f"\nRead entire file:\n{content}")

# Reading line by line
print("Reading line by line:")
with open(file_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        print(f"  {line_num}: {line.strip()}")

# Reading all lines into list
with open(file_path, 'r') as f:
    lines = f.readlines()
print(f"\nLines as list: {[l.strip() for l in lines]}")

# Append mode
with open(file_path, 'a') as f:
    f.write("Line 5: Appended line\n")

# =============================================================================
# 2. Path Operations (pathlib)
# =============================================================================
print("\n=== Path Operations ===")

# Using pathlib (modern approach)
path = Path(TEMP_DIR) / "data" / "sensor_data.txt"
print(f"Path object: {path}")
print(f"Parent: {path.parent}")
print(f"Name: {path.name}")
print(f"Stem: {path.stem}")
print(f"Suffix: {path.suffix}")
print(f"Exists: {path.exists()}")

# Create directory and file
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("Sensor reading: 42.5\n")
print(f"\nCreated: {path}")
print(f"Contents: {path.read_text()}")

# Glob pattern matching
(Path(TEMP_DIR) / "data" / "sensor_001.txt").write_text("data")
(Path(TEMP_DIR) / "data" / "sensor_002.txt").write_text("data")

print("\nFiles matching 'sensor_*.txt':")
for p in Path(TEMP_DIR).glob("**/*.txt"):
    print(f"  {p}")

# =============================================================================
# 3. JSON Files
# =============================================================================
print("\n=== JSON Files ===")

# Robot configuration as dict
robot_config = {
    "name": "Turtlebot",
    "wheels": 2,
    "sensors": ["lidar", "camera", "imu"],
    "position": {"x": 0.0, "y": 0.0, "theta": 0.0},
    "parameters": {
        "max_speed": 2.0,
        "max_angular_speed": 1.5
    }
}

# Write JSON
json_path = Path(TEMP_DIR) / "robot_config.json"
with open(json_path, 'w') as f:
    json.dump(robot_config, f, indent=2)
print(f"Written JSON to: {json_path}")

# Read JSON
with open(json_path, 'r') as f:
    loaded_config = json.load(f)
print(f"Loaded: {loaded_config['name']} with {loaded_config['wheels']} wheels")

# JSON string conversion
json_str = json.dumps(robot_config, indent=2)
print(f"\nJSON string:\n{json_str[:100]}...")

# Parse JSON string
data = json.loads('{"x": 1.0, "y": 2.0}')
print(f"\nParsed: {data}")

# =============================================================================
# 4. CSV Files
# =============================================================================
print("\n=== CSV Files ===")

# Write CSV
csv_path = Path(TEMP_DIR) / "trajectory.csv"
trajectory = [
    {"timestamp": 0.0, "x": 0.0, "y": 0.0, "theta": 0.0},
    {"timestamp": 0.1, "x": 0.1, "y": 0.0, "theta": 0.1},
    {"timestamp": 0.2, "x": 0.2, "y": 0.1, "theta": 0.2},
    {"timestamp": 0.3, "x": 0.3, "y": 0.2, "theta": 0.3},
]

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp", "x", "y", "theta"])
    writer.writeheader()
    writer.writerows(trajectory)
print(f"Written CSV to: {csv_path}")

# Read CSV
print("\nReading CSV:")
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  t={row['timestamp']}: ({row['x']}, {row['y']}, {row['theta']})")

# Simple CSV without headers
simple_csv = Path(TEMP_DIR) / "simple.csv"
with open(simple_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([1, 2, 3])
    writer.writerow([4, 5, 6])

with open(simple_csv, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
print(f"\nSimple CSV: {rows}")

# =============================================================================
# 5. Exception Handling
# =============================================================================
print("\n=== Exception Handling ===")


def divide(a, b):
    """Division with proper error handling."""
    try:
        result = a / b
    except ZeroDivisionError:
        print("  Error: Division by zero!")
        return None
    except TypeError as e:
        print(f"  Error: Invalid types - {e}")
        return None
    else:
        # Executed if no exception
        print(f"  Success: {a} / {b} = {result}")
        return result
    finally:
        # Always executed
        print("  (cleanup would go here)")


divide(10, 2)
divide(10, 0)
divide("10", 2)


# Multiple exceptions
def risky_operation(data):
    """Handle multiple exception types."""
    try:
        value = int(data["value"])
        result = 100 / value
        return result
    except KeyError:
        print("  Missing 'value' key")
    except ValueError:
        print("  Invalid value format")
    except ZeroDivisionError:
        print("  Value is zero")
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")
    return None


print("\nMultiple exceptions:")
risky_operation({"value": "10"})
risky_operation({})
risky_operation({"value": "abc"})
risky_operation({"value": "0"})


# Custom exceptions
class RobotError(Exception):
    """Base exception for robot errors."""
    pass


class CollisionError(RobotError):
    """Raised when robot collides."""
    def __init__(self, position):
        self.position = position
        super().__init__(f"Collision detected at {position}")


class BatteryLowError(RobotError):
    """Raised when battery is low."""
    def __init__(self, level):
        self.level = level
        super().__init__(f"Battery low: {level}%")


def move_robot(distance, battery_level, obstacle_at=None):
    """Move robot with exception handling."""
    if battery_level < 10:
        raise BatteryLowError(battery_level)
    if obstacle_at is not None and distance >= obstacle_at:
        raise CollisionError((obstacle_at, 0))
    return f"Moved {distance}m"


print("\nCustom exceptions:")
try:
    print(f"  {move_robot(10, 50)}")
    print(f"  {move_robot(10, 5)}")
except BatteryLowError as e:
    print(f"  Battery error: {e}")
except CollisionError as e:
    print(f"  Collision error: {e}")

# =============================================================================
# 6. Context Managers
# =============================================================================
print("\n=== Context Managers ===")


# Custom context manager using class
class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name="Timer"):
        self.name = name

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.perf_counter() - self.start
        print(f"  {self.name}: {self.elapsed * 1000:.2f} ms")
        return False  # Don't suppress exceptions


with Timer("List creation"):
    data = [i ** 2 for i in range(10000)]

with Timer("Sum"):
    total = sum(data)


# Context manager using generator
from contextlib import contextmanager


@contextmanager
def working_directory(path):
    """Temporarily change working directory."""
    import os
    original = os.getcwd()
    try:
        os.chdir(path)
        print(f"  Changed to: {os.getcwd()}")
        yield
    finally:
        os.chdir(original)
        print(f"  Restored to: {os.getcwd()}")


print("\nWorking directory context manager:")
with working_directory(TEMP_DIR):
    print(f"  Inside context: {os.getcwd()}")


# File-like context manager
@contextmanager
def open_with_logging(filename, mode='r'):
    """Open file with logging."""
    print(f"  Opening {filename} in '{mode}' mode")
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()
        print(f"  Closed {filename}")


print("\nFile context manager with logging:")
with open_with_logging(file_path, 'r') as f:
    first_line = f.readline()
    print(f"  First line: {first_line.strip()}")

# =============================================================================
# 7. Binary Files
# =============================================================================
print("\n=== Binary Files ===")

import struct

# Write binary data
binary_path = Path(TEMP_DIR) / "data.bin"

# Pack: 3 floats and 1 int
data = struct.pack('fffi', 1.5, 2.5, 3.5, 42)
with open(binary_path, 'wb') as f:
    f.write(data)
print(f"Written binary data: {len(data)} bytes")

# Read binary data
with open(binary_path, 'rb') as f:
    data = f.read()
values = struct.unpack('fffi', data)
print(f"Read back: {values}")

# =============================================================================
# Cleanup
# =============================================================================
print(f"\n=== Cleanup ===")
print(f"Temp directory: {TEMP_DIR}")
print("(Not deleted for inspection)")


if __name__ == "__main__":
    print("\n=== Script Complete ===")
    print("Run: python3 05_file_io.py")
