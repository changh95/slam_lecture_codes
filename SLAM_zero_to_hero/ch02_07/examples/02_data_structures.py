#!/usr/bin/env python3
"""
Basic Python Programming - Part 2: Data Structures
Topics: Lists, tuples, dictionaries, sets
"""

# =============================================================================
# 1. Lists - Mutable ordered sequences
# =============================================================================
print("=== Lists ===")

# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, None]
empty = []

print(f"Numbers: {numbers}")
print(f"Mixed types: {mixed}")

# Indexing and slicing
print(f"\nIndexing:")
print(f"  First element: {numbers[0]}")
print(f"  Last element: {numbers[-1]}")
print(f"  Slice [1:3]: {numbers[1:3]}")
print(f"  Slice [::2]: {numbers[::2]}")  # Every 2nd element
print(f"  Reversed: {numbers[::-1]}")

# List operations
print(f"\nList operations:")
numbers.append(6)
print(f"  After append(6): {numbers}")

numbers.insert(0, 0)
print(f"  After insert(0, 0): {numbers}")

numbers.extend([7, 8, 9])
print(f"  After extend([7,8,9]): {numbers}")

removed = numbers.pop()
print(f"  After pop(): {numbers}, removed: {removed}")

numbers.remove(5)
print(f"  After remove(5): {numbers}")

# List methods
print(f"\nList methods:")
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"  Original: {nums}")
print(f"  Length: {len(nums)}")
print(f"  Sum: {sum(nums)}")
print(f"  Min: {min(nums)}, Max: {max(nums)}")
print(f"  Count of 1: {nums.count(1)}")
print(f"  Index of 4: {nums.index(4)}")

nums_sorted = sorted(nums)
print(f"  Sorted (new list): {nums_sorted}")

nums.sort()
print(f"  Sort in-place: {nums}")

# List comprehensions
print(f"\nList comprehensions:")
squares = [x**2 for x in range(5)]
print(f"  Squares: {squares}")

evens = [x for x in range(10) if x % 2 == 0]
print(f"  Evens: {evens}")

# Nested comprehension
matrix = [[i*3 + j for j in range(3)] for i in range(3)]
print(f"  Matrix: {matrix}")

# =============================================================================
# 2. Tuples - Immutable ordered sequences
# =============================================================================
print("\n=== Tuples ===")

# Creating tuples
point = (1.0, 2.0, 3.0)
single = (42,)  # Note the comma
empty_tuple = ()

print(f"Point: {point}")
print(f"Single element tuple: {single}")

# Tuple unpacking
x, y, z = point
print(f"Unpacked: x={x}, y={y}, z={z}")

# Unpacking with *
first, *rest = [1, 2, 3, 4, 5]
print(f"First: {first}, Rest: {rest}")

# Named tuples (better than plain tuples)
from collections import namedtuple

Point3D = namedtuple('Point3D', ['x', 'y', 'z'])
p = Point3D(1.0, 2.0, 3.0)
print(f"\nNamed tuple: {p}")
print(f"Access by name: p.x={p.x}, p.y={p.y}")

# Tuples as dictionary keys (lists can't be keys)
locations = {
    (0, 0): "origin",
    (1, 0): "east",
    (0, 1): "north"
}
print(f"Tuple as dict key: {locations[(0, 0)]}")

# =============================================================================
# 3. Dictionaries - Key-value mappings
# =============================================================================
print("\n=== Dictionaries ===")

# Creating dictionaries
robot = {
    "name": "Turtlebot",
    "wheels": 2,
    "sensors": ["lidar", "camera", "imu"],
    "position": {"x": 0.0, "y": 0.0}
}

print(f"Robot: {robot}")
print(f"Name: {robot['name']}")
print(f"Sensors: {robot['sensors']}")
print(f"Position x: {robot['position']['x']}")

# Safe access with get
print(f"\nSafe access:")
print(f"  robot.get('name'): {robot.get('name')}")
print(f"  robot.get('speed', 1.0): {robot.get('speed', 1.0)}")  # Default value

# Dictionary operations
print(f"\nDictionary operations:")
robot["speed"] = 2.5  # Add new key
print(f"  After adding 'speed': {robot.get('speed')}")

robot.update({"battery": 100, "status": "active"})
print(f"  After update: battery={robot['battery']}, status={robot['status']}")

# Dictionary methods
print(f"\nDictionary methods:")
print(f"  Keys: {list(robot.keys())}")
print(f"  Values: {list(robot.values())[:3]}...")  # First 3 values
print(f"  Items: {list(robot.items())[:2]}...")    # First 2 items

# Iterating over dictionary
print(f"\nIteration:")
for key, value in robot.items():
    if isinstance(value, (str, int, float)):
        print(f"  {key}: {value}")

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
print(f"\nDict comprehension: {squares_dict}")

# =============================================================================
# 4. Sets - Unordered unique elements
# =============================================================================
print("\n=== Sets ===")

# Creating sets
sensors_a = {"lidar", "camera", "imu", "gps"}
sensors_b = {"camera", "radar", "ultrasonic", "imu"}

print(f"Set A: {sensors_a}")
print(f"Set B: {sensors_b}")

# Set operations
print(f"\nSet operations:")
print(f"  Union (A | B): {sensors_a | sensors_b}")
print(f"  Intersection (A & B): {sensors_a & sensors_b}")
print(f"  Difference (A - B): {sensors_a - sensors_b}")
print(f"  Symmetric diff (A ^ B): {sensors_a ^ sensors_b}")

# Set methods
sensors_a.add("depth_camera")
print(f"\nAfter add: {sensors_a}")

sensors_a.discard("gps")  # No error if not found
print(f"After discard: {sensors_a}")

# Membership testing (fast!)
print(f"\n'lidar' in sensors_a: {'lidar' in sensors_a}")

# Remove duplicates from list
duplicates = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique = list(set(duplicates))
print(f"\nRemove duplicates: {duplicates} -> {unique}")

# Frozen sets (immutable, can be dict keys)
frozen = frozenset([1, 2, 3])
print(f"Frozen set: {frozen}")

# =============================================================================
# 5. Copying Data Structures
# =============================================================================
print("\n=== Copying ===")

import copy

original = [[1, 2], [3, 4]]

# Reference (not a copy!)
reference = original
reference[0][0] = 999
print(f"Reference (same object): original={original}")
original[0][0] = 1  # Reset

# Shallow copy
shallow = original.copy()  # or list(original) or original[:]
shallow[0][0] = 888
print(f"Shallow copy: original={original}")  # Inner list affected!
original[0][0] = 1  # Reset

# Deep copy
deep = copy.deepcopy(original)
deep[0][0] = 777
print(f"Deep copy: original={original}")  # Original unchanged!

# =============================================================================
# 6. Common Patterns
# =============================================================================
print("\n=== Common Patterns ===")

# Enumerate - get index and value
items = ["apple", "banana", "cherry"]
print("Enumerate:")
for i, item in enumerate(items):
    print(f"  {i}: {item}")

# Zip - iterate multiple lists together
names = ["Robot1", "Robot2", "Robot3"]
speeds = [1.0, 2.0, 1.5]
print("\nZip:")
for name, speed in zip(names, speeds):
    print(f"  {name}: {speed} m/s")

# Dictionary from two lists
name_speed = dict(zip(names, speeds))
print(f"\nDict from zip: {name_speed}")

# Sorting with key
robots = [("A", 3), ("B", 1), ("C", 2)]
sorted_robots = sorted(robots, key=lambda x: x[1])
print(f"\nSorted by second element: {sorted_robots}")

# Filter and map
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
squared = list(map(lambda x: x**2, numbers))
print(f"\nFilter evens: {evens}")
print(f"Map square: {squared}")


if __name__ == "__main__":
    print("\n=== Script Complete ===")
    print("Run: python3 02_data_structures.py")
