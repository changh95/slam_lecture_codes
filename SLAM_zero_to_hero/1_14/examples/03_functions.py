#!/usr/bin/env python3
"""
Basic Python Programming - Part 3: Functions
Topics: Function definitions, arguments, return values, decorators, lambdas
"""

import functools
import time
from typing import List, Tuple, Optional, Callable

# =============================================================================
# 1. Basic Functions
# =============================================================================
print("=== Basic Functions ===")


def greet(name: str) -> str:
    """Simple function with type hints and docstring."""
    return f"Hello, {name}!"


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


print(greet("SLAM Developer"))
print(f"add(3, 5) = {add(3, 5)}")

# =============================================================================
# 2. Default Arguments
# =============================================================================
print("\n=== Default Arguments ===")


def create_robot(name: str, wheels: int = 4, sensors: List[str] = None) -> dict:
    """Create robot configuration with defaults.

    Note: Never use mutable default arguments like sensors=[]
    Always use None and create new list inside function.
    """
    if sensors is None:
        sensors = []
    return {"name": name, "wheels": wheels, "sensors": sensors}


robot1 = create_robot("Turtlebot")
robot2 = create_robot("Quadruped", wheels=4, sensors=["lidar", "camera"])

print(f"Robot 1: {robot1}")
print(f"Robot 2: {robot2}")

# =============================================================================
# 3. *args and **kwargs
# =============================================================================
print("\n=== *args and **kwargs ===")


def sum_all(*args):
    """Sum any number of arguments."""
    return sum(args)


def print_config(**kwargs):
    """Print configuration from keyword arguments."""
    for key, value in kwargs.items():
        print(f"  {key}: {value}")


print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")

print("\nprint_config output:")
print_config(robot="Turtlebot", speed=2.0, autonomous=True)


def flexible_function(*args, **kwargs):
    """Function that accepts any arguments."""
    print(f"  args: {args}")
    print(f"  kwargs: {kwargs}")


print("\nflexible_function(1, 2, a='x', b='y'):")
flexible_function(1, 2, a='x', b='y')

# =============================================================================
# 4. Multiple Return Values
# =============================================================================
print("\n=== Multiple Return Values ===")


def get_robot_state() -> Tuple[float, float, float]:
    """Return position as tuple."""
    x, y, theta = 1.0, 2.0, 0.5
    return x, y, theta


x, y, theta = get_robot_state()
print(f"Position: x={x}, y={y}, theta={theta}")


def process_data(data: List[float]) -> Tuple[float, float, float]:
    """Return multiple statistics."""
    return min(data), max(data), sum(data) / len(data)


data = [1.0, 2.0, 3.0, 4.0, 5.0]
min_val, max_val, avg = process_data(data)
print(f"Stats: min={min_val}, max={max_val}, avg={avg}")

# =============================================================================
# 5. Lambda Functions
# =============================================================================
print("\n=== Lambda Functions ===")

# Simple lambda
square = lambda x: x ** 2
print(f"square(5) = {square(5)}")

# Lambda with multiple arguments
distance = lambda x1, y1, x2, y2: ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
print(f"distance(0, 0, 3, 4) = {distance(0, 0, 3, 4)}")

# Lambda in sorting
robots = [("A", 3), ("B", 1), ("C", 2)]
sorted_robots = sorted(robots, key=lambda r: r[1])
print(f"Sorted by priority: {sorted_robots}")

# Lambda with filter and map
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
doubled = list(map(lambda x: x * 2, numbers))
print(f"Evens: {evens}")
print(f"Doubled: {doubled}")

# =============================================================================
# 6. Higher-Order Functions
# =============================================================================
print("\n=== Higher-Order Functions ===")


def apply_operation(data: List[float], operation: Callable[[float], float]) -> List[float]:
    """Apply a function to each element."""
    return [operation(x) for x in data]


import math

data = [0, 0.5, 1.0, 1.5, 2.0]
sin_values = apply_operation(data, math.sin)
print(f"sin({data}) = {[round(v, 3) for v in sin_values]}")


def create_multiplier(factor: float) -> Callable[[float], float]:
    """Return a function that multiplies by factor."""
    def multiplier(x: float) -> float:
        return x * factor
    return multiplier


double = create_multiplier(2)
triple = create_multiplier(3)
print(f"double(5) = {double(5)}")
print(f"triple(5) = {triple(5)}")

# =============================================================================
# 7. Decorators
# =============================================================================
print("\n=== Decorators ===")


def timer(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"  [{func.__name__}] took {(end - start) * 1000:.2f} ms")
        return result
    return wrapper


def debug(func):
    """Decorator to print function calls and returns."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"  {func.__name__} returned {result}")
        return result
    return wrapper


@timer
def slow_function():
    """A function that takes some time."""
    time.sleep(0.01)
    return "done"


@debug
def multiply(a, b):
    return a * b


print("Timer decorator:")
slow_function()

print("\nDebug decorator:")
multiply(3, 4)


# Decorator with arguments
def repeat(times: int):
    """Decorator factory that repeats function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


@repeat(3)
def say_hello():
    print("  Hello!")


print("\nRepeat decorator (3 times):")
say_hello()

# =============================================================================
# 8. Closures
# =============================================================================
print("\n=== Closures ===")


def make_counter():
    """Create a counter using closure."""
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter


counter = make_counter()
print(f"counter() = {counter()}")
print(f"counter() = {counter()}")
print(f"counter() = {counter()}")


def make_pid_controller(kp: float, ki: float, kd: float):
    """Create a PID controller using closure."""
    integral = 0.0
    prev_error = 0.0

    def control(error: float, dt: float) -> float:
        nonlocal integral, prev_error
        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0
        prev_error = error

        return kp * error + ki * integral + kd * derivative

    return control


pid = make_pid_controller(1.0, 0.1, 0.01)
print(f"\nPID controller output:")
print(f"  error=1.0: {pid(1.0, 0.1):.3f}")
print(f"  error=0.5: {pid(0.5, 0.1):.3f}")
print(f"  error=0.2: {pid(0.2, 0.1):.3f}")

# =============================================================================
# 9. Generators
# =============================================================================
print("\n=== Generators ===")


def fibonacci(n: int):
    """Generate first n Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


print(f"First 10 Fibonacci numbers: {list(fibonacci(10))}")


def sensor_readings(count: int):
    """Simulate sensor readings (generator saves memory)."""
    import random
    for i in range(count):
        yield {"id": i, "value": random.random()}


print("\nSimulated sensor readings:")
for reading in sensor_readings(3):
    print(f"  {reading}")

# Generator expression (like list comprehension but lazy)
squares_gen = (x ** 2 for x in range(1000000))
print(f"\nGenerator expression (lazy): {squares_gen}")
print(f"First 5 values: {[next(squares_gen) for _ in range(5)]}")


if __name__ == "__main__":
    print("\n=== Script Complete ===")
    print("Run: python3 03_functions.py")
