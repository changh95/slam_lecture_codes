#!/usr/bin/env python3
"""
Basic Python Programming - Part 1: Fundamentals
Topics: Variables, data types, operators, control flow
"""

# =============================================================================
# 1. Variables and Data Types
# =============================================================================
print("=== Variables and Data Types ===")

# Numbers
integer_num = 42
float_num = 3.14159
complex_num = 3 + 4j

print(f"Integer: {integer_num}, type: {type(integer_num).__name__}")
print(f"Float: {float_num}, type: {type(float_num).__name__}")
print(f"Complex: {complex_num}, type: {type(complex_num).__name__}")

# Strings
name = "SLAM"
multiline = """This is a
multiline string"""

print(f"String: {name}")
print(f"Length: {len(name)}")
print(f"Uppercase: {name.upper()}")
print(f"Lowercase: {name.lower()}")

# Booleans
is_running = True
is_stopped = False
print(f"Booleans: {is_running}, {is_stopped}")

# None (null equivalent)
result = None
print(f"None type: {result}, is None: {result is None}")

# =============================================================================
# 2. Type Conversion
# =============================================================================
print("\n=== Type Conversion ===")

# String to number
x = int("42")
y = float("3.14")
print(f"int('42') = {x}, float('3.14') = {y}")

# Number to string
s = str(42)
print(f"str(42) = '{s}'")

# Boolean conversion
print(f"bool(0) = {bool(0)}, bool(1) = {bool(1)}")
print(f"bool('') = {bool('')}, bool('hello') = {bool('hello')}")
print(f"bool([]) = {bool([])}, bool([1,2]) = {bool([1,2])}")

# =============================================================================
# 3. Operators
# =============================================================================
print("\n=== Operators ===")

# Arithmetic
a, b = 17, 5
print(f"a={a}, b={b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")      # Float division
print(f"a // b = {a // b}")    # Integer division
print(f"a % b = {a % b}")      # Modulo
print(f"a ** b = {a ** b}")    # Power

# Comparison
print(f"\na == b: {a == b}")
print(f"a != b: {a != b}")
print(f"a > b: {a > b}")
print(f"a >= b: {a >= b}")

# Logical
x, y = True, False
print(f"\nx and y: {x and y}")
print(f"x or y: {x or y}")
print(f"not x: {not x}")

# =============================================================================
# 4. Control Flow
# =============================================================================
print("\n=== Control Flow ===")

# if-elif-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
print(f"Score {score} -> Grade {grade}")

# Ternary operator
status = "pass" if score >= 60 else "fail"
print(f"Status: {status}")

# =============================================================================
# 5. Loops
# =============================================================================
print("\n=== Loops ===")

# for loop with range
print("For loop with range(5):")
for i in range(5):
    print(f"  i = {i}")

# for loop with step
print("\nFor loop with range(0, 10, 2):")
for i in range(0, 10, 2):
    print(f"  i = {i}")

# while loop
print("\nWhile loop:")
count = 0
while count < 3:
    print(f"  count = {count}")
    count += 1

# break and continue
print("\nBreak and continue:")
for i in range(10):
    if i == 2:
        continue  # Skip 2
    if i == 5:
        break     # Stop at 5
    print(f"  i = {i}")

# =============================================================================
# 6. String Formatting
# =============================================================================
print("\n=== String Formatting ===")

name = "Robot"
x, y, z = 1.234, 5.678, 9.012

# f-strings (Python 3.6+) - recommended
print(f"Position: ({x}, {y}, {z})")
print(f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")  # 2 decimal places

# format method
print("Name: {}, X: {:.1f}".format(name, x))

# %-formatting (old style, not recommended)
print("Name: %s, X: %.2f" % (name, x))

# Padding and alignment
print(f"{'left':<10}|{'center':^10}|{'right':>10}")
print(f"{42:05d}")  # Zero-padded
print(f"{3.14159:.3f}")  # 3 decimal places
print(f"{1000000:,}")  # Thousands separator

# =============================================================================
# 7. Input (commented out for non-interactive execution)
# =============================================================================
print("\n=== User Input ===")
print("# user_input = input('Enter your name: ')")
print("# age = int(input('Enter your age: '))")
print("# (Commented out for non-interactive execution)")


if __name__ == "__main__":
    print("\n=== Script Complete ===")
    print("Run: python3 01_basics.py")
