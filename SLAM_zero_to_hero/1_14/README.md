# Basic Python Programming

This tutorial covers fundamental Python programming concepts essential for robotics and SLAM development.

## Topics Covered

| File | Topics |
|------|--------|
| `01_basics.py` | Variables, data types, operators, control flow, loops |
| `02_data_structures.py` | Lists, tuples, dictionaries, sets, comprehensions |
| `03_functions.py` | Functions, lambdas, decorators, generators |
| `04_classes.py` | Classes, inheritance, magic methods, dataclasses |
| `05_file_io.py` | File I/O, JSON, CSV, exceptions, context managers |
| `06_modules_venv.py` | Modules, packages, virtual environments (venv) |

---

## How to Run

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
# .venv\Scripts\activate

# Install dependencies
pip install numpy opencv-python

# Run examples
cd examples
python3 01_basics.py
python3 02_data_structures.py
# ... etc

# Deactivate when done
deactivate
```

### Without Virtual Environment

```bash
cd examples
python3 01_basics.py
```

---

## Key Concepts

### 1. Variables and Types
```python
# Numbers
x = 42          # int
y = 3.14        # float

# Strings
name = "Robot"
msg = f"Name: {name}"  # f-string

# Collections
items = [1, 2, 3]       # list (mutable)
point = (1.0, 2.0)      # tuple (immutable)
config = {"speed": 2.0}  # dict
```

### 2. Control Flow
```python
# Conditional
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# Loop
for i in range(10):
    print(i)

# List comprehension
squares = [x**2 for x in range(10)]
```

### 3. Functions
```python
def process_data(data: list, threshold: float = 0.5) -> list:
    """Process data with threshold filter."""
    return [x for x in data if x > threshold]

# Lambda
square = lambda x: x ** 2
```

### 4. Classes
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

class Robot:
    def __init__(self, name: str):
        self.name = name

    def move(self, dx: float, dy: float):
        # ...
```

### 5. File I/O
```python
# JSON
import json

with open("config.json", "r") as f:
    config = json.load(f)

# CSV
import csv

with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
```

### 6. Virtual Environment (venv)
```bash
# Create
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install packages
pip install numpy scipy opencv-python

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

---

## Common Packages for Robotics

| Package | Description |
|---------|-------------|
| `numpy` | Numerical computing |
| `scipy` | Scientific computing |
| `opencv-python` | Computer vision |
| `matplotlib` | Plotting |
| `rerun-sdk` | 3D visualization |
| `transforms3d` | 3D transformations |

---

## Best Practices

1. **Use type hints** for better code documentation
   ```python
   def add(a: float, b: float) -> float:
       return a + b
   ```

2. **Use virtual environments** for every project
   - Never install packages globally
   - Keep dependencies isolated

3. **Never use conda** (per course guidelines)
   - Use `venv` + `pip` instead

4. **Use f-strings** for formatting
   ```python
   print(f"Position: ({x:.2f}, {y:.2f})")
   ```

5. **Use context managers** for resources
   ```python
   with open("file.txt") as f:
       data = f.read()
   ```
