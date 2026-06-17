# SymForce: Symbolic Computing for Robotics

Python exercise using [SymForce](https://github.com/symforce-org/symforce) for symbolic math, code generation, and factor-graph optimization in robotics.

---

## Project Structure

```
part3_ch01_16/
├── README.md
├── CMakeLists.txt
├── Dockerfile
└── examples/
    └── symforce_basics.py   # Symbolic variables, geometry types, factor graph optimization, code generation
```

> **Note:** `CMakeLists.txt` conditionally builds `symforce_example` (C++) only when the `symforce` CMake package is found. The primary exercise is the Python script above.

---

## Build

Dependencies:
- **SymForce** (Python) — required. Install via `pip install symforce`.
- **symforce CMake package** — optional. `symforce_example` (C++) is built only when found.

```bash
# Local (Python only — no build step needed)
pip install symforce

# Local (optional C++ build)
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part3_ch01_16
```

---

## Run

### Local

```bash
python3 examples/symforce_basics.py
```

### Docker

```bash
docker run -it --rm slam_zero_to_hero:part3_ch01_16
python3 examples/symforce_basics.py
```

---

## References

- [SymForce GitHub](https://github.com/symforce-org/symforce)
- [SymForce Documentation](https://symforce.org/)
