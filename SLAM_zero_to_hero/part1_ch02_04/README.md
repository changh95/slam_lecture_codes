# C++ CPU Profiler with easy_profiler

This tutorial demonstrates how to profile C++ code using [easy_profiler](https://github.com/yse/easy_profiler), a lightweight cross-platform profiler library for C++.

We'll profile OpenCV feature detection algorithms (FAST, SIFT) across multiple CPU threads - a common scenario in SLAM systems.

## Quick Start (Docker)

```bash
# 1. Build the Docker image
docker build . -t slam_zero_to_hero:part1_ch02_04

# 2. Allow the container to talk to your X server
#    Docker:  xhost +local:docker
#    Podman:  xhost +local:
xhost +local:

# 3. Run profiler and visualize (all-in-one)
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part1_ch02_04

# 4. The GUI opens automatically after the profile run completes.
#    To get a shell instead, override the entrypoint:
#       docker run -it --rm -e DISPLAY=$DISPLAY \
#           -v /tmp/.X11-unix:/tmp/.X11-unix \
#           slam_zero_to_hero:part1_ch02_04 bash
#    then run: ./profile_features && profiler_gui profile.prof
```

> **Note**: Run `xhost -local:` afterwards to revoke display access if desired.
> On Wayland sessions, ensure XWayland is running (it is by default on GNOME/KDE).

## Local Build (without Docker)

```bash
# Dependencies: OpenCV, easy_profiler
mkdir build && cd build
cmake .. && make -j

# Run and visualize
./profile_features
profiler_gui profile.prof
```

---

## Topics Covered

### 1. easy_profiler Basics
- `EASY_PROFILER_ENABLE` - Enable profiling
- `EASY_BLOCK("name")` - Mark a code block for profiling
- `EASY_FUNCTION()` - Automatically profile entire function
- `EASY_THREAD("name")` - Name a thread for visualization
- `profiler::dumpBlocksToFile()` - Save profile data

### 2. Multi-threaded Profiling
- Profile code running on multiple CPU threads
- Visualize thread execution timeline
- Identify bottlenecks and load imbalance

### 3. OpenCV Feature Detection
- FAST detector (fast, many keypoints)
- SIFT detector (slower, more robust)
- Compare performance characteristics

---

## Understanding the Profile Output

The generated `.prof` file can be opened in `profiler_gui`:

```
Thread 0 (Main)
├── ProcessImage [100ms]
│   ├── LoadImage [5ms]
│   └── WaitForWorkers [95ms]

Thread 1 (Worker-0)
├── FAST_Detection [15ms]
└── SIFT_Detection [80ms]

Thread 2 (Worker-1)
├── FAST_Detection [14ms]
└── SIFT_Detection [82ms]

Thread 3 (Worker-2)
├── FAST_Detection [16ms]
└── SIFT_Detection [78ms]
```

---

## Why CPU Profiling Matters for SLAM

- **Identify bottlenecks**: Feature detection, descriptor extraction, matching
- **Optimize thread distribution**: Balance work across CPU cores
- **Real-time requirements**: Ensure processing fits within frame budget
- **Compare algorithms**: FAST vs ORB vs SIFT performance
