# C++ CPU Profiler with easy_profiler

This tutorial demonstrates how to profile C++ code using [easy_profiler](https://github.com/yse/easy_profiler), a lightweight cross-platform profiler library for C++.

We'll profile OpenCV feature detection algorithms (FAST, SIFT) across multiple CPU threads - a common scenario in SLAM systems.

## How to build

Dependencies: OpenCV, easy_profiler

Local build:
```bash
mkdir build
cd build
cmake ..
make -j
```

Docker build:
```bash
docker build . -t slam_zero_to_hero:1_11
```

## How to run

Local:
```bash
# Run the profiler example (generates profile.prof)
./build/profile_features

# View the profile (requires easy_profiler GUI on host)
profiler_gui profile.prof
```

Docker:
```bash
# Run and copy profile output
docker run -it --rm -v $(pwd)/output:/output slam_zero_to_hero:1_11

# Then open profile.prof with easy_profiler GUI on your host machine
```

## Installing easy_profiler GUI (on host machine)

```bash
# Ubuntu
sudo apt install easy-profiler

# Or build from source with GUI
git clone https://github.com/yse/easy_profiler.git
cd easy_profiler && mkdir build && cd build
cmake .. -DEASY_PROFILER_NO_GUI=OFF
make -j && sudo make install
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
