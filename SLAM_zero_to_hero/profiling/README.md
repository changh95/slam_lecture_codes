# SLAM Profiling

Performance profiling of SLAM algorithms using [easy_profiler](https://github.com/yse/easy_profiler) across multiple hardware platforms.

Algorithms instrumented: **ORB-SLAM2**, **FAST-LIO2**, **GLIM**

---

## Directory Structure

```
profiling/
├── README.md                  # This file
├── requirements.txt           # Python dependencies for analysis script
├── analyze_profiler.py        # Python analysis and comparison tool
├── convert_prof.sh            # Shell helper to convert .prof -> JSON
├── common/
│   └── install_easy_profiler.sh   # Build & install easy_profiler from source
├── orb_slam2/
│   ├── Dockerfile.profiler    # ORB-SLAM2 with easy_profiler instrumentation
│   ├── README.md
│   └── patches/               # Instrumented source files
├── fast_lio2/
│   ├── Dockerfile.profiler    # FAST-LIO2 with easy_profiler instrumentation
│   ├── README.md
│   └── patches/               # Instrumented source files
└── glim/
    ├── Dockerfile.profiler    # GLIM with easy_profiler instrumentation
    ├── README.md
    └── patches/               # Instrumented source files
```

---

## Platform Support

| Platform          | Architecture | Notes                          |
|-------------------|--------------|--------------------------------|
| Desktop (x86_64)  | x86_64       | Reference baseline             |
| Jetson Thor       | arm64        | NVIDIA GB10 SoC                |
| DGX SPARK         | arm64        | NVIDIA GB10 SoC                |
| Jetson Orin       | arm64        | NVIDIA Orin module             |
| Jetson Orin Nano  | arm64        | NVIDIA Orin Nano module        |
| Raspberry Pi 5    | arm64        | Broadcom BCM2712               |

---

## How to Build

### Prerequisites

- Docker (or Podman) installed and running
- For Jetson platforms: JetPack 5.x or 6.x with Docker support

### ORB-SLAM2

```bash
# Build context must be the profiling/ directory
cd profiling/

# x86_64 desktop
docker build -f orb_slam2/Dockerfile.profiler -t slam:orbslam2-profiler .

# arm64 (Jetson, cross-build requires qemu-user-static)
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE=arm64v8/ubuntu:jammy \
  -f orb_slam2/Dockerfile.profiler -t slam:orbslam2-profiler-arm64 .

# Raspberry Pi (same as arm64)
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE=arm64v8/ubuntu:jammy \
  -f orb_slam2/Dockerfile.profiler -t slam:orbslam2-profiler-rpi .
```

### FAST-LIO2

```bash
cd profiling/

# x86_64 desktop
docker build -f fast_lio2/Dockerfile.profiler -t slam:fastlio2-profiler .

# arm64 (Jetson)
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE=arm64v8/ros:noetic \
  -f fast_lio2/Dockerfile.profiler -t slam:fastlio2-profiler-arm64 .
```

### GLIM

```bash
cd profiling/

# x86_64 desktop (with CUDA)
docker build -f glim/Dockerfile.profiler -t slam:glim-profiler .

# DGX SPARK
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04 \
  -f glim/Dockerfile.profiler -t slam:glim-profiler-dgx .

# Raspberry Pi (CPU only)
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE=arm64v8/ubuntu:jammy \
  -f glim/Dockerfile.profiler -t slam:glim-profiler-rpi .
```

---

## How to Run and Collect Profiler Data

### 1. Install easy_profiler (inside Docker or on bare metal)

```bash
bash common/install_easy_profiler.sh
```

This builds and installs `libeasy_profiler` and `easy_profiler_converter` from source. The converter is needed to turn binary `.prof` files into JSON for analysis.

### 2. Run the instrumented algorithm

Each Docker image is configured to write a `output.prof` binary when the algorithm finishes. Mount a host directory to retrieve the file:

```bash
# Example: ORB-SLAM2 with TUM dataset
docker run --rm \
  -v /path/to/dataset:/dataset \
  -v $(pwd)/results:/results \
  orb_slam2_profiling:desktop \
  /slam/run.sh /dataset /results/orb_slam2_desktop.prof

# Example: FAST-LIO2 with custom bag
docker run --rm \
  -v /path/to/bag:/bag \
  -v $(pwd)/results:/results \
  fast_lio2_profiling:desktop \
  /slam/run.sh /bag/scan.bag /results/fast_lio2_desktop.prof
```

### 3. Convert .prof to JSON

```bash
# Using the helper script
./convert_prof.sh results/fast_lio2_desktop.prof

# Or directly
easy_profiler_converter results/fast_lio2_desktop.prof results/fast_lio2_desktop.json
```

The JSON format expected by `analyze_profiler.py`:

```json
{
  "version": "...",
  "timeUnits": "ns",
  "blockDescriptors": [],
  "threads": [
    {
      "threadId": 123,
      "threadName": "MainThread",
      "children": [
        {
          "id": 0,
          "name": "SLAM/FrameProcess",
          "start": 1000000,
          "stop": 1005000,
          "descriptor": 0,
          "children": [
            {
              "name": "SLAM/FeatureExtraction",
              "start": 1000100,
              "stop": 1002000,
              "children": []
            }
          ]
        }
      ]
    }
  ]
}
```

All profiled blocks must have names starting with `SLAM/` to be picked up by the analyzer.

---

## Analyzing Results with analyze_profiler.py

### Setup

```bash
pip install -r requirements.txt
```

### Single file summary

```bash
python3 analyze_profiler.py results/fast_lio2_desktop.json
```

Prints a rich table with: Count, Mean, Median, Min, Max, Std, P95, P99, Total (all in ms), plus key insights (slowest block, FPS estimate).

### Cross-platform comparison

```bash
python3 analyze_profiler.py --compare \
  results/fast_lio2_desktop.json \
  results/fast_lio2_jetson_orin.json \
  results/fast_lio2_rpi.json
```

The first file is the reference. Speedup ratios are shown relative to it (> 1.0 means faster than reference).

### Auto-conversion from .prof

If you pass a `.prof` file directly, the script will attempt to call `easy_profiler_converter` automatically:

```bash
python3 analyze_profiler.py results/fast_lio2_desktop.prof
```

### Export to CSV or JSON

```bash
# CSV
python3 analyze_profiler.py --export csv --output results.csv \
  results/fast_lio2_desktop.json results/fast_lio2_jetson_orin.json

# JSON
python3 analyze_profiler.py --export json --output results.json \
  results/*.json
```

### Generate comparison chart

```bash
python3 analyze_profiler.py --plot comparison.png \
  results/fast_lio2_desktop.json \
  results/fast_lio2_jetson_orin.json \
  results/fast_lio2_rpi.json
```

Saves a bar chart (PNG/PDF) comparing mean execution times per block across all platforms.

---

## Instrumentation Convention

Profiler blocks use a flat two-level naming convention shared across all algorithms:

```
SLAM/<component>
```

This allows the analysis script to compare the same component across different SLAM systems.

| Block Name                        | Used In          | Description                        |
|-----------------------------------|------------------|------------------------------------|
| `SLAM/FrameProcess`               | All              | Full frame processing pipeline     |
| `SLAM/FeatureExtraction`          | ORB-SLAM2        | ORB feature extraction             |
| `SLAM/FeatureMatching`            | ORB-SLAM2        | Feature matching (BoW, projection) |
| `SLAM/Preprocessing`              | FAST-LIO2, GLIM  | Point cloud preprocessing          |
| `SLAM/ICP`                        | FAST-LIO2        | ICP / scan matching                |
| `SLAM/IMUProcessing`              | FAST-LIO2        | IMU processing + undistortion      |
| `SLAM/LocalMapping`               | ORB-SLAM2, GLIM  | Local mapping / sub-mapping        |
| `SLAM/GlobalMapping`              | GLIM             | Global mapping + optimization      |
| `SLAM/LoopClosureDetection`       | ORB-SLAM2, GLIM  | Loop closure detection             |
| `SLAM/LoopClosureCorrection`      | ORB-SLAM2, GLIM  | Loop closure correction            |
| `SLAM/GlobalBA`                   | ORB-SLAM2        | Global bundle adjustment           |
| `SLAM/MapUpdate`                  | FAST-LIO2        | Map update (ikd-tree)              |
| `SLAM/TrackLocalMap`              | ORB-SLAM2        | Track local map                    |

In C++ source, blocks are added with the easy_profiler macros:

```cpp
#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#else
#define EASY_BLOCK(...)
#define EASY_END_BLOCK
#define EASY_FUNCTION(...)
#define EASY_PROFILER_ENABLE
#endif

void processFrame() {
    EASY_BLOCK("SLAM/FrameProcess", profiler::colors::Red);
    extractFeatures();  // contains its own EASY_BLOCK("SLAM/FeatureExtraction")
    matchFeatures();    // contains its own EASY_BLOCK("SLAM/FeatureMatching")
}

// At the end of the run:
profiler::dumpBlocksToFile("/output/profiler_result.prof");
```

---

## File Naming Convention

Output files should follow the pattern `<algorithm>_<platform>.prof` / `.json` for automatic platform and algorithm detection by `analyze_profiler.py`:

```
fast_lio2_desktop.json
fast_lio2_jetson_orin.json
fast_lio2_jetson_orin_nano.json
fast_lio2_jetson_thor.json
fast_lio2_dgx_spark.json
fast_lio2_rpi.json
orb_slam2_desktop.json
glim_desktop.json
```
