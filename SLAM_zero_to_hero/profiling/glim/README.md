# GLIM v1.0.0 easy_profiler Instrumentation

This directory contains easy_profiler instrumentation patches for [GLIM](https://github.com/koide3/glim) v1.0.0, a modular 3D LiDAR-based SLAM library.

## What is instrumented

| File | Method | Block label | Color |
|------|--------|-------------|-------|
| `cloud_preprocessor.cpp` | `CloudPreprocessor::preprocess()` | `SLAM/Preprocessing` | Yellow |
| `odometry_estimation_imu.cpp` | `OdometryEstimationIMU::insert_frame()` | `SLAM/FrameProcess` | Red |
| `sub_mapping.cpp` | `SubMapping::insert_frame()` | `SLAM/LocalMapping` | Yellow |
| `global_mapping.cpp` | `GlobalMapping::insert_submap()` | `SLAM/GlobalMapping` | Purple |
| `global_mapping.cpp` | `GlobalMapping::optimize()` | `SLAM/GlobalMapping/Optimize` | Red200 |
| `global_mapping_pose_graph.cpp` | `GlobalMappingPoseGraph::optimize()` | `SLAM/LoopClosureCorrection` | Magenta |
| `global_mapping_pose_graph.cpp` | `GlobalMappingPoseGraph::find_loop_candidates()` | `SLAM/LoopClosureDetection` | Magenta |

## Files

```
glim/
  patches/
    CMakeLists.txt                  # Patched build file with easy_profiler option
    cloud_preprocessor.cpp          # Preprocessing instrumentation
    odometry_estimation_imu.cpp     # IMU odometry instrumentation
    sub_mapping.cpp                 # Local submap instrumentation
    global_mapping.cpp              # Global mapping instrumentation
    global_mapping_pose_graph.cpp   # Pose graph / loop closure instrumentation
    offline_viewer.cpp              # Profiler init/dump helper (library wrapper)
  Dockerfile.profiler               # Docker build with all dependencies + patches applied
  README.md                         # This file
```

## Build

The Docker build context is the `profiling/` directory (one level above this folder).

```bash
cd /path/to/SLAM_zero_to_hero/profiling
docker build -f glim/Dockerfile.profiler -t glim-profiler .
```

## Run

```bash
docker run --rm -v /path/to/output:/output glim-profiler <your-glim-command>
```

On SIGINT the profiler dumps `glim_profiler.prof` into `/output/`.

## View results

Use the easy_profiler GUI (`profiler_gui`) to open the `.prof` file, or use the `convert_prof.sh` script in the parent `profiling/` directory.

## How the include guard works

Each patched `.cpp` file contains the following block after the existing includes:

```cpp
#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#else
#define EASY_FUNCTION(...)
#define EASY_END_BLOCK
#define EASY_PROFILER_ENABLE
#endif
```

When `BUILD_WITH_EASY_PROFILER=OFF` (the default), all profiler macros expand to nothing, so the patched files compile identically to the originals with zero overhead.

## CMake option

```bash
cmake .. -DBUILD_WITH_EASY_PROFILER=ON
```

This sets `-DBUILD_WITH_EASY_PROFILER` and links `easy_profiler` into the `glim` shared library.
