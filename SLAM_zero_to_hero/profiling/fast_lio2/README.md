# FAST-LIO2 Profiling with easy_profiler

This directory contains everything needed to build and run FAST-LIO2 with
[easy_profiler](https://github.com/yse/easy_profiler) instrumentation.

## Directory layout

```
fast_lio2/
  patches/
    CMakeLists.txt        - patched build file (adds WITH_PROFILER option)
    laserMapping.cpp      - main node with frame-level profiler blocks
    preprocess.cpp        - LiDAR preprocessing with profiler block
    IMU_Processing.hpp    - IMU processing with profiler blocks
  Dockerfile.profiler     - Docker image that builds the instrumented binary
  README.md               - this file
```

The build context for Docker is the `profiling/` directory (one level up).

## Instrumented blocks

| Block name                      | Color   | Source location              |
|---------------------------------|---------|------------------------------|
| `SLAM/FrameProcess`             | Red     | laserMapping.cpp main loop   |
| `SLAM/IMUProcessing`            | Orange  | laserMapping.cpp + IMU_Processing.hpp |
| `SLAM/FOVSegment`               | Cyan    | laserMapping.cpp             |
| `SLAM/ICP`                      | Blue    | h_share_model()              |
| `SLAM/MapUpdate`                | Green   | map_incremental()            |
| `SLAM/Preprocessing`            | Yellow  | preprocess.cpp               |
| `SLAM/IMUProcessing/Undistort`  | Orange  | ImuProcess::UndistortPcl()   |

## Build

```bash
# From slam_lecture_codes/SLAM_zero_to_hero/profiling/
docker build -f fast_lio2/Dockerfile.profiler -t fast_lio2_profiler .
```

## Run

```bash
docker run --rm \
  -v /your/rosbag/dir:/data \
  -v /your/output/dir:/output \
  fast_lio2_profiler \
  roslaunch fast_lio mapping_avia.launch
```

On `Ctrl-C` (SIGINT) the node dumps `fast_lio2_profiler.prof` to `/output/`.

## View results

Open the `.prof` file with the easy_profiler GUI (`profiler_gui`):

```bash
profiler_gui /your/output/dir/fast_lio2_profiler.prof
```

The `convert_prof.sh` script in the parent `profiling/` directory can convert
the binary profile to a CSV for offline analysis.
