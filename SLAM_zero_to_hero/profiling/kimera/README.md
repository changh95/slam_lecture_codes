# Kimera-VIO Profiling with easy_profiler

Performance instrumentation for [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO)
using [easy_profiler](https://github.com/yse/easy_profiler).

## What is Profiled

| Block name | Source file | What it measures |
|---|---|---|
| `SLAM/FrameProcess` | `pipeline/StereoImuPipeline.cpp` | Full per-frame processing time |
| `SLAM/FeatureTracking` | `frontend/Tracker.cpp` | Lucas-Kanade optical flow tracking |
| `SLAM/FeatureExtraction` | `frontend/Tracker.cpp` | FAST corner detection |
| `SLAM/RANSAC` | `frontend/Tracker.cpp` | 5-point RANSAC outlier rejection |
| `SLAM/StereoMatching` | `frontend/StereoMatcher.cpp` | Sparse stereo depth reconstruction |
| `SLAM/IMUIntegration` | `backend/VioBackend.cpp` | IMU preintegration (GTSAM) |
| `SLAM/VIOOptimization` | `backend/VioBackend.cpp` | iSAM2 / Levenberg-Marquardt factor graph optimization |

## Build

```bash
cd profiling/
docker build -t slam:kimera-profiler \
    -f kimera/Dockerfile.profiler .
```

## Run

Mount an EuRoC MAV sequence (e.g. MH_01_easy) at `/data`:

```bash
docker run --rm \
    -v /path/to/euroc/MH_01_easy/mav0:/data \
    -v $(pwd)/results/kimera:/output \
    slam:kimera-profiler
```

The `.prof` file is written to `/output/kimera_profile.prof` on the host.

## Dataset

**EuRoC MAV** — stereo + IMU sequences.

Download:
```bash
# From the slam_lecture_codes project root:
python3 download_euroc.py   # if available, or manually from:
# https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf (data links in paper)
# Or:
wget http://rpg.ifi.uzh.ch/docs/IJRR17_Burri_EuRoC/MH_01_easy.zip
```

Expected structure:
```
/data/
  cam0/data/          # left stereo images (*.png)
  cam1/data/          # right stereo images (*.png)
  imu0/data.csv       # IMU measurements
  state_groundtruth_estimate0/data.csv
```

## Viewing the Profile

```bash
# Convert .prof -> .json for web viewing
easy_profiler_converter /output/kimera_profile.prof /output/kimera_profile.json

# Or open with easy_profiler GUI
profiler_gui /output/kimera_profile.prof
```

## Patch Details

The `profiler_patch.sh` script:
1. Inserts the `#ifdef BUILD_WITH_EASY_PROFILER` guard into each source file
2. Inserts `EASY_BLOCK(...)` at the entry of each major function
3. Patches `CMakeLists.txt` to add `option(WITH_PROFILER)` and conditional `easy_profiler` linkage
4. Patches `examples/KimeraVIO.cpp` to call `EASY_PROFILER_ENABLE` and `profiler::dumpBlocksToFile`

The `src/` directory contains annotated patch documentation files showing exactly which
functions are instrumented and what the guard pattern looks like.
