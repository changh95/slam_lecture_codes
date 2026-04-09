# ORB-SLAM2 Profiling with easy_profiler

This directory contains easy_profiler-instrumented patches for ORB-SLAM2 and a Dockerfile to build and run the profiled binary.

## What is instrumented

| File | Blocks added |
|---|---|
| `System.cc` | `EASY_PROFILER_ENABLE` in constructor; dump to `/output/orb_slam2_profiler.prof` in `Shutdown()` |
| `Tracking.cc` | `GrabImageStereo`, `GrabImageRGBD`, `GrabImageMonocular` (FrameProcess); `Track`; `TrackLocalMap` |
| `ORBextractor.cc` | `operator()` (FeatureExtraction); `ComputeKeyPointsOctTree` |
| `ORBmatcher.cc` | `SearchByBoW`, `SearchByProjection`, `SearchForTriangulation` |
| `LocalMapping.cc` | Main processing block inside `Run()` |
| `LoopClosing.cc` | `DetectLoop` block, `CorrectLoop`, `RunGlobalBundleAdjustment` |
| `Optimizer.cc` | `BundleAdjustment` (GlobalBA), `LocalBundleAdjustment` |

## Build commands

### Docker (recommended)

Build context is the `profiling/` directory:

```bash
cd /path/to/SLAM_zero_to_hero/profiling

docker build \
    -f orb_slam2/Dockerfile.profiler \
    -t slam:orbslam2-profiler \
    .
```

Run with a dataset mounted and collect the profile output:

```bash
docker run --rm \
    -v /path/to/dataset:/dataset:ro \
    -v /path/to/output:/output \
    slam:orbslam2-profiler \
    ./Examples/Monocular/mono_tum \
        Vocabulary/ORBvoc.txt \
        Examples/Monocular/TUM1.yaml \
        /dataset/rgbd_dataset_freiburg1_xyz
```

The profiler dump is written to `/output/orb_slam2_profiler.prof` inside the container (bind-mounted to the host `/path/to/output/`).

### Native build (without Docker)

Requires easy_profiler installed system-wide or pointed to via `CMAKE_PREFIX_PATH`:

```bash
cd /path/to/Portable_ORB_SLAM2
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_PROFILER=ON
make -j$(nproc)
```

To build **without** profiling (default):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_PROFILER=OFF
make -j$(nproc)
```

## Viewing the profile

Use the easy_profiler GUI (`profiler_gui`) to open `orb_slam2_profiler.prof`:

```bash
profiler_gui /output/orb_slam2_profiler.prof
```

Or convert to CSV using the `convert_prof.sh` script at the top of the `profiling/` directory.

## Cross-compilation note

The patched `CMakeLists.txt` replaces `-march=native` with a conditional that omits the flag when `CMAKE_CROSSCOMPILING` is set, enabling ARM/embedded targets.
