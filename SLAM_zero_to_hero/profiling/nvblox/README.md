# nvblox Profiler

Profiles the nvblox 3D reconstruction pipeline using the Python `py_profiler` wrapper
(same schema as easy_profiler JSON output).

nvblox is open-source (Apache 2.0): https://github.com/nvidia-isaac/nvblox

## Architecture

nvblox provides both a C++ API and Python bindings (`nvblox_python`). The profiler
wraps the Python bindings to time each pipeline stage:

| Profiler block            | nvblox operation                          |
|---------------------------|-------------------------------------------|
| `nvblox/Init`             | Mapper construction, layer allocation     |
| `nvblox/FrameProcess`     | Full per-frame pipeline                   |
| `nvblox/LoadDepth`        | Read depth PNG from disk                  |
| `nvblox/Integrate`        | TSDF integration (GPU kernel)             |
| `nvblox/IntegrateColor`   | Color integration (GPU kernel)            |
| `nvblox/UpdateMesh`       | Incremental mesh extraction (GPU)         |
| `nvblox/ExtractMesh`      | Final mesh query                          |

If Python bindings are not available (architecture mismatch), falls back to timing
the `fuse_3dmatch` C++ binary as a single `nvblox/FullRun` block.

## Supported Datasets

| Dataset        | Format                              | Download                                              |
|----------------|-------------------------------------|-------------------------------------------------------|
| 3DMatch        | `depth/` `color/` `pose/` dirs      | http://3dmatch.cs.princeton.edu/                      |
| Replica        | `depth/` `color/` `pose/` dirs      | https://github.com/facebookresearch/Replica-Dataset   |
| TUM RGB-D      | `depth.txt` `rgb.txt` `groundtruth.txt` | https://cvg.cit.tum.de/data/datasets/rgbd-dataset  |

nvblox's own bundled examples use 3DMatch scenes. The recommended test scene is:

```
sun3d-home_at-home_at_scan1_2013_jan_1
```

## Build

```bash
cd profiling/nvblox/
docker build -f Dockerfile.profiler -t slam:nvblox-profiler .
```

Build time: ~15-25 minutes (compiles nvblox + CUDA kernels from source).

## Run

```bash
docker run --rm --gpus all \
  -v ~/data/3dmatch/sun3d-home_at:/data \
  -v ~/profiling_output:/output \
  slam:nvblox-profiler \
  python3 track_nvblox_profiled.py /data /output/nvblox_profiler.json
```

With TUM RGB-D dataset:

```bash
docker run --rm --gpus all \
  -v ~/data/rgbd_dataset_freiburg1_desk:/data \
  -v ~/profiling_output:/output \
  slam:nvblox-profiler \
  python3 track_nvblox_profiled.py /data /output/nvblox_profiler.json --voxel-size 0.02
```

Limit frames for a quick test:

```bash
python3 track_nvblox_profiled.py /data /output/nvblox_profiler.json --max-frames 200
```

## Analyze

Use the shared `analyze_profiler.py` in the `profiling/` root:

```bash
python3 ../analyze_profiler.py /output/nvblox_profiler.json
```
