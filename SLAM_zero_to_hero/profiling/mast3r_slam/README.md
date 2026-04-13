# MASt3R-SLAM Profiling

Instruments MASt3R-SLAM pipeline stages using `py_profiler.py` and emits
JSON compatible with `../analyze_profiler.py`.

## Profiled blocks

| Block | Description |
|---|---|
| `SLAM/Init` | Model loading and dataset setup |
| `SLAM/FrameProcess` | Full per-frame pipeline |
| `SLAM/ImageLoad` | Image I/O from disk |
| `SLAM/FeatureMatching` | MASt3R transformer inference (dense matching) |
| `SLAM/PoseEstimation` | SE3 pose extraction from match result |
| `SLAM/MapUpdate` | 3D map / keyframe insertion |
| `SLAM/LoopClosure` | Loop-closure check (if supported by installed version) |

## Dataset

Uses **TUM RGB-D** format (recommended for MASt3R-SLAM):
- `rgb/` directory with colour frames
- `rgb.txt` with `timestamp relative/path` lines

Download: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download>

Sequence suggestion: `rgbd_dataset_freiburg1_desk` (handheld, indoor, ~600 frames).

Other supported datasets: EuRoC (MAV), 7-Scenes, ETH3D SLAM — pass the
appropriate dataset class if `mast3r_slam.dataloader` exposes loaders for them.

## Build

```bash
docker build -f Dockerfile.profiler -t mast3r_slam_profiler .
```

> **Note:** MASt3R-SLAM requires `pip install -r requirements.txt` which
> pulls dust3r/mast3r sub-modules. The `|| true` guards in the Dockerfile
> tolerate optional C-extension failures (roma, lietorch) that some CUDA
> versions cannot compile. Core Python inference still works.

## Run

```bash
docker run --rm --gpus all \
  -v /path/to/tum_sequence:/data/tum:ro \
  -v /path/to/output:/output \
  mast3r_slam_profiler \
  python3 /profiling/run_profiled.py /data/tum /output/mast3r_slam.json --max-frames 300
```

## Analyse

```bash
python3 ../analyze_profiler.py /output/mast3r_slam.json
```
