# MASt3R-SLAM Rerun Visualization

Streams MASt3R-SLAM output to [Rerun](https://rerun.io) in real-time.

## What is logged

| Rerun entity | Type | Description |
|---|---|---|
| `camera` | `Transform3D` | Camera pose (SE3 from SLAM) |
| `camera/image` | `Image` | Current RGB frame (JPEG compressed) |
| `camera/depth` | `DepthImage` | Registered depth (TUM depth scale 1/5000 m) |
| `trajectory` | `LineStrips3D` | Full camera trajectory |
| `map/pointcloud` | `Points3D` | Back-projected dense point cloud |
| `axes` | `Arrows3D` | World coordinate frame axes |

## Dataset

Uses **TUM RGB-D** format:
- `rgb/` with colour frames and `rgb.txt` timestamp index
- `depth/` with depth frames and `depth.txt` timestamp index

Download: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download>

Recommended sequence: `rgbd_dataset_freiburg1_desk`

### Camera intrinsics

TUM fr1 defaults are pre-set (`--fx 525 --fy 525 --cx 319.5 --cy 239.5`).
Override with `--fx`, `--fy`, `--cx`, `--cy` for other sequences or cameras.

## Build

```bash
docker build -t mast3r_slam_rerun .
```

> Same dependency caveats as the profiling container apply (see
> `../../profiling/mast3r_slam/README.md`).

## Run

### Live web viewer (recommended for headless servers)

```bash
docker run --rm --gpus all \
  -p 9090:9090 -p 9876:9876 \
  -v /path/to/tum_sequence:/data/tum:ro \
  mast3r_slam_rerun \
  python3 /rerun_viz/track_rerun.py /data/tum --web --max-frames 500
```

Then open **http://localhost:9090** in a browser.

### Save .rrd for later replay

```bash
docker run --rm --gpus all \
  -v /path/to/tum_sequence:/data/tum:ro \
  -v /path/to/output:/output \
  mast3r_slam_rerun \
  python3 /rerun_viz/track_rerun.py /data/tum \
    --rrd /output/mast3r_slam.rrd --max-frames 500

# Replay on host
rerun /output/mast3r_slam.rrd
```

## Notes

- `rerun-sdk` version is installed as `latest` from PyPI; pin with
  `rerun-sdk==0.x.y` in the Dockerfile if you need reproducible builds.
- The `|| true` guards in `pip install -r requirements.txt` tolerate
  optional C-extension builds (roma, lietorch) that require matching CUDA
  compiler toolchains. Core transformer inference works without them.
- If MASt3R-SLAM's Python API changes after commit `e6f4e3d`, adjust the
  import paths in `track_rerun.py` (`mast3r_slam.slam.SLAM`,
  `mast3r_slam.config.load_config`, `mast3r_slam.dataloader.TUMDataset`).
