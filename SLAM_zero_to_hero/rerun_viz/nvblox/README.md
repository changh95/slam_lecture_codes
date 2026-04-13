# Rerun Visualization for nvblox

Live (or post-run) visualization of [nvblox](https://github.com/nvidia-isaac/nvblox)
3D reconstruction using the [Rerun](https://rerun.io) viewer.

## Strategy

| Output source       | Approach                                       |
|---------------------|------------------------------------------------|
| nvblox Python API   | Integrate frames live, push incremental mesh   |
| Pre-built .ply file | Load mesh, replay in Rerun (no GPU required)   |

## Rerun entities

| Entity               | Content                                         |
|----------------------|-------------------------------------------------|
| `world/axes`         | XYZ reference frame (static)                   |
| `nvblox/pose`        | Camera/sensor transform per frame               |
| `nvblox/pose/sensor` | Small box at sensor position                   |
| `nvblox/mesh`        | Reconstructed triangle mesh (colour if available) |
| `nvblox/tsdf_voxels` | TSDF voxel centres coloured by weight          |

## Build

```bash
cd rerun_viz/nvblox/
docker build -t slam:nvblox-rerun .
```

## Run - Live Mode

Integrates depth frames one by one and streams the growing mesh to Rerun.

```bash
docker run --rm --gpus all \
  -p 9090:9090 -p 9876:9876 \
  -v ~/data/3dmatch/sun3d-home_at:/data \
  slam:nvblox-rerun \
  python3 nvblox_rerun.py /data --mode live --voxel-size 0.05
```

Open viewer: **http://localhost:9090/?url=rerun+http://localhost:9876/proxy**

## Run - Replay Mode

Display a pre-built .ply mesh (no GPU required after build):

```bash
docker run --rm \
  -p 9090:9090 -p 9876:9876 \
  -v ~/output:/output \
  slam:nvblox-rerun \
  python3 nvblox_rerun.py --mode replay --mesh-ply /output/nvblox_mesh.ply
```

Open viewer: **http://localhost:9090/?url=rerun+http://localhost:9876/proxy**

## Options

| Flag              | Default           | Description                                     |
|-------------------|-------------------|-------------------------------------------------|
| `--mode`          | `live`            | `live` or `replay`                              |
| `--mesh-ply`      | `/output/nvblox_mesh.ply` | PLY file for replay mode               |
| `--voxel-size`    | `0.05`            | TSDF voxel size in metres                       |
| `--mesh-interval` | `10`              | Update mesh every N frames (live)               |
| `--max-frames`    | `0` (all)         | Limit frames processed (0 = all)                |

## Datasets

Same datasets as the profiler:

| Dataset        | Notes                                                          |
|----------------|----------------------------------------------------------------|
| 3DMatch        | `depth/` `color/` `pose/` layout - primary nvblox test data   |
| Replica        | Same layout as 3DMatch                                         |
| TUM RGB-D      | `depth.txt` `rgb.txt` `groundtruth.txt` format                 |

Download 3DMatch from: http://3dmatch.cs.princeton.edu/
