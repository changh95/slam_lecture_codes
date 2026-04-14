# nvblox Profiler

Profiles the nvblox 3D reconstruction pipeline on a Replica scene by running
`fuse_replica --timing_output_path` and converting nvblox's built-in
`nvblox::timing::Timer` report into the shared `analyze_profiler` CSV schema.

nvblox is open-source (Apache 2.0): https://github.com/nvidia-isaac/nvblox

## Why this approach

Earlier iterations of this profiler used nvblox's Python bindings (wrapped in
`py_profiler.py`) to time each mapper call. That path is dead on aarch64 — the
`nvblox` wheel doesn't publish for ARM and building bindings from source is
fragile. nvblox already has a rich per-call `Timer` that tracks count, total,
running mean, stddev, min, and max for every stage of the fuser, TSDF, ESDF,
mesh, color, view-calculator, and GPU-hash pipelines — so we just dump it.

## Captured blocks (53 total)

fuse_replica instruments the pipeline with nvblox's own timers. After parsing
they are prefixed with `SLAM/` to match the analyze_profiler CSV convention.
Key top-level entries:

| Block | What it measures |
|---|---|
| `SLAM/fuser/time_per_frame` | Wall clock of one main-loop iteration |
| `SLAM/fuser/file_loading` | PNG + pose read for one frame |
| `SLAM/fuser/integrate_depth` | Full depth → TSDF integration |
| `SLAM/fuser/integrate_color` | Full color integration |
| `SLAM/fuser/integrate_esdf` | ESDF propagation from TSDF |
| `SLAM/fuser/mesh` | Incremental mesh update |
| `SLAM/tsdf/integrate` | GPU TSDF kernel dispatch (core of integrate_depth) |
| `SLAM/color/integrate` | GPU color kernel dispatch |
| `SLAM/esdf/integrate` | GPU ESDF kernel dispatch |
| `SLAM/mesh/gpu/mesh_blocks` | GPU marching-cubes mesh-blocks kernel |
| `SLAM/view_calculator/raycast` | Frustum raycast for visible-block set |
| `SLAM/gpu_hash/transfer/reset` | GPU hash table reset between frames |

Plus ~40 sub-blocks (allocate_blocks, sphere_trace, kernel_table,
neighbor_bands, pack_out, etc.) — see `/output/nvblox_stats.csv`.

## Build

```bash
cd profiling/
docker build -f nvblox/Dockerfile.profiler -t slam:nvblox-profiler .
```

Build time: ~15-25 minutes (compiles nvblox + CUDA kernels from source).

## Run

Mount a Replica dataset root (the one that contains `Replica/cam_params.json`
and `Replica/<scene>/`) at `/data`:

```bash
docker run --rm --gpus all \
  -v ~/data/replica:/data \
  -v ~/profiling_output:/output \
  slam:nvblox-profiler
```

The default entrypoint is `run_profile.sh`, which runs
`fuse_replica /data/Replica/office0 --num_frames=500 --timing_output_path=...`
and then parses the timing report into `nvblox_stats.csv`.

Environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `DATASET_PATH` | `/data/Replica/office0` | Scene root |
| `NUM_FRAMES` | `500` | Negative = process all frames |
| `OUTPUT_TIMING` | `/output/nvblox_timing.txt` | Raw nvblox timing dump |
| `OUTPUT_MESH` | `/output/nvblox_mesh.ply` | Final mesh output |
| `OUTPUT_CSV` | `/output/nvblox_stats.csv` | Parsed stats CSV |

Full-run example against a different scene:

```bash
docker run --rm --gpus all \
  -e DATASET_PATH=/data/Replica/room0 \
  -e NUM_FRAMES=-1 \
  -v ~/data/replica:/data \
  -v ~/profiling_output:/output \
  slam:nvblox-profiler
```

## Analyze

The exported CSV already matches `analyze_profiler.py`'s `--export csv` schema,
so it can be concatenated with the other SLAM systems' stats for cross-system
comparison. Note that nvblox's Timer does not track median/p95/p99, so those
columns are left blank.

```bash
# Cross-system comparison against ORB-SLAM2:
python3 ../analyze_profiler.py --compare \
  results/orb_slam2_desktop.json results/nvblox_desktop.json
```

## Supported datasets

The Dockerfile entrypoint targets Replica, which is the canonical public
dataset for the `fuse_replica` binary. nvblox also ships `fuse_3dmatch` and
`fuse_redwood`; swap the binary in `run_profile.sh` if you need those
pipelines instead.
