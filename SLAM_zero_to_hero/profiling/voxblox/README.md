# Voxblox – easy_profiler Instrumentation

Adds [easy_profiler](https://github.com/yse/easy_profiler) `EASY_BLOCK`
instrumentation to the major processing stages of
[voxblox](https://github.com/ethz-asl/voxblox) (ROS Noetic / catkin), so
you can get an ORB-SLAM2-style per-component breakdown of where CPU time
goes in the TSDF pipeline.

## Instrumented blocks

| Block name | Source location | What it captures |
|---|---|---|
| `SLAM/FrameProcess` | `voxblox_ros/src/tsdf_server.cc` – `TsdfServer::insertPointcloud` | Full per-frame server work (TF lookup, point cloud transform, integrator dispatch) |
| `SLAM/TsdfIntegration` | `voxblox/src/integrator/tsdf_integrator.cc` – all three `integratePointCloud` overloads | Top-level TSDF integration window (thread spawn + join + store-block flush) |
| `SLAM/TsdfIntegration/Worker` | `Simple/Fast TsdfIntegrator::integrateFunction` | Per-worker-thread integration work (one per `integrator_threads`) |
| `SLAM/MeshGeneration` | `voxblox/include/voxblox/mesh/mesh_integrator.h` – `MeshIntegrator::generateMesh` | Top-level mesh extraction (thread spawn + join + post-process) |
| `SLAM/MeshGeneration/Worker` | `MeshIntegrator::generateMeshBlocksFunction` | Per-worker-thread block-by-block mesh extraction |
| `SLAM/MeshPublish` | `voxblox_ros/src/tsdf_server.cc` – `TsdfServer::generateMesh` | ROS-level mesh publication (also fires on service calls). Only present if a mesh subscriber exists. |
| `SLAM/EsdfIntegration` | `voxblox/src/integrator/esdf_integrator.cc` – `EsdfIntegrator::updateFromTsdfLayer*` | ESDF distance propagation. Only present if an ESDF server is running (the cow_and_lady launch uses TSDF only). |

Worker-thread blocks are captured thanks to `EASY_THREAD("…")` registration
at the top of each worker entry point. Without that, easy_profiler silently
drops blocks logged from threads it has never seen.

## How it works

Two shell scripts run at image-build time, before `catkin build`:

- **`cmake_patch.sh`** adds an `option(WITH_PROFILER ...)` block to *both*
  `voxblox/CMakeLists.txt` and `voxblox_ros/CMakeLists.txt`, guards
  `add_definitions(-DBUILD_WITH_EASY_PROFILER)` on it, and links the
  `easy_profiler` library into the `voxblox`, `voxblox_ros`, and
  `voxblox_node` targets. The old version only patched `voxblox_ros`,
  which silently disabled every instrumented block in the inner library.

- **`profiler_patch.sh`** uses Python+`re` (not `sed`/`awk`, because voxblox
  function signatures span multiple lines) to insert
  `EASY_BLOCK("SLAM/…")` / `EASY_THREAD("…")` macros at the start of each
  target function, all guarded by `#ifdef BUILD_WITH_EASY_PROFILER`.

`tsdf_server_node.cc` is additionally patched to call
`EASY_PROFILER_ENABLE` right after `ros::init`, and to install an
`atexit()` handler that calls `profiler::dumpBlocksToFile("/output/voxblox_profiler.prof")`
before shutdown.

## Build (Docker)

```bash
# From SLAM_zero_to_hero/profiling/
docker build -f voxblox/Dockerfile.profiler -t slam:voxblox-profiler .
```

## Run

The ROS1 `rosbag play` C++ binary has a performance pathology on
aarch64 that stalls the cow_and_lady bag at ~0.2 Hz (burns 100% CPU,
publishes almost nothing). `profile_run.sh` works around it by pushing
the bag through voxblox with our own Python player
(`rerun_viz/voxblox/bag_player.py`, which just iterates messages via the
`rosbag` library and republishes them via `rospy`).

`profile_run.sh` and `bag_player.py` are baked into the image as `/app/`,
so the run is a single command:

```bash
docker run --rm \
  -v /path/to/cow_and_lady_dataset.bag:/data/input.bag:ro \
  -v /tmp/voxblox_profile_out:/output \
  slam:voxblox-profiler
```

Environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `BAG_PATH` | `/data/input.bag` | Bag file path inside the container |
| `LAUNCH_FILE` | `cow_and_lady_dataset.launch` | Voxblox launch file |
| `PLAYBACK_RATE` | `2.0` | Playback speed for bag_player.py (2× is fine for profiling) |

The run script writes `voxblox_profiler.prof` to `/output/`. Convert to
JSON for downstream tooling (same command that ships with easy_profiler):

```bash
docker run --rm -v /tmp/voxblox_profile_out:/out --entrypoint bash \
  slam:voxblox-profiler -c \
  'profiler_converter /out/voxblox_profiler.prof /out/voxblox.json'
```

## Sample results (DGX Spark, cow_and_lady @ voxel_size=0.05)

See `SLAM_zero_to_hero/perf_bench/dgx_spark/voxblox.{prof,json}`.
Summary for one full bag (142 s, ~2830 frames, 20 integrator threads,
stock Release build — no LTO/`-march=native`):

| Block | count | total (s) | avg |
|---|---|---|---|
| `SLAM/FrameProcess` | 2,829 | 173.7 | 61.4 ms |
| `SLAM/TsdfIntegration` | 2,713 | 165.7 | 61.1 ms |
| `SLAM/TsdfIntegration/Worker` | 54,260 | 2,747.8 | 50.6 ms |
| `SLAM/MeshGeneration` | 411 | 0.7 | 1.75 ms |
| `SLAM/MeshGeneration/Worker` | 8,220 | 5.6 | 0.68 ms |

FrameProcess time is dominated by TsdfIntegration (~96%), as expected
for voxblox without carving or ESDF — most cycles go into ray casting
and voxel hashing in the worker threads, while mesh extraction is
negligible (<0.5% of total).

## Datasets

| Dataset | Download | Topics |
|---|---|---|
| cow_and_lady | `wget http://rpg.ifi.uzh.ch/docs/IJRR17_Loquercio/datasets/cow_and_lady_dataset.bag` | `/camera/depth_registered/points`, `/kinect/vrpn_client/estimated_transform` |

cow_and_lady is the canonical voxblox benchmark and is what the stock
launch file expects.
