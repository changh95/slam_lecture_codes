# Voxblox – easy_profiler Instrumentation

Adds [easy_profiler](https://github.com/yse/easy_profiler) `EASY_BLOCK`
instrumentation to the major processing stages of
[voxblox](https://github.com/ethz-asl/voxblox) (ROS Noetic / catkin).

## Instrumented blocks

| Block name | Source location | Colour |
|---|---|---|
| `SLAM/FrameProcess` | `voxblox_ros/src/tsdf_server.cc` – `insertPointcloudWithTf()` | Orange |
| `SLAM/Preprocessing` | `voxblox/src/integrator/tsdf_integrator.cc` – `integrateFunction()` | Dark Green |
| `SLAM/TsdfIntegration` | `tsdf_integrator.cc` – all three `integratePointCloud()` overloads | Red |
| `SLAM/EsdfIntegration` | `voxblox/src/integrator/esdf_integrator.cc` + `esdf_server.cc` | Blue |
| `SLAM/MeshGeneration` | `voxblox/src/mesh/mesh_integrator.cc` – `generateMesh()` | Magenta |

## How it works

`profiler_patch.sh` uses `awk`/`sed` to insert `EASY_NONSCOPED_BLOCK` macros
at the entry point of each function listed above, all guarded by
`#ifdef BUILD_WITH_EASY_PROFILER`.  `cmake_patch.sh` adds
`option(WITH_PROFILER ...)` and `find_package(easy_profiler)` to
`voxblox_ros/CMakeLists.txt`.

`voxblox_node.cc` is patched to call `EASY_PROFILER_ENABLE` after
`ros::init` and to install a `SIGINT` handler that calls
`profiler::dumpBlocksToFile("/output/voxblox.prof")` before shutdown.

## Build (Docker)

```bash
# From SLAM_zero_to_hero/profiling/
docker build \
  -f voxblox/Dockerfile.profiler \
  -t voxblox-profiler \
  .
```

## Run

```bash
docker run --rm \
  -v /path/to/your.bag:/data/input.bag:ro \
  -v $(pwd)/output:/output \
  voxblox-profiler \
  roslaunch voxblox_ros euroc.launch \
    voxblox_path:=/output/voxblox.prof
```

Send `SIGINT` (Ctrl-C) to trigger the dump, then convert:

```bash
easy_profiler_converter /output/voxblox.prof /output/voxblox.json
```

## Datasets

| Dataset | Topics needed | Notes |
|---|---|---|
| [cow_and_lady](https://github.com/ethz-asl/voxblox#dataset) | `/cam0/image_raw`, `/imu0`, `/tf` | Standard ETH ASL benchmark |
| EuRoC MAV | `/cam0/image_raw`, `/imu0` | Used with voxblox's `euroc.launch` |
| Rosbag with `/velodyne_points` + TF | `/velodyne_points`, `/tf`, `/tf_static` | LiDAR mode; needs external pose source |

Download cow_and_lady:
```bash
wget http://rpg.ifi.uzh.ch/docs/IJRR17_Loquercio/datasets/cow_and_lady_dataset.bag
```
