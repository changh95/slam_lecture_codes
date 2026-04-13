# Voxblox – Rerun Visualization

Live visualization of [voxblox](https://github.com/ethz-asl/voxblox) TSDF/ESDF
mapping output using [Rerun](https://rerun.io) 0.21.0 in a browser.

## Architecture

```
rosbag play  -->  voxblox_node  -->  /voxblox_node/tsdf_pointcloud
                                -->  /voxblox_node/esdf_pointcloud
                                -->  /voxblox_node/surface_pointcloud
                                -->  /voxblox_node/mesh
                                          |
                                   ros_rerun_bridge.py
                                          |
                               Rerun web viewer :9090
```

## Rerun streams

| Stream | Source topic | Description |
|---|---|---|
| `slam/tsdf_cloud` | `/voxblox_node/tsdf_pointcloud` | Per-frame TSDF surface points, coloured by height |
| `slam/tsdf_map` | accumulated | Voxel-deduplicated global TSDF map |
| `slam/esdf_cloud` | `/voxblox_node/esdf_pointcloud` | ESDF voxels coloured by distance-to-surface |
| `slam/esdf_map` | accumulated | Global ESDF map |
| `slam/surface_cloud` | `/voxblox_node/surface_pointcloud` | Extracted surface points (grey) |
| `slam/mesh` | `/voxblox_node/mesh` | Mesh vertices as point cloud |
| `world` | static | XYZ axis arrows |

## Build (Docker)

```bash
# From SLAM_zero_to_hero/rerun_viz/
docker build -f voxblox/Dockerfile -t voxblox-rerun voxblox/
```

## Run

```bash
docker run --rm \
  -v /path/to/cow_and_lady_dataset.bag:/data/input.bag:ro \
  -p 9090:9090 -p 9877:9877 \
  voxblox-rerun
```

Then open: **http://localhost:9090/?url=ws://localhost:9877**

### Override launch file

```bash
docker run --rm \
  -e LAUNCH_FILE=cow_and_lady.launch \
  -v /path/to/bag:/data/input.bag:ro \
  -p 9090:9090 -p 9877:9877 \
  voxblox-rerun
```

## Datasets

| Dataset | Download | Topics |
|---|---|---|
| cow_and_lady | `wget http://rpg.ifi.uzh.ch/docs/IJRR17_Loquercio/datasets/cow_and_lady_dataset.bag` | `/cam0/image_raw`, `/imu0`, `/tf` |
| EuRoC MAV | [ASL dataset page](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) | `/cam0/image_raw`, `/imu0` |
| LiDAR bag | custom | `/velodyne_points`, `/tf`, `/tf_static` (needs external pose) |

Voxblox's standard test dataset is **cow_and_lady** from ETH ASL.
For LiDAR-only setups, provide an external pose source (e.g. LOAM, LIO-SAM)
publishing to `/tf` alongside `/velodyne_points`.

## Requirements

- Docker with internet access (pulls `ros:noetic`, installs `rerun-sdk==0.21.0`)
- Ports 9090 and 9877 available on the host
- A rosbag compatible with the chosen voxblox launch file
