# Kimera-VIO + Rerun Visualization

Live visualization of [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO) output
using [Rerun](https://rerun.io) (v0.21.0, compatible with Python 3.8 / ROS Noetic).

Kimera-VIO-ROS publishes odometry, feature-track images, and mesh markers over ROS1.
The `ros_rerun_bridge.py` script subscribes to those topics and streams them to a
Rerun web viewer accessible from any browser — no RViz required.

## Architecture

```
rosbag  -->  Kimera-VIO-ROS  -->  ROS topics  -->  ros_rerun_bridge.py  -->  Rerun web viewer
                                                                              :9090 (HTTP)
                                                                              :9877 (WebSocket)
```

## Topics Visualized

| ROS topic | Rerun entity | Content |
|---|---|---|
| `/kimera_vio_ros/odometry` | `slam/pose` | VIO pose estimate |
| `/kimera_vio_ros/imu_odometry` | `slam/imu_pose` | IMU-propagated pose |
| `/kimera_vio_ros/optimized_odometry` | `slam/optimized_pose` | Loop-closure-corrected pose |
| `/kimera_vio_ros/path` | `slam/trajectory` | VIO trajectory (cyan) |
| `/kimera_vio_ros/optimized_path` | `slam/optimized_trajectory` | Optimized trajectory (green) |
| `/kimera_vio_ros/frontend/feature_tracks` | `slam/feature_tracks` | Tracked feature image |
| `/kimera_vio_ros/mesh` | `slam/mesh` | 3D mesh markers (best-effort) |

Fallback topics `/odometry` and `/path` are also subscribed for standalone Kimera-VIO use.

## Build

```bash
cd rerun_viz/
docker build -t slam:kimera-rerun \
    -f kimera/Dockerfile .
```

## Run

Mount an EuRoC MAV rosbag (recorded with Kimera-VIO-ROS) at `/data/input.bag`:

```bash
docker run --rm \
    -p 9090:9090 -p 9877:9877 \
    -v /path/to/euroc_bag.bag:/data/input.bag:ro \
    slam:kimera-rerun
```

Then open in a browser:
```
http://localhost:9090/?url=ws://localhost:9877
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BAG_PATH` | `/data/input.bag` | Path to the input rosbag |
| `LAUNCH_FILE` | `kimera_vio_ros_euroc.launch` | Kimera-VIO-ROS launch file |
| `BAG_RATE` | `1.0` | Rosbag playback speed multiplier |

## Dataset

**EuRoC MAV** — stereo + IMU sequences, recorded by ASL ETH Zurich.

Download (e.g. MH_01_easy):
```bash
wget http://rpg.ifi.uzh.ch/docs/IJRR17_Burri_EuRoC/MH_01_easy.zip
unzip MH_01_easy.zip
```

For use with Kimera-VIO-ROS you need a ROS bag version of the dataset.
Pre-made bags are available from the EuRoC dataset page:
```
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
```

## Notes

- `rerun-sdk==0.21.0` is pinned because it is the last version that supports Python 3.8
  (the default Python on Ubuntu 20.04 / ROS Noetic).
- The feature-track image subscriber requires `cv_bridge`. If unavailable, raw byte
  decoding is used as a fallback.
- Mesh visualization uses `visualization_msgs/Marker` (TRIANGLE_LIST / LINE_LIST).
  Full mesh support (pcl_msgs/PolygonMesh) is not yet implemented.
