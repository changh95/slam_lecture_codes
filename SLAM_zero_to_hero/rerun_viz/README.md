# Rerun Visualization for SLAM Systems

Live (or post-run) visualization of **ORB-SLAM2**, **FAST-LIO2**, and **GLIM**
using the [Rerun](https://rerun.io) viewer. Each system has its own Dockerfile
that builds an unmodified upstream SLAM stack, runs it on a dataset, and
forwards the output to a Rerun web viewer on port 9090.

## Strategy per system

| System      | Output source              | Approach                                  |
|-------------|----------------------------|-------------------------------------------|
| ORB-SLAM2   | `CameraTrajectory.txt`     | Run SLAM, then Python post-run replay     |
| FAST-LIO2   | ROS topics                 | ROS -> Rerun Python bridge (live)         |
| GLIM        | ROS topics (rviz_viewer)   | ROS -> Rerun Python bridge (live)         |

All three use the same viewer URL pattern:
**`http://localhost:9090/?url=rerun+http://localhost:9876/proxy`**

## Prerequisites

- Docker with `--gpus all` support (NVIDIA Container Toolkit)
- Dataset available on the host (KITTI for ORB-SLAM2, rosbag for FAST-LIO2/GLIM)

---

## ORB-SLAM2

**Build:**

```bash
cd orb_slam2/
docker build -t slam:orb_slam2-rerun .
```

**Run on KITTI sequence 00:**

```bash
docker run --rm -it \
  -p 9090:9090 -p 9876:9876 \
  -v ~/data/kitti_vo_slam/dataset:/data \
  slam:orb_slam2-rerun \
  /data/sequences/00
```

The container runs mono ORB-SLAM2 on the sequence, saves
`CameraTrajectory.txt`, then replays poses + left camera image into Rerun
at 10 fps (adjust with `--fps`).

Open **http://localhost:9090/?url=rerun+http://localhost:9876/proxy**

---

## FAST-LIO2

**Build:**

```bash
cd fast_lio2/
docker build -t slam:fast_lio2-rerun .
```

**Run with a rosbag:**

```bash
docker run --rm -it \
  -p 9090:9090 -p 9876:9876 \
  -v ~/data/fastlio_bags:/data \
  -e BAG_PATH=/data/my_sequence.bag \
  slam:fast_lio2-rerun
```

FAST-LIO2 is launched with `mapping_velodyne.launch` (rviz disabled). The
Python bridge subscribes to `/Odometry`, `/cloud_registered`, `/path` and
forwards them live to Rerun.

Open **http://localhost:9090/?url=rerun+http://localhost:9876/proxy**

**Using a different launch file** (e.g. Ouster, Livox Avia):
edit `entrypoint.sh` and change `mapping_velodyne.launch` to the matching
launch file and provide the correct bag topics.

---

## GLIM

**Build:**

```bash
cd glim/
docker build -t slam:glim-rerun .
```

This builds GTSAM 4.2.0, iridescence, gtsam_points v1.0.0, GLIM v1.0.0, and
glim_ros1 from upstream sources. Takes ~15-20 minutes.

**Run with a rosbag (e.g. HILTI 2022):**

```bash
docker run --rm -it \
  -p 9090:9090 -p 9876:9876 \
  -v ~/data/hilti_2022:/data \
  -e BAG_PATH=/data/exp14_basement_2.bag \
  -e IMU_TOPIC=/alphasense/imu \
  -e POINTS_TOPIC=/hesai/pandar \
  slam:glim-rerun
```

The entrypoint patches `config_ros.json` to your topic names, enables the
`librviz_viewer.so` extension (so GLIM publishes `/glim_ros/odom`,
`/glim_ros/points`, `/glim_ros/map`), runs `glim_rosbag`, and the Python
bridge forwards everything to Rerun.

Open **http://localhost:9090/?url=rerun+http://localhost:9876/proxy**

**Default topics** (if env vars are not set): `/os_cloud_node/imu`,
`/os_cloud_node/points` (Ouster). Override with `IMU_TOPIC` / `POINTS_TOPIC`.

---

## Browser URL quirk

The served HTML at `http://localhost:9090` does NOT auto-embed the gRPC
connect URL in rerun-sdk 0.31. You **must** open the viewer with the query
string:

```
http://localhost:9090/?url=rerun+http://localhost:9876/proxy
```

Without `?url=...` you will see Rerun's default intro page with no data.

## Port conflicts

Change the host-side mapping and match the port in the URL. E.g. to run two
SLAM systems simultaneously on ports 19090/19876 and 29090/29876:

```bash
docker run -p 19090:9090 -p 19876:9876 ... slam:fast_lio2-rerun
docker run -p 29090:9090 -p 29876:9876 ... slam:glim-rerun
```

Then open:
- `http://localhost:19090/?url=rerun+http://localhost:19876/proxy`
- `http://localhost:29090/?url=rerun+http://localhost:29876/proxy`

## What you'll see in each viewer

| Entity           | ORB-SLAM2 | FAST-LIO2 | GLIM |
|------------------|-----------|-----------|------|
| `slam/pose`      | Camera frame | LiDAR+IMU body | Odometry frame |
| `slam/pose/body` | Car box   | Sensor box | Sensor box |
| `slam/pose/cam`  | Left image | - | - |
| `slam/cloud`     | -         | Current scan (height-colored) | Current frame points |
| `slam/map`       | -         | (see `slam/cloud`) | Accumulated map |
| `slam/trajectory`| Running line | `/path` line strip | - |
| `world`          | XYZ axes  | XYZ axes  | XYZ axes |
