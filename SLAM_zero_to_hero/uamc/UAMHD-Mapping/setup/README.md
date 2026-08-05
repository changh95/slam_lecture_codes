# setup/

Data-acquisition scripts for the onboard computer (Ubuntu 22.04 + ROS 2 Humble).
See **§4 Software setup** in the top-level `README.md` for the full walkthrough.

Run order:

**One-time (per machine):**
1. `setup_depthai.sh` — install depthai + ROS 2 apt deps, pin `depthai-ros` to `humble`.
2. `setup_fastdds.sh` — tune kernel UDP buffers for large DDS messages (`sudo`).
3. `build_ws.sh` — prepare Livox driver for ROS 2 and `colcon build` the workspace.

**Per run:**
4. `sensor_bringup.sh` — start PTP master, configure Fast DDS, source the workspace, launch LiDAR + camera drivers. `Ctrl+C` stops all cleanly.
5. `bag_record.sh` — record sensor topics to a timestamped ROS 2 bag.

`init_ptp_master.sh` is called by `sensor_bringup.sh`; it can also be run standalone.

**Per-hardware edits to make:**
- `init_ptp_master.sh` → `IFACE` (NIC connected to the LiDARs; default `eno1`).
- `bag_record.sh` → `LIDAR_A` / `LIDAR_B` (per-unit Livox topic IDs — discover with `ros2 topic list | grep livox`).
- `sensor_bringup.sh` → the launch files / camera model if your sensor set differs.

> These scripts expect a ROS 2 workspace layout (`src/`, `install/`) under this directory. The driver sources (`livox_ros_driver2`, `depthai-ros`, …) are cloned in, not vendored here.
