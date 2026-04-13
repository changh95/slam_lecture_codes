#!/bin/bash
# Entrypoint for GLIM-GPU + Rerun bridge container.
# Uses GPU VGICP (IntegratedVGICPFactorGPU) for scan registration.
set -e
source /opt/ros/noetic/setup.bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

cd /glim/config
# Use GPU config variants
sed -i "s/config_odometry_cpu/config_odometry_gpu/g" config.json

IMU_TOPIC="${IMU_TOPIC:-/os_cloud_node/imu}"
POINTS_TOPIC="${POINTS_TOPIC:-/os_cloud_node/points}"
# HILTI 2022 Hesai PandarXT-32 <- Alphasense IMU (TUM tx,ty,tz,qx,qy,qz,qw)
T_LIDAR_IMU="${T_LIDAR_IMU:--0.001,-0.00855,0.055,0.7071068,-0.7071068,0.0,0.0}"
export IMU_TOPIC POINTS_TOPIC T_LIDAR_IMU

python3 - << 'PY'
import re, json, os

def strip_comments(s: str) -> str:
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)  # /* ... */ blocks
    s = re.sub(r"//.*", "", s)             # // line comments
    return s

# Patch config_ros.json
with open("config_ros.json") as f:
    text = strip_comments(f.read())
cfg = json.loads(text)
cfg["glim_ros"]["imu_topic"] = os.environ["IMU_TOPIC"]
cfg["glim_ros"]["points_topic"] = os.environ["POINTS_TOPIC"]
cfg["glim_ros"]["image_topic"] = ""
cfg["glim_ros"]["extension_modules"] = ["librviz_viewer.so"]
with open("config_ros.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(f"config_ros.json: imu={cfg['glim_ros']['imu_topic']} points={cfg['glim_ros']['points_topic']}")

# Patch config_sensors.json with extrinsic calibration
with open("config_sensors.json") as f:
    text = strip_comments(f.read())
sensors = json.loads(text)
tli = [float(x) for x in os.environ["T_LIDAR_IMU"].split(",")]
assert len(tli) == 7, "T_LIDAR_IMU must be 7 floats: tx,ty,tz,qx,qy,qz,qw"
sensors["sensors"]["T_lidar_imu"] = tli
with open("config_sensors.json", "w") as f:
    json.dump(sensors, f, indent=2)
print(f"config_sensors.json: T_lidar_imu={tli}")
PY

roscore &
sleep 3

BAG="${BAG_PATH:-/data/input.bag}"
echo "=== GLIM (GPU VGICP) on $BAG ==="
/usr/local/lib/glim_ros/glim_rosbag _config_path:=/glim/config "$BAG" &
GLIM_PID=$!
sleep 3

python3 /app/ros_rerun_bridge.py &
BRIDGE_PID=$!

echo ""
echo "=== Open http://localhost:9090/?url=ws://localhost:9877 ==="
echo ""

wait $GLIM_PID
echo "GLIM finished. Bridge still serving on :9090. Ctrl+C to exit."
wait $BRIDGE_PID
