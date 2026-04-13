#!/bin/bash
# Entrypoint for Kimera-VIO + Rerun bridge container.
# Expects a rosbag at /data/input.bag (EuRoC MAV recorded via Kimera-VIO-ROS).
#
# Environment variables:
#   BAG_PATH        Path to the rosbag (default: /data/input.bag)
#   LAUNCH_FILE     ROS launch file name (default: kimera_vio_ros_euroc.launch)
#   BAG_RATE        Playback rate multiplier (default: 1.0)
set -e

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# 1. roscore
roscore &
ROSCORE_PID=$!
sleep 3

# 2. Kimera-VIO-ROS launch (visualization disabled; Rerun replaces RViz)
LAUNCH_FILE="${LAUNCH_FILE:-kimera_vio_ros_euroc.launch}"
roslaunch kimera_vio_ros "$LAUNCH_FILE" \
    use_lcd:=false \
    visualize:=false \
    &
KIMERA_PID=$!
sleep 6

# 3. Rerun bridge (subscribes to Kimera topics + serves web viewer)
python3 /app/ros_rerun_bridge.py &
BRIDGE_PID=$!
sleep 2

echo ""
echo "=== Kimera-VIO + Rerun ==="
echo "    Open: http://localhost:9090/?url=ws://localhost:9877"
echo ""

# 4. Play the rosbag (blocks until done)
BAG="${BAG_PATH:-/data/input.bag}"
BAG_RATE="${BAG_RATE:-1.0}"
if [ -f "$BAG" ]; then
    echo "Playing rosbag: $BAG  (rate=${BAG_RATE})"
    rosbag play "$BAG" --clock -r "$BAG_RATE"
else
    echo "No rosbag found at $BAG"
    echo "Set BAG_PATH env var or mount a bag at /data/input.bag"
    echo "Keeping bridge alive for manual topic publishing..."
fi

echo "Bag finished. Bridge still serving on :9090. Ctrl+C to exit."
wait $BRIDGE_PID
