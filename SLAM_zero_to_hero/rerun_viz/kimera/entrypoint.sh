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

# 2. Rerun bridge FIRST, so it subscribes before kimera publishes anything
python3 /app/ros_rerun_bridge.py &
BRIDGE_PID=$!
sleep 3

# 3. Kimera-VIO-ROS in OFFLINE mode (reads bag itself). use_lcd=true
# enables /kimera_vio_ros/pose_graph from the LCD module.
LAUNCH_FILE="${LAUNCH_FILE:-kimera_vio_ros_euroc.launch}"
BAG="${BAG_PATH:-/data/input.bag}"
# Fast-loading ORBvoc.txt (DBoW2 native text format, ~seconds to load).
# We used to point at ORBvoc.yml, whose cv::FileStorage YAML parser has
# pathological O(n^2) behaviour on aarch64 and never finished loading
# the 258 MB file. DBoW2's load() auto-dispatches on extension: .txt is
# the fast path.
VOCAB_PATH="${VOCAB_PATH:-/vocab/ORBvoc.txt}"
if [ ! -f "$VOCAB_PATH" ]; then
    echo "WARN: $VOCAB_PATH missing, falling back to the slow ORBvoc.yml" >&2
    VOCAB_PATH="$(rospack find kimera_vio)/vocabulary/ORBvoc.yml"
fi
echo "Using ORB vocabulary: $VOCAB_PATH"

roslaunch kimera_vio_ros "$LAUNCH_FILE" \
    use_lcd:=false \
    visualize:=true \
    use_rviz:=true \
    viz_type:=0 \
    online:=false \
    rosbag_path:="$BAG" \
    &
KIMERA_PID=$!
sleep 3

# 4. Separate rosbag play (raw stereo topics only) so the bridge gets
# cam0/cam1 images. Kimera is in offline mode and reads the bag itself,
# so these topics don't conflict.
BAG_RATE="${BAG_RATE:-1.0}"
if [ -f "$BAG" ]; then
    echo "Playing rosbag in parallel for raw stereo display: $BAG"
    rosbag play --clock -r "$BAG_RATE" "$BAG" --topics /cam0/image_raw /cam1/image_raw &
    BAG_PID=$!
fi
sleep 1

echo ""
echo "=== Kimera-VIO + Rerun ==="
echo "    Open: http://localhost:9090/?url=ws://localhost:9877"
echo ""

# 5. Wait for kimera and the parallel bag-play to finish.
wait $KIMERA_PID 2>/dev/null || true
if [ -n "${BAG_PID:-}" ]; then
    wait "$BAG_PID" 2>/dev/null || true
fi
echo "Kimera + bag finished. Bridge still serving on :9090. Ctrl+C to exit."
wait $BRIDGE_PID
