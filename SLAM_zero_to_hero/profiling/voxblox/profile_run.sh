#!/bin/bash
# Inside-container script: launch voxblox + push the cow_and_lady bag
# through it via bag_player.py (because the noetic C++ rosbag player
# stalls at ~0.2 Hz on aarch64), then signal voxblox to dump the
# easy_profiler trace to /output/voxblox_profiler.prof.
set -e
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

mkdir -p /output

BAG="${BAG_PATH:-/data/input.bag}"
LAUNCH="${LAUNCH_FILE:-cow_and_lady_dataset.launch}"
RATE="${PLAYBACK_RATE:-2.0}"   # 2x for faster profiling

if [ ! -f "$BAG" ]; then
  echo "ERROR: no bag at $BAG" >&2
  exit 1
fi

# 1. roscore
roscore &
sleep 3

# 2. voxblox_node (without the launch file's own bag player)
roslaunch voxblox_ros "$LAUNCH" play_bag:=false &
LAUNCH_PID=$!
sleep 5

# 3. Push the bag through voxblox via the custom Python player.
echo "Playing $BAG at rate ${RATE}x"
python3 /app/bag_player.py "$BAG" "$RATE"
echo "Bag finished, waiting 2s for voxblox to drain..."
sleep 2

# 4. SIGINT voxblox so its atexit handler dumps the profile.
PID=$(pgrep -f tsdf_server || true)
if [ -n "$PID" ]; then
  echo "Sending SIGINT to voxblox_node (pid $PID) to dump profile..."
  kill -INT "$PID" || true
  # Wait up to 10s for the dump file to appear.
  for i in $(seq 1 10); do
    if [ -f /output/voxblox_profiler.prof ]; then
      echo "Profile dumped: $(ls -lh /output/voxblox_profiler.prof)"
      exit 0
    fi
    sleep 1
  done
  echo "WARN: profile dump not found after 10s" >&2
fi
kill "$LAUNCH_PID" 2>/dev/null || true
