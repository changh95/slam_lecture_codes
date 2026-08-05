#!/bin/bash
# sensor_bringup.sh — Initialize PTP + Fast DDS, then launch all sensor drivers.
# Prerequisites (one-time):
#   ./setup_depthai.sh      — install depthai apt deps
#   ./setup_fastdds.sh      — tune kernel UDP buffers for DDS
# Ctrl+C stops everything cleanly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Track background PIDs ──
PIDS=()

cleanup() {
    echo ""
    echo "=== Shutting down all sensor drivers ==="

    # Send SIGINT to each ros2 launch process (it propagates to child nodes)
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Sending SIGINT to PID $pid ..."
            kill -INT "$pid" 2>/dev/null || true
        fi
    done

    # Give ros2 launch up to 5 seconds to tear down nodes gracefully
    for i in $(seq 1 5); do
        all_dead=true
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                all_dead=false
                break
            fi
        done
        if $all_dead; then break; fi
        sleep 1
    done

    # Force-kill anything still alive
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Force-killing PID $pid ..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    wait 2>/dev/null || true
    echo "=== All sensor processes stopped ==="
    exit 0
}

trap cleanup SIGINT SIGTERM

# ═══════════════════════════════════════════════════════════
# 1. Initialize PTP Master
# ═══════════════════════════════════════════════════════════
echo "============================================"
echo "[1/6] Initializing PTP Master ..."
echo "============================================"
"$SCRIPT_DIR/init_ptp_master.sh"
echo ""

# ═══════════════════════════════════════════════════════════
# 2. Configure Fast DDS
# ═══════════════════════════════════════════════════════════
echo "============================================"
echo "[2/6] Configuring Fast DDS middleware ..."
echo "============================================"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

FASTDDS_XML="$SCRIPT_DIR/config/fastdds_profile.xml"
if [ -f "$FASTDDS_XML" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$FASTDDS_XML"
    echo "  FASTRTPS_PROFILES   = $FASTRTPS_DEFAULT_PROFILES_FILE"
else
    unset FASTRTPS_DEFAULT_PROFILES_FILE
    echo "  FASTRTPS_PROFILES   = (not found: $FASTDDS_XML — using Fast DDS defaults)"
fi
echo "  RMW_IMPLEMENTATION  = $RMW_IMPLEMENTATION"
echo "  ROS_DOMAIN_ID       = $ROS_DOMAIN_ID"

# Warn if kernel UDP buffers are still at the tiny default
RMEM_MAX=$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)
if [ "$RMEM_MAX" -lt 67108864 ]; then
    echo "  WARNING: net.core.rmem_max=${RMEM_MAX} is below 64 MB."
    echo "           Run ./setup_fastdds.sh once to fix this (requires sudo)."
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 3. Source ROS 2 Humble + workspace overlay
# ═══════════════════════════════════════════════════════════
echo "============================================"
echo "[3/6] Sourcing ROS 2 Humble + workspace ..."
echo "============================================"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
echo "  Done."
echo ""

# ═══════════════════════════════════════════════════════════
# 4. Launch livox_ros_driver2 (MID360)
# ═══════════════════════════════════════════════════════════
echo "============================================"
echo "[4/6] Launching livox_ros_driver2 (MID360) ..."
echo "============================================"
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
PIDS+=($!)
echo "  PID: ${PIDS[-1]}"
sleep 2
echo ""

# ═══════════════════════════════════════════════════════════
# 5. Launch livox_ros2_driver (Avia / older LiDAR models)
# ═══════════════════════════════════════════════════════════
echo "============================================"
echo "[5/6] Launching livox_ros2_driver ..."
echo "============================================"
ros2 launch livox_ros2_driver livox_lidar_launch.py &
PIDS+=($!)
echo "  PID: ${PIDS[-1]}"
sleep 2
echo ""

# ═══════════════════════════════════════════════════════════
# 6. Launch depthai camera driver (Oak-D Pro, 640×480)
# ═══════════════════════════════════════════════════════════
echo "============================================"
echo "[6/6] Launching depthai_ros_driver (640x480) ..."
echo "============================================"
ros2 launch depthai_ros_driver camera.launch.py \
    camera_model:=OAK-D-PRO \
    params_file:="$SCRIPT_DIR/config/camera_640x480.yaml" &
PIDS+=($!)
echo "  PID: ${PIDS[-1]}"
echo ""

# ═══════════════════════════════════════════════════════════
echo "============================================"
echo "  All sensors launched.  PIDs: ${PIDS[*]}"
echo "  Press Ctrl+C to stop all drivers."
echo "============================================"

# Block until any child exits or we receive a signal
wait
