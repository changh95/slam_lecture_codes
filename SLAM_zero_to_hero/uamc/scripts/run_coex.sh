#!/usr/bin/env bash
# Run U-AMC's FAST-LIVO2-ROS2 (Avia LiDAR + IMU + Oak-D RGB) on a UAMC ROS 2 bag,
# optionally with RViz, and write the trajectory, the coloured map and screenshots to /out.
#
# Mounts expected inside the container:
#   /data     ~/data/gwanghwamun_coex, read-only (bags under extracted/)
#   /out      writable results dir
#   /uamc     this folder (config/, scripts/), read-only
# Env:
#   BAG=/data/extracted/lvi_set_2_restamped   CONFIG=/uamc/config/coex_avia_lio.yaml
#   LAUNCH=mapping_aviz_lvi.launch.py   (the Avia launch; Mid-360 does not work on this bag, see NOTES.md)
#   RATE=1.0          bag playback rate
#   DURATION=         seconds of bag to play (empty = whole bag)
#   RVIZ=false        true = RViz live on $DISPLAY (Xvfb :99 is started if DISPLAY is unset)
#                     To watch on the host: -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
#   SHOTS=""          bag-play seconds at which to grab the RViz window, e.g. "120 330"
set -uo pipefail

BAG="${BAG:-/data/extracted/lvi_set_2_restamped}"
CONFIG="${CONFIG:-/uamc/config/coex_avia_lio.yaml}"
LAUNCH="${LAUNCH:-mapping_aviz_lvi.launch.py}"
RATE="${RATE:-1.0}"
DURATION="${DURATION:-}"
RVIZ="${RVIZ:-false}"
SHOTS="${SHOTS:-}"
OUT=/out
WS=${WS:-/fast_livo_ws}
PKG=$WS/src/fast_livo

set +u   # ROS setup scripts read unset variables
source /opt/ros/humble/setup.bash
source ${WS}/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-$((RANDOM % 100 + 100))}   # keep parallel containers apart
export ROS_LOCALHOST_ONLY=1

# FAST-LIVO2 writes to <source dir>/Log/{result,pcd}; point both at /out.
mkdir -p $OUT/pcd $PKG/Log
rm -rf $PKG/Log/result $PKG/Log/pcd
ln -s $OUT $PKG/Log/result
ln -s $OUT/pcd $PKG/Log/pcd

grep -q 'livox_ros_driver2/msg/CustomMsg' "$BAG/metadata.yaml" || {
  echo "[run] $BAG/metadata.yaml still types the Avia topic as livox_interfaces; run"
  echo "      python3 download_gwanghwamun_coex.py --extract --patch-type   first."; exit 1; }

# Job control on: a non-interactive shell otherwise starts background jobs with SIGINT
# ignored, and FAST-LIVO2 only writes its PCD map when SIGINT reaches it.
set -m

XVFB_PID=""
if [ "$RVIZ" = true ]; then
  if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:99
    Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp >$OUT/xvfb.log 2>&1 &
    XVFB_PID=$!
    for _ in $(seq 1 40); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.25; done
  fi
  export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
fi

# Grab our own RViz window (its title starts with the config path), not the root window:
# on a real desktop (DISPLAY passed in) the root window is whatever sits on top of RViz.
shot() {
  local w; w=$(xwininfo -display "$DISPLAY" -root -tree 2>/dev/null \
    | awk -v t="\"$RVIZ_CFG" 'index($0, t) && / - RViz": \(/ {print $1; exit}')
  import -display "$DISPLAY" -window "${w:-root}" "$1"
}

case "$LAUNCH" in *mid360*) PARAM_ARG=mid360_params_file ;; *) PARAM_ARG=avia_params_file ;; esac
echo "[run] ros2 launch fast_livo $LAUNCH $PARAM_ARG:=$CONFIG"
ros2 launch fast_livo "$LAUNCH" use_rviz:=False "$PARAM_ARG:=$CONFIG" >$OUT/fastlivo.log 2>&1 &
LAUNCH_PID=$!
if [ "$RVIZ" = true ]; then
  # LiDAR-visual-inertial clouds carry camera RGB; LiDAR-inertial ones only intensity.
  if grep -q '^ *img_en: *1' "$CONFIG"; then RVIZ_CFG=${RVIZ_CFG:-/uamc/config/uamc_lvi.rviz}
  else RVIZ_CFG=${RVIZ_CFG:-/uamc/config/uamc_lio.rviz}; fi
  rviz2 -d "$RVIZ_CFG" >$OUT/rviz.log 2>&1 &
  RVIZ_PID=$!
fi
for _ in $(seq 1 60); do
  ros2 topic list 2>/dev/null | grep -q '^/aft_mapped_to_init$' && break; sleep 1
done
sleep 5

# Play only the LiDAR, IMU and image topics the config subscribes to.
cfg() { sed -n "s/^ *$1: *\"\([^\"]*\)\".*/\1/p" "$CONFIG"; }
TOPICS=($(cfg lid_topic) $(cfg imu_topic))
grep -q '^ *img_en: *1' "$CONFIG" && TOPICS+=($(cfg img_topic))
echo "[run] ros2 bag play $BAG -r $RATE --topics ${TOPICS[*]} ${DURATION:+(first $DURATION s)}"
SECONDS=0
if [ -n "$DURATION" ]; then
  timeout -s INT "$DURATION" ros2 bag play "$BAG" -r "$RATE" --topics "${TOPICS[@]}" >$OUT/bag_play.log 2>&1 &
else
  ros2 bag play "$BAG" -r "$RATE" --topics "${TOPICS[@]}" >$OUT/bag_play.log 2>&1 &
fi
PLAY_PID=$!
for T in $SHOTS; do
  while [ $SECONDS -lt "$T" ] && kill -0 $PLAY_PID 2>/dev/null; do sleep 1; done
  shot "$OUT/rviz_t${T}.png" && echo "[run] wrote rviz_t${T}.png"
done
wait $PLAY_PID
WALL=$SECONDS
echo "[run] bag playback wall-clock: ${WALL} s"
sleep 10
[ "$RVIZ" = true ] && shot "$OUT/rviz_end.png" && echo "[run] wrote rviz_end.png"

# The trajectory is flushed per pose; the PCD map only as the node unwinds after SIGINT.
echo "[run] SIGINT -> nodes"
[ "$RVIZ" = true ] && kill -INT $RVIZ_PID 2>/dev/null
kill -INT $LAUNCH_PID 2>/dev/null
for _ in $(seq 1 120); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
kill -KILL $LAUNCH_PID 2>/dev/null
[ -n "$XVFB_PID" ] && kill $XVFB_PID 2>/dev/null

grep -E "saved to|point count" $OUT/fastlivo.log | sed 's/\x1b\[[0-9;]*m//g'
# all_raw_points.pcd holds every coloured point of every scan. FAST-LIVO2's own 0.15 m
# downsampling overflows on a scene this size and writes the raw cloud a second time, so
# voxelise it here into pcd/map.pcd and drop both big files unless KEEP_RAW=1.
if [ -f $OUT/pcd/all_raw_points.pcd ]; then
  python3 /uamc/scripts/voxel_pcd.py $OUT/pcd/all_raw_points.pcd $OUT/pcd/map.pcd --leaf "${LEAF:-0.1}" \
    && { [ "${KEEP_RAW:-0}" = 1 ] || rm -f $OUT/pcd/all_raw_points.pcd $OUT/pcd/all_downsampled_points.pcd; }
fi
SEQ=$(cfg seq_name)
python3 /uamc/scripts/traj_stats.py "$OUT/$SEQ.txt" --wall $WALL | tee $OUT/stats.txt
