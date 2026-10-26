#!/usr/bin/env bash
# Run U-AMC's FAST-LIVO2-ROS2 on a UAMC ROS 2 bag (Avia or Mid-360 LiDAR, its IMU and, for LVI,
# the Oak-D camera), optionally with RViz and a camera-coloured map from both LiDARs, and write the
# trajectory, the maps and screenshots to /out.
#
# Mounts expected inside the container:
#   /data     ~/data/gwanghwamun_coex, read-only (bags under extracted/)
#   /out      writable results dir
#   /uamc     this folder (config/, scripts/), read-only
# Env:
#   BAG=/data/extracted/lvi_coex_set_2   CONFIG=/uamc/config/coex_mid360_lvi.yaml
#   LAUNCH=           default by the config's lidar_type: mapping_mid360_lvi.launch.py for the
#                     Mid-360 (8), else mapping_aviz_lvi.launch.py (Avia)
#   RATE=1.0          bag playback rate
#   DURATION=         seconds of bag to play (empty = whole bag)
#   RVIZ=false        true = RViz live on $DISPLAY (Xvfb :99 is started if DISPLAY is unset)
#                     To watch on the host: -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
#                     GPU rendering there: --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
#                     (compute too: libGLX_nvidia needs libnvidia-gpucomp, else RViz silently falls back to llvmpipe)
#   HOLD_RVIZ=        true = after the run, leave RViz up with the final map until its window is
#                     closed; default true on a host display, false on Xvfb
#   SHOTS=""          bag-play seconds at which to grab the RViz window, e.g. "120 330"
#   RVIZ_CFG=         RViz config; default follows the scanner: uamc_lvi_follow.rviz / uamc_lio_follow.rviz
#                     by img_en, uamc_split_follow.rviz with COLOR_MAP. uamc_lio.rviz / uamc_lvi.rviz are
#                     fixed views of COEX in the Avia frame
#   COLOR_MAP=false   true = also play the Avia LiDAR and the camera and run color_mapper.py: both LiDARs
#                     placed by this run's poses and coloured from the camera -> /out/pcd/color_map.pcd.
#                     Needs a Mid-360 config (the mapper takes the poses as Mid-360 IMU poses)
#   IMG_OFFSET=       color_mapper.py --img-offset; default 1200.293 for lvi_coex_set_2, else 0
#   EXTRA_TOPICS=""   more bag topics to play than the config subscribes to
#   MAPPER=""         command run alongside FAST-LIVO2 (log: /out/mapper.log), SIGINTed at the end,
#                     e.g. "python3 /uamc/scripts/color_mapper.py /out/<seq_name>.txt /out/pcd/color_map.pcd"
set -uo pipefail

BAG="${BAG:-/data/extracted/lvi_coex_set_2}"
CONFIG="${CONFIG:-/uamc/config/coex_mid360_lvi.yaml}"
MID360=false; grep -q '^ *lidar_type: *8\b' "$CONFIG" && MID360=true
if [ "$MID360" = true ]; then LAUNCH="${LAUNCH:-mapping_mid360_lvi.launch.py}"
else LAUNCH="${LAUNCH:-mapping_aviz_lvi.launch.py}"; fi
COLOR_MAP="${COLOR_MAP:-false}"
case "$BAG" in *lvi_coex_set_2*) IMG_OFFSET="${IMG_OFFSET:-1200.293}" ;; *) IMG_OFFSET="${IMG_OFFSET:-0}" ;; esac
if [ "$COLOR_MAP" = true ] && [ "$MID360" != true ]; then
  echo "[run] COLOR_MAP=true needs a Mid-360 config (lidar_type: 8), e.g. coex_mid360_lio.yaml"; exit 1
fi
RATE="${RATE:-1.0}"
DURATION="${DURATION:-}"
RVIZ="${RVIZ:-false}"
SHOTS="${SHOTS:-}"
[ "$RVIZ" = true ] || SHOTS=""   # screenshots are of the RViz window
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
  # On the host display with the NVIDIA driver passed in (--gpus all, or the OCI hook with
  # NVIDIA_DRIVER_CAPABILITIES=graphics), render on the GPU; Xvfb and no-GPU runs stay on llvmpipe.
  if [ -z "$XVFB_PID" ] && ldconfig -p | grep -q libGLX_nvidia; then
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
  else
    export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
  fi
  echo "[run] RViz OpenGL renderer: $(glxinfo -B 2>/dev/null | sed -n 's/^OpenGL renderer string: //p')"
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
  # LiDAR-visual-inertial clouds carry camera RGB; LiDAR-inertial ones only intensity, and the
  # split view shows the coloured map beside them.
  if [ "$COLOR_MAP" = true ]; then RVIZ_CFG=${RVIZ_CFG:-/uamc/config/uamc_split_follow.rviz}
  elif grep -q '^ *img_en: *1' "$CONFIG"; then RVIZ_CFG=${RVIZ_CFG:-/uamc/config/uamc_lvi_follow.rviz}
  else RVIZ_CFG=${RVIZ_CFG:-/uamc/config/uamc_lio_follow.rviz}; fi
  rviz2 -d "$RVIZ_CFG" --ros-args -p use_sim_time:=true >$OUT/rviz.log 2>&1 &
  RVIZ_PID=$!
fi
for _ in $(seq 1 60); do
  ros2 topic list 2>/dev/null | grep -q '^/aft_mapped_to_init$' && break; sleep 1
done
sleep 5

# Play only the LiDAR, IMU and image topics the config subscribes to. --clock publishes
# /clock for the use_sim_time: true in the configs (and RViz above).
cfg() { sed -n "s/^ *$1: *\"\([^\"]*\)\".*/\1/p" "$CONFIG"; }
TOPICS=($(cfg lid_topic) $(cfg imu_topic))
grep -q '^ *img_en: *1' "$CONFIG" && TOPICS+=($(cfg img_topic))
TOPICS+=(${EXTRA_TOPICS:-})
SEQ=$(cfg seq_name)
if [ "$COLOR_MAP" = true ]; then
  for t in /livox/lidar_3JEDM180010C211 /oak/rgb/image_raw; do   # uamc_geom.AVIA_TOPIC, IMAGE_TOPIC
    [[ " ${TOPICS[*]} " == *" $t "* ]] || TOPICS+=($t)
  done
  MAPPER=${MAPPER:-"python3 /uamc/scripts/color_mapper.py $OUT/$SEQ.txt $OUT/pcd/color_map.pcd --img-offset $IMG_OFFSET"}
fi
MAPPER_PID=""
if [ -n "${MAPPER:-}" ]; then
  ( eval exec "$MAPPER" ) >$OUT/mapper.log 2>&1 &
  MAPPER_PID=$!
fi
echo "[run] ros2 bag play $BAG -r $RATE --topics ${TOPICS[*]} ${DURATION:+(first $DURATION s)}"
SECONDS=0
if [ -n "$DURATION" ]; then
  timeout -s INT "$DURATION" ros2 bag play "$BAG" -r "$RATE" --clock --topics "${TOPICS[@]}" >$OUT/bag_play.log 2>&1 &
else
  ros2 bag play "$BAG" -r "$RATE" --clock --topics "${TOPICS[@]}" >$OUT/bag_play.log 2>&1 &
fi
PLAY_PID=$!
# podman stop (SIGTERM to this script) ends playback early but keeps the normal shutdown, so the
# trajectory and maps are still written. Use a long enough timeout, e.g. podman stop -t 900.
STOPPED=""
trap 'echo "[run] SIGTERM -> stopping"; STOPPED=1; kill -INT $PLAY_PID 2>/dev/null' TERM
# Closing the RViz window (clean exit 0) ends the run: stop the bag and fall through to the
# normal shutdown, so the trajectory and PCD are still written. An RViz crash (non-zero exit,
# see NOTES.md) does not stop the run.
RVIZ_RC=""
rviz_closed() {
  [ "$RVIZ" = true ] || return 1
  [ -z "$RVIZ_RC" ] && ! kill -0 $RVIZ_PID 2>/dev/null && { wait $RVIZ_PID; RVIZ_RC=$?; }
  [ "$RVIZ_RC" = 0 ]
}
for T in $SHOTS; do
  while [ $SECONDS -lt "$T" ] && kill -0 $PLAY_PID 2>/dev/null && ! rviz_closed; do sleep 1; done
  rviz_closed && break
  shot "$OUT/rviz_t${T}.png" && echo "[run] wrote rviz_t${T}.png"
done
while kill -0 $PLAY_PID 2>/dev/null; do
  if rviz_closed; then echo "[run] RViz closed -> stopping"; kill -INT $PLAY_PID 2>/dev/null; break; fi
  sleep 1
done
wait $PLAY_PID
WALL=$SECONDS
echo "[run] bag playback wall-clock: ${WALL} s"
rviz_closed || { sleep 10; [ "$RVIZ" = true ] && shot "$OUT/rviz_end.png" && echo "[run] wrote rviz_end.png"; }

# The trajectory is flushed per pose; the PCD map only as the node unwinds after SIGINT.
echo "[run] SIGINT -> nodes"
HOLD_RVIZ=${HOLD_RVIZ:-$([ -z "$XVFB_PID" ] && echo true || echo false)}
[ "$RVIZ" = true ] && [ "$HOLD_RVIZ" != true ] && kill -INT $RVIZ_PID 2>/dev/null
kill -INT $LAUNCH_PID 2>/dev/null
for _ in $(seq 1 120); do kill -0 $LAUNCH_PID 2>/dev/null || break; sleep 1; done
kill -KILL $LAUNCH_PID 2>/dev/null
# The mapper finishes the scans the last poses cover, then writes its map.
if [ -n "$MAPPER_PID" ]; then
  kill -INT $MAPPER_PID 2>/dev/null
  for _ in $(seq 1 600); do kill -0 $MAPPER_PID 2>/dev/null || break; sleep 1; done   # map build can take minutes
  kill -KILL $MAPPER_PID 2>/dev/null
  tail -1 $OUT/mapper.log
fi
[ -n "$XVFB_PID" ] && kill $XVFB_PID 2>/dev/null

grep -E "saved to|point count" $OUT/fastlivo.log | sed 's/\x1b\[[0-9;]*m//g'
# all_raw_points.pcd holds every coloured point of every scan. FAST-LIVO2's own 0.15 m
# downsampling overflows on a scene this size and writes the raw cloud a second time, so
# voxelise it here into pcd/map.pcd and drop both big files unless KEEP_RAW=1.
if [ -f $OUT/pcd/all_raw_points.pcd ]; then
  python3 /uamc/scripts/voxel_pcd.py $OUT/pcd/all_raw_points.pcd $OUT/pcd/map.pcd --leaf "${LEAF:-0.1}" \
    && { [ "${KEEP_RAW:-0}" = 1 ] || rm -f $OUT/pcd/all_raw_points.pcd $OUT/pcd/all_downsampled_points.pcd; }
fi
python3 /uamc/scripts/traj_stats.py "$OUT/$SEQ.txt" --wall $WALL | tee $OUT/stats.txt

# Everything is written; RViz keeps showing the final map until its window is closed
# (or podman stop). RViz already closed by the user skips this.
if [ "$RVIZ" = true ] && [ "$HOLD_RVIZ" = true ] && [ -z "$STOPPED" ] && kill -0 $RVIZ_PID 2>/dev/null; then
  echo "[run] done; RViz stays open with the final map -- close its window to exit"
  while kill -0 $RVIZ_PID 2>/dev/null && [ -z "$STOPPED" ]; do sleep 1; done
  kill -INT $RVIZ_PID 2>/dev/null
fi
