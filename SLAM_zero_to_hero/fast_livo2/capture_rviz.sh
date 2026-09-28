#!/usr/bin/env bash
# Screenshot FAST-LIVO2's rviz view (colourized map + path) to a PNG with no
# host X server: rviz renders onto a private Xvfb display inside the container
# while run_avia.sh plays the bag, and ImageMagick grabs that display.
#
#   capture_rviz.sh [capture_at_seconds ...]     (default: 130, near bag end)
#
# Mounts/env are the same as run_avia.sh (/data, /out, avia.yaml, /run.sh); mount
# config/fast_livo2_overview.rviz over rviz_cfg/fast_livo2.rviz for the whole-loop view.
# Several times may be given; each writes /out/rviz_t<sec>.png.
set -uo pipefail

TIMES=("$@")
[ ${#TIMES[@]} -eq 0 ] && TIMES=(130)

export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export DISABLE_ROS1_EOL_WARNINGS=1   # rviz otherwise opens a modal EOL dialog
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp >/out/xvfb.log 2>&1 &
XVFB_PID=$!
for _ in $(seq 1 40); do
  xdpyinfo -display :99 >/dev/null 2>&1 && break
  sleep 0.25
done
xdpyinfo -display :99 >/dev/null 2>&1 || { echo "Xvfb :99 never came up" >&2; exit 1; }

RVIZ=true bash /run.sh >/out/capture_run.log 2>&1 &
RUN_PID=$!

# run_avia.sh starts rosbag play only after FAST-LIVO2 advertises its topics.
for _ in $(seq 1 240); do
  grep -q "rosbag play" /out/capture_run.log 2>/dev/null && break
  sleep 0.5
done
echo "bag playback started"

ELAPSED=0
for T in "${TIMES[@]}"; do
  sleep $((T - ELAPSED)); ELAPSED=$T
  import -display :99 -window root "/out/rviz_t${T}.png"
  echo "wrote /out/rviz_t${T}.png"
done

wait "$RUN_PID"
kill "$XVFB_PID" 2>/dev/null
cat /out/capture_run.log
