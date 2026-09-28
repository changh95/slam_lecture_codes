#!/usr/bin/env bash
# Run MonoGS with its GUI inside a virtual X server and screenshot the window
# every SHOT_EVERY seconds. The GUI closes itself when SLAM ends, so the last
# screenshot is the (almost) final map.
#
#   run_gui_capture.sh <config.yaml> [out_dir]
#
# env: SHOT_EVERY (s, default 15), RES (default 1920x1080)
set -euo pipefail
CFG=${1:-configs/rgbd/tum/fr1_desk.yaml}
OUT=${2:-/MonoGS/results/gui_shots}
SHOT_EVERY=${SHOT_EVERY:-15}
RES=${RES:-1920x1080}
mkdir -p "$OUT"
cd /MonoGS

Xvfb :99 -screen 0 ${RES}x24 +extension GLX >/dev/null 2>&1 &
XVFB=$!
trap 'kill $XVFB 2>/dev/null || true' EXIT
export DISPLAY=:99
sleep 2

python3 slam.py --config "$CFG" &
SLAM=$!
i=0
while kill -0 $SLAM 2>/dev/null; do
  sleep "$SHOT_EVERY"
  kill -0 $SLAM 2>/dev/null || break
  import -window root "$OUT/shot_$(printf %03d $i).png" 2>/dev/null || true
  i=$((i+1))
done
wait $SLAM
echo "[run_gui_capture] $i screenshots in $OUT"
