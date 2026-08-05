#!/usr/bin/env bash
# Capture the rviz view of a running SVO Pro demo to a PNG, with no host X
# server involved: rviz renders onto a private Xvfb display, so the screenshot
# contains the visualisation and nothing else.
#
#   capture_rviz.sh <bag> [stereo|mono] [capture_at_seconds]
#
# capture_at_seconds is measured from the start of bag playback (default 75),
# late enough for several laps of trajectory and a populated landmark cloud.
set -euo pipefail

BAG=${1:?usage: capture_rviz.sh <bag> [stereo|mono] [capture_at_seconds]}
MODE=${2:-stereo}
AT=${3:-75}

OUT=${OUT:-/results/$(basename "${BAG%.bag}")_${MODE}}
mkdir -p "$OUT"
SHOT="$OUT/rviz.png"

export DISPLAY=:99
Xvfb :99 -screen 0 1600x900x24 -nolisten tcp >"$OUT/xvfb.log" 2>&1 &
XVFB_PID=$!
cleanup() {
  set +e
  [ -n "${RUN_PID:-}" ] && kill "$RUN_PID" 2>/dev/null
  # run_fpv.sh spawns roscore/rviz/rosbag as children; take the whole group.
  pkill -f "rosbag play"   2>/dev/null
  pkill -f rviz            2>/dev/null
  pkill -f svo_node        2>/dev/null
  sleep 1
  kill "$XVFB_PID" 2>/dev/null
}
trap cleanup EXIT

# Wait for the X server to accept connections before launching anything on it.
for _ in $(seq 1 40); do
  xdpyinfo -display :99 >/dev/null 2>&1 && break
  sleep 0.25
done
xdpyinfo -display :99 >/dev/null 2>&1 || { echo "Xvfb :99 never came up" >&2; exit 1; }

OUT="$OUT" /svo_ws/scripts/run_fpv.sh "$BAG" "$MODE" >"$OUT/capture_run.log" 2>&1 &
RUN_PID=$!

echo "capturing at t+${AT}s of playback ..."
# run_fpv.sh only starts the bag once svo_node advertises, so wait for the
# playback log to exist rather than assuming playback began immediately.
for _ in $(seq 1 240); do
  [ -f "$OUT/play.log" ] && break
  sleep 0.5
done
sleep "$AT"

import -display :99 -window root "$SHOT"
echo "wrote $SHOT"
identify "$SHOT"
