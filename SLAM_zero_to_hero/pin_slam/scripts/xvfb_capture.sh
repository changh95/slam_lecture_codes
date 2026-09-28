#!/bin/bash
# Run PIN-SLAM with its Open3D GUI (-v) on a virtual X display and grab
# screenshots of the whole screen.  Used for headless verification.
#
# Usage (inside the container):
#   demo_scripts/xvfb_capture.sh <out_dir> <interval_s> <pin_slam args...>
# Screenshots land in <out_dir>/gui_XXXX.png; the last one is copied to
# <out_dir>/gui_final.png.  PIN-SLAM's GUI keeps running after SLAM finishes,
# so pass -m: the script stops once the final global mesh is built and has
# had time to render.
set -u
OUT=$1; INTERVAL=$2; shift 2
mkdir -p "$OUT"
Xvfb :99 -screen 0 1920x1080x24 +extension GLX +render -noreset >/dev/null 2>&1 &
XVFB=$!
export DISPLAY=:99
sleep 2
python3 pin_slam.py "$@" 2>&1 | tee "$OUT/pin_slam_gui.log" &
i=0
while true; do
    sleep "$INTERVAL"
    i=$((i+1))
    xwd -root -silent | convert xwd:- "$OUT/gui_$(printf %04d $i).png"
    if grep -q "Reconstructing the global mesh done" "$OUT/pin_slam_gui.log" 2>/dev/null; then
        # SLAM loop is over; give the GUI time to draw the final map / mesh
        sleep 60
        xwd -root -silent | convert xwd:- "$OUT/gui_final.png"
        break
    fi
done
pkill -f pin_slam.py
kill $XVFB
