#!/bin/bash
# Run DSP-SLAM on KITTI 07 inside a virtual X display and screenshot the
# Pangolin map viewer (with the reconstructed car meshes) and the frame window.
#
#   /DSP-SLAM/scripts/run_kitti07_headless.sh [out_dir]
#
# out_dir (default /results) receives map/ (Cameras.txt, MapObjects.txt,
# MapPoints.txt, CameraTrajectory.txt), dsp_slam.log and the screenshots.
# SHOT_EVERY=N also screenshots the map every N seconds while running (default
# off: the viewer looks at the start of the sequence, which the car leaves for
# most of the run and only comes back to at the loop closure).
set -u
OUT=${1:-/results}
SHOT_EVERY=${SHOT_EVERY:-0}
mkdir -p "$OUT/map"
cd /DSP-SLAM

export DISPLAY=:99
Xvfb :99 -screen 0 1920x1200x24 +extension GLX >/dev/null 2>&1 &
XVFB=$!
sleep 2

START=$(date +%s)
./dsp_slam Vocabulary/ORBvoc.bin configs/KITTI04-12.yaml /data/kitti/07 "$OUT/map" \
    > "$OUT/dsp_slam.log" 2>&1 &
PID=$!

# Xvfb has no window manager, so Pangolin never receives the ConfigureNotify it
# sizes its views from, and draws into a corner. Resizing the window once sends
# it. The OpenCV frame window opens on top of the map, so move it below.
until xdotool search --name "Map Viewer" >/dev/null 2>&1 && \
      xdotool search --name "Current Frame" >/dev/null 2>&1; do
    kill -0 $PID 2>/dev/null || break
    sleep 1
done
sleep 3
MV=$(xdotool search --name "Map Viewer" | head -1)
for w in $(xdotool search --name "Current Frame"); do xdotool windowmove "$w" 0 740; done
xdotool windowsize "$MV" 1280 720

shot_map()   { import -window "$MV" "$OUT/$1.png" 2>/dev/null && echo "[shot] $OUT/$1.png"; }
shot_root()  { import -window root -crop 1280x1130+0+0 "$OUT/$1.png" 2>/dev/null && echo "[shot] $OUT/$1.png"; }

n=0
while kill -0 $PID 2>/dev/null; do
    sleep 5
    if grep -q "Map and trajectory saved" "$OUT/dsp_slam.log"; then
        END=$(date +%s)
        echo "[run] tracked 1101 frames in $((END - START)) s (including start-up)"
        sleep 5                         # let the viewer draw the final map
        shot_map  map_final
        shot_root screen_final          # map viewer + frame window
        # Zoom out (scroll wheel over the view) for a wider view of the map.
        xdotool mousemove 727 360
        for i in $(seq 1 ${ZOOM_OUT_CLICKS:-8}); do xdotool click 5; sleep 0.3; done
        sleep 2
        shot_map  map_final_overview
        break
    fi
    n=$((n + 5))
    if (( SHOT_EVERY > 0 && n % SHOT_EVERY == 0 )); then shot_map "map_$(printf %04d $n)s"; fi
done

kill $PID 2>/dev/null; sleep 1; kill -9 $PID 2>/dev/null
kill $XVFB 2>/dev/null
grep -q "Map and trajectory saved" "$OUT/dsp_slam.log" || {
    echo "[run] dsp_slam exited before saving the map"; tail -30 "$OUT/dsp_slam.log"; exit 1; }
exit 0
