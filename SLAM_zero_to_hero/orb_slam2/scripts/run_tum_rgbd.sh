#!/bin/bash
# Run the stock rgbd_tum (Pangolin Map Viewer + OpenCV Current Frame) on a TUM
# RGB-D sequence and screenshot both windows while it runs.
#
# Runs inside the container:
#   run_tum_rgbd.sh <sequence dir> [settings yaml] [associations file]
# Uses $DISPLAY if one is passed in (your desktop); otherwise starts a private
# Xvfb (but see README: under Xvfb the stock viewer aborts after ~30-40 s).
# Writes to the current directory (mount /out and use -w /out):
#   CameraTrajectory.txt, KeyFrameTrajectory.txt   (TUM format, from rgbd_tum)
#   viewer.png      Map Viewer | Current Frame, grabbed ~2 s before the end
#   shots/NNN.png   one grab every 2 s during the run
set -u
SEQ=${1:?usage: run_tum_rgbd.sh <sequence dir> [settings] [associations]}
YAML=${2:-/Portable_ORB_SLAM2/Examples/RGB-D/TUM1.yaml}
ASSOC=${3:-/Portable_ORB_SLAM2/Examples/RGB-D/associations/fr1_desk.txt}
VOC=/Portable_ORB_SLAM2/Vocabulary/ORBvoc.txt

XVFB=
if [ -z "${DISPLAY:-}" ]; then
    Xvfb :99 -screen 0 1680x800x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    XVFB=$!
    export DISPLAY=:99
    for _ in $(seq 100); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.1; done
fi

mkdir -p shots
/Portable_ORB_SLAM2/Examples/RGB-D/rgbd_tum "$VOC" "$YAML" "$SEQ" "$ASSOC" &
SLAM=$!

win() { xdotool search --onlyvisible --name "ORB-SLAM2: $1" 2>/dev/null | head -1; }

# Both windows open at the same spot and overlap: put them side by side.
moved=0; n=0
while kill -0 $SLAM 2>/dev/null; do
    mv=$(win "Map Viewer"); fr=$(win "Current Frame")
    if [ $moved = 0 ] && [ -n "$mv" ] && [ -n "$fr" ]; then
        xdotool windowmove "$mv" 0 0 windowmove "$fr" 1040 0; moved=1; sleep 1; continue
    fi
    # Only grab while both windows exist: after the last frame, Shutdown()
    # tears them down and a grab would be blank.
    if [ -n "$mv" ] && [ -n "$fr" ]; then
        f=$(printf 'shots/%03d.png' $n)
        if import -window "$mv" /tmp/mv.png 2>/dev/null && import -window "$fr" /tmp/fr.png 2>/dev/null; then
            convert /tmp/mv.png /tmp/fr.png -background black -gravity north +append "$f" && n=$((n+1))
        fi
    fi
    sleep 2
done
wait $SLAM; rc=$?
[ -n "$XVFB" ] && kill $XVFB 2>/dev/null

# The last grab can land mid-teardown; take the second-to-last as the result.
last=$(ls shots/*.png 2>/dev/null | tail -2 | head -1)
[ -n "$last" ] && cp "$last" viewer.png
echo "rgbd_tum exit code: $rc, screenshots: $n, viewer.png <- ${last:-none}"
exit $rc
