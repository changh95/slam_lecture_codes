#!/bin/bash
# Play a KITTI velodyne sequence through the SuMa++ visualizer without anyone
# clicking: autorun mode (see NOTES.md) opens the first scan, plays to the end
# (or SUMA_MAX_SCANS), writes poses/runtime/screenshots to $OUT and quits.
#
#   run_autorun.sh <velodyne dir> [out dir] [max scans]
#
# With no $DISPLAY it starts its own Xvfb (Mesa llvmpipe, OpenGL 4.5 on the
# CPU). With -e DISPLAY=:1 (host X server) the NVIDIA OpenGL driver renders
# instead, which is much faster. RangeNet++ (TensorRT) is on the GPU either way.
set -e
VELO=${1:-/data/sequences/00/velodyne}
OUT=${2:-/results}
MAX=${3:-0}
CFG=${SUMA_CONFIG:-/ws/config/kitti.xml}
mkdir -p "$OUT"

if [ -z "$DISPLAY" ]; then
  export DISPLAY=:99
  Xvfb :99 -screen 0 1920x1080x24 +extension GLX -nolisten tcp >/tmp/xvfb.log 2>&1 &
  XVFB=$!
  sleep 2
fi
export QT_X11_NO_MITSHM=1 XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
glxinfo -B 2>/dev/null | grep -E "OpenGL renderer|OpenGL core profile version" | tee "$OUT/opengl.txt" || true

FIRST=$(ls "$VELO" | grep '\.bin$' | head -1)
export SUMA_AUTOPLAY=1 SUMA_EXIT=1 SUMA_OUTPUT_DIR="$OUT"
[ "$MAX" != "0" ] && export SUMA_MAX_SCANS=$MAX

cd /ws/semantic_suma/bin
T0=$(date +%s)
rc=0
./visualizer "$CFG" "$VELO/$FIRST" > "$OUT/visualizer.log" 2>&1 || rc=$?
T1=$(date +%s)
echo "visualizer exit code $rc, wall time: $((T1-T0)) s" | tee -a "$OUT/visualizer.log"
grep -E "\[autorun\] (finished|wrote)|ERROR|terminate|what\(\)" "$OUT/visualizer.log" || true
[ -n "$XVFB" ] && kill $XVFB 2>/dev/null || true
exit $rc
