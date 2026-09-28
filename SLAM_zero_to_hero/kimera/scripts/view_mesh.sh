#!/bin/bash
# Show a saved Kimera-Semantics mesh in RViz without replaying the bag:
# roscore -> view_mesh.py (PLY -> voxblox mesh msg) -> RViz.
#
#   view_mesh.sh [mesh.ply] [rviz config]
#   env: HEADLESS=1    private Xvfb, screenshot to $OUT/rviz_view_mesh.png, then exit
#        OUT=/out
set -e
PLY=${1:-/out/kimera_semantics_mesh.ply}
CFG=${2:-/kimera/config/kimera_semantics_uhumans2.rviz}
OUT=${OUT:-/out}
if [ -z "$DISPLAY" ]; then HEADLESS=1; fi
HEADLESS=${HEADLESS:-0}
[ -f "$PLY" ] || { echo "mesh not found: $PLY"; exit 1; }

cleanup() { kill $(jobs -p) 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

roscore > /dev/null 2>&1 &
until rostopic list > /dev/null 2>&1; do sleep 0.5; done
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 world view_mesh > /dev/null 2>&1 &
if [ "$HEADLESS" = 1 ]; then
  Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &
  export DISPLAY=:99
  sleep 2
fi
# NVIDIA GL when the container got the GPU (nvidia-container-runtime) and a real
# display; Mesa software GL otherwise (Xvfb has no GPU)
if [ "$HEADLESS" = 0 ] && ldconfig -p | grep -q libGLX_nvidia; then
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
else
  export LIBGL_ALWAYS_SOFTWARE=1
fi
echo "[view] RViz GL: $(glxinfo -B 2>/dev/null | grep 'renderer string' | cut -d: -f2)"
rviz -d "$CFG" > /dev/null 2>&1 &
sleep 5
python3 /kimera/scripts/view_mesh.py "$PLY" &
until rostopic echo -n1 /kimera_semantics_node/mesh > /dev/null 2>&1; do sleep 1; done
if [ "$HEADLESS" = 1 ]; then
  sleep 15   # software GL needs a while for a few million triangles
  import -window root "$OUT/rviz_view_mesh.png"
  echo "[view] screenshot: $OUT/rviz_view_mesh.png"
else
  echo "[view] Ctrl-C to quit"; wait
fi
