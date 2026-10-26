#!/usr/bin/env bash
# Render a finished run in RViz on a private Xvfb display and screenshot it:
# /out/pcd/map.pcd (coloured map) + /out/<SEQ>.txt (trajectory) -> /out/rviz_map_<view>.png
#
#   capture_map.sh [SEQ]           (default lvi_coex_set_2_mid360_lvi)
# Env: VIEWS="top:1.45:4.0 oblique:0.6:3.9"   name:pitch:yaw (rad) per screenshot
#      ZOOM=1.0                   scale on the auto-fitted camera distance
#      MAP=/out/pcd/map.pcd       map to show; the PNG is named after it, e.g. color_map.pcd -> rviz_color_map_<view>.png
#      CEIL=2.5                   drop map points higher than this above the trajectory's
#                                 highest pose (the COEX atrium roof hides everything); "none" keeps all
set -uo pipefail
SEQ="${1:-lvi_coex_set_2_mid360_lvi}"
OUT=/out
VIEWS="${VIEWS:-top:1.45:4.0 oblique:0.6:3.9}"
ZOOM="${ZOOM:-1.0}"
CEIL="${CEIL:-2.5}"
MAP="${MAP:-$OUT/pcd/map.pcd}"
TAG=$(basename "$MAP" .pcd)
set +u; source /opt/ros/humble/setup.bash; set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-$((RANDOM % 100 + 100))} ROS_LOCALHOST_ONLY=1
export DISPLAY=:98 LIBGL_ALWAYS_SOFTWARE=1
set -m

Xvfb :98 -screen 0 1920x1080x24 -nolisten tcp >$OUT/xvfb_map.log 2>&1 &
XVFB=$!
for _ in $(seq 1 40); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.25; done

# Fit the camera to the trajectory: centre on it, distance ~ its largest horizontal extent.
read FX FY FZ DIST ZTOP < <(python3 - "$OUT/$SEQ.txt" "$ZOOM" <<'P'
import sys, numpy as np
p = np.loadtxt(sys.argv[1], ndmin=2)[:, 1:4]
c = (p.min(0) + p.max(0)) / 2
ext = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 20.0)
print(*(round(v, 2) for v in c), round(1.3 * ext * float(sys.argv[2]), 1), round(p[:, 2].max(), 2))
P
)
ZARG=()
[ "$CEIL" != none ] && ZARG=(--zmax "$(python3 -c "print($ZTOP + $CEIL)")")
python3 /uamc/scripts/show_map.py "$MAP" $OUT/$SEQ.txt "${ZARG[@]}" >$OUT/show_map.log 2>&1 &
PUB=$!
for V in $VIEWS; do
  IFS=: read NAME PITCH YAW <<<"$V"
  sed -e "s/@FX@/$FX/;s/@FY@/$FY/;s/@FZ@/$FZ/;s/@DIST@/$DIST/;s/@PITCH@/$PITCH/;s/@YAW@/$YAW/" \
    /uamc/config/uamc_map.rviz >/tmp/view_$NAME.rviz
  rviz2 -d /tmp/view_$NAME.rviz >$OUT/rviz_${TAG}_$NAME.log 2>&1 &
  RV=$!
  sleep "${WAIT:-25}"
  import -display :98 -window root "$OUT/rviz_${TAG}_$NAME.png" && echo "wrote rviz_${TAG}_$NAME.png (centre $FX $FY $FZ, distance $DIST, ${ZARG[*]:-no ceiling cut})"
  kill -INT $RV; wait $RV 2>/dev/null
done
kill -INT $PUB; kill $XVFB 2>/dev/null
grep -h "published" $OUT/show_map.log
