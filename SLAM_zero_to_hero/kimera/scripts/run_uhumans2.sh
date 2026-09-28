#!/bin/bash
# Kimera-Semantics on one uHumans2 bag, end to end, inside the container:
# roscore -> depth_image_proc + kimera_semantics_node -> RViz -> rosbag play
# -> save the mesh (PLY) -> RViz screenshot -> mesh/run statistics.
#
#   run_uhumans2.sh [bag] [rate]
#   env: OUT=/out            output directory (PLY, PNG, logs, stats)
#        HEADLESS=1          RViz on a private Xvfb, screenshot taken at the end
#                            (default when $DISPLAY is empty); HEADLESS=0 uses $DISPLAY
#        RVIZ=0              no RViz at all
#        CSV=...             label csv (default: /kimera/config/uhumans2_office_...csv
#                            for an office bag, archviz1 for an apartment bag)
set -e
# default: the LZ4 copy of the office bag (plays in real time), else the bz2 original
BAG=$1
if [ -z "$BAG" ]; then
  for BAG in /data/uHumans2_office_s1_00h_lz4.bag /data/uHumans2_office_s1_00h.bag; do
    [ -f "$BAG" ] && break
  done
fi
RATE=${2:-1.0}
OUT=${OUT:-/out}
RVIZ=${RVIZ:-1}
case "$(basename "$BAG")" in
  *apartment*) CSV=${CSV:-$(rospack find kimera_semantics_ros)/cfg/tesse_multiscene_archviz1_segmentation_mapping.csv}
               RVIZ_CFG=kimera_semantics_uhumans2_apartment.rviz ;;
  *)           CSV=${CSV:-/kimera/config/uhumans2_office_segmentation_mapping.csv}
               RVIZ_CFG=kimera_semantics_uhumans2.rviz ;;
esac
if [ -z "$DISPLAY" ]; then HEADLESS=1; fi
HEADLESS=${HEADLESS:-0}
mkdir -p "$OUT"
[ -f "$BAG" ] || { echo "bag not found: $BAG"; exit 1; }

cleanup() { kill $(jobs -p) 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

roscore > "$OUT/roscore.log" 2>&1 &
until rostopic list > /dev/null 2>&1; do sleep 0.5; done

roslaunch /kimera/launch/kimera_semantics_uhumans2.launch \
  mesh_filename:="$OUT/kimera_semantics_mesh.ply" \
  semantic_label_2_color_csv_filepath:="$CSV" > "$OUT/kimera_semantics.log" 2>&1 &
python3 /kimera/scripts/monitor.py "$OUT/run_stats.txt" > /dev/null 2>&1 &
MON=$!

if [ "$RVIZ" = 1 ]; then
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
  echo "[run] RViz GL: $(glxinfo -B 2>/dev/null | grep 'renderer string' | cut -d: -f2)" | tee "$OUT/rviz_gl.txt"
  rviz -d /kimera/config/$RVIZ_CFG > "$OUT/rviz.log" 2>&1 &
fi
until rosservice list 2>/dev/null | grep -q /kimera_semantics_node/generate_mesh; do sleep 0.5; done
sleep 3

echo "[run] playing $(basename "$BAG") at ${RATE}x"
T0=$(date +%s)
rosbag play --clock -q -r "$RATE" "$BAG"
T1=$(date +%s)
sleep 3   # let the last clouds integrate and one more mesh update go out

rosservice call /kimera_semantics_node/generate_mesh > /dev/null
if [ "$RVIZ" = 1 ] && [ "$HEADLESS" = 1 ]; then
  sleep 5
  import -window root "$OUT/rviz_semantic_mesh.png"
  echo "[run] screenshot: $OUT/rviz_semantic_mesh.png"
fi
kill -INT $MON 2>/dev/null || true; sleep 2   # monitor.py writes run_stats.txt on exit
echo "wall_clock_play_s $((T1 - T0))" >> "$OUT/run_stats.txt"
echo "---- run stats ----"; cat "$OUT/run_stats.txt"
echo "---- mesh ----"; python3 /kimera/scripts/mesh_stats.py "$OUT/kimera_semantics_mesh.ply" "$CSV" | tee "$OUT/mesh_stats.txt"
if [ "$RVIZ" = 1 ] && [ "$HEADLESS" = 0 ]; then
  echo "[run] bag finished; RViz stays open -- Ctrl-C to quit"; wait
fi
