#!/usr/bin/env bash
# Drive rviz/rviz2 on a private Xvfb with real (XTEST) mouse input and check
# what each gesture did to the saved view.
#
#   mouse_test.sh <ViewClass> [out_dir]
#     ViewClass  e.g. slam_zero_to_hero/UnifiedOrbit, rviz/Orbit,
#                rviz_default_plugins/Orbit, slam_zero_to_hero/UnifiedTopDownOrtho
#
# After every gesture the config is saved with Ctrl+S and the Views/Current
# block is read back, so the check is on the controller's own numbers
# (Distance, Focal Point, Yaw, Pitch / Scale, X, Y, Angle), not on pixels.
# Exit status 0 = every gesture did what the unified scheme promises for
# slam_zero_to_hero/* classes, or what stock RViz does for the others.
CLASS=${1:?usage: mouse_test.sh <ViewClass> [out_dir]}
OUT=${2:-/tmp/mouse_test}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -d /opt/ros/noetic ]; then
  ROS=1; source /opt/ros/noetic/setup.bash; RVIZ=rviz
  export DISABLE_ROS1_EOL_WARNINGS=1   # otherwise a modal dialog covers the view
else
  ROS=2; source /opt/ros/${ROS_DISTRO:-$(ls /opt/ros | head -1)}/setup.bash; RVIZ=rviz2
fi
export LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1

PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

Xvfb :99 -screen 0 1280x800x24 -nolisten tcp >/dev/null 2>&1 & PIDS+=($!)
export DISPLAY=:99
sleep 1

if [ $ROS = 1 ]; then
  roscore >"$OUT/roscore.log" 2>&1 & PIDS+=($!)
  sleep 4
  rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link >/dev/null 2>&1 & PIDS+=($!)
else
  ros2 run tf2_ros static_transform_publisher --frame-id map --child-frame-id base_link \
    >/dev/null 2>&1 & PIDS+=($!)
fi

CFG="$OUT/test.rviz"
python3 "$HERE/make_config.py" "$ROS" "$CLASS" >"$CFG"
$RVIZ -d "$CFG" >"$OUT/rviz.log" 2>&1 & PIDS+=($!)
for _ in $(seq 60); do
  xdotool search --name "RViz" >/dev/null 2>&1 && break
  sleep 1
done
sleep 6   # let the first frames render and the view settle

CX=640; CY=430   # inside the render panel (docks are hidden in make_config.py)
STEP=0
save() {
  # No window manager: keyboard focus follows the pointer.
  xdotool mousemove $CX $CY; sleep 0.3
  xdotool key ctrl+s; sleep 1.5
  STEP=$((STEP + 1))
  cp "$CFG" "$OUT/view_${STEP}_$1.rviz"
  python3 "$HERE/read_view.py" "$CFG" "$1" >>"$OUT/views.jsonl"
  import -window root "$OUT/shot_${STEP}_$1.png" 2>/dev/null
}
drag() {  # drag <button> <dx> <dy>, in 20 small steps so Qt sees motion events
  xdotool mousemove $CX $CY; sleep 0.2
  xdotool mousedown "$1"; sleep 0.2
  for _ in $(seq 20); do xdotool mousemove_relative -- $(($2 / 20)) $(($3 / 20)); sleep 0.03; done
  sleep 0.2; xdotool mouseup "$1"; sleep 0.5
}

: >"$OUT/views.jsonl"
save start
drag 3 120 80;  save right_drag
xdotool mousemove $CX $CY click --repeat 3 --delay 150 4; sleep 0.5; save wheel_up
drag 1 120 0;   save left_drag
drag 2 -120 -80; save middle_drag

grep -iE "error|fatal|failed to load|PluginlibFactory" "$OUT/rviz.log" | grep -v "Stereo is NOT SUPPORTED" >"$OUT/rviz_errors.log"
python3 "$HERE/check_views.py" "$CLASS" "$OUT/views.jsonl" "$OUT/rviz_errors.log"
