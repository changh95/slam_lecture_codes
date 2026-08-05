#!/usr/bin/env bash
# Run SVO Pro on a UZH-FPV bag end to end: launch the pipeline, record the
# estimated trajectory, play the bag, then evaluate against ground truth.
#
#   run_fpv.sh /data/indoor_forward_3_snapdragon_with_gt.bag [stereo|mono] [--headless]
set -euo pipefail

BAG=${1:?usage: run_fpv.sh <bag> [stereo|mono] [--headless]}
MODE=${2:-stereo}
shift || true; shift || true

RVIZ=true
RATE=${RATE:-1.0}
START=${START:-0}
for a in "$@"; do
  case "$a" in
    --headless) RVIZ=false ;;
  esac
done

[ -f "$BAG" ] || { echo "no such bag: $BAG" >&2; exit 1; }
case "$MODE" in stereo|mono) ;; *) echo "mode must be stereo or mono" >&2; exit 1 ;; esac

OUT=${OUT:-/results/$(basename "${BAG%.bag}")_$MODE}
mkdir -p "$OUT"
echo "=== SVO Pro $MODE on $(basename "$BAG") -> $OUT ==="

# The `--` keeps this script's own positional arguments (the bag path, mode and
# flags) from being parsed as ROS setup options. See scripts/entrypoint.sh.
source /opt/ros/noetic/setup.bash --
source /svo_ws/devel/setup.bash --

# roscore first, so rosbag record and the launch file share one master.
if ! rostopic list >/dev/null 2>&1; then
  roscore >"$OUT/roscore.log" 2>&1 &
  ROSCORE_PID=$!
  until rostopic list >/dev/null 2>&1; do sleep 0.5; done
fi

cleanup() {
  set +e
  [ -n "${PLAY_PID:-}"   ] && kill "$PLAY_PID"   2>/dev/null
  [ -n "${REC_PID:-}"    ] && kill -INT "$REC_PID" 2>/dev/null
  [ -n "${SVO_PID:-}"    ] && kill "$SVO_PID"    2>/dev/null
  sleep 2
  [ -n "${ROSCORE_PID:-}" ] && kill "$ROSCORE_PID" 2>/dev/null
}
trap cleanup EXIT

roslaunch svo_ros "fpv_vio_${MODE}.launch" rviz:="$RVIZ" \
  >"$OUT/svo.log" 2>&1 &
SVO_PID=$!

# Wait for the node to advertise its pose topic before playing anything,
# otherwise the first seconds of the sequence are dropped.
echo "waiting for /svo/pose_imu ..."
for _ in $(seq 1 120); do
  rostopic list 2>/dev/null | grep -q "^/svo/pose_imu$" && break
  sleep 0.5
done
rostopic list 2>/dev/null | grep -q "^/svo/pose_imu$" \
  || { echo "svo_node never came up; see $OUT/svo.log" >&2; tail -40 "$OUT/svo.log" >&2; exit 1; }

rosbag record -O "$OUT/svo_traj.bag" /svo/pose_imu >"$OUT/record.log" 2>&1 &
REC_PID=$!
sleep 2

echo "playing bag (rate=$RATE, start=$START) ..."
rosbag play --clock -r "$RATE" -s "$START" "$BAG" >"$OUT/play.log" 2>&1 &
PLAY_PID=$!
wait "$PLAY_PID" || true
PLAY_PID=""

sleep 3
kill -INT "$REC_PID" 2>/dev/null || true
wait "$REC_PID" 2>/dev/null || true
REC_PID=""

echo "=== evaluating ==="
EXTRA=()
[ "$MODE" = "mono" ] && EXTRA+=(--correct_scale)
python3 /svo_ws/scripts/eval_fpv.py \
  --gt-bag "$BAG" --est-bag "$OUT/svo_traj.bag" --out "$OUT" "${EXTRA[@]}"

echo "=== done: $OUT ==="
