#!/usr/bin/env bash
# Run Cerberus 2.0 on one recorded Go1 sequence, inside the container.
#
# Mounts expected:
#   /data   dataset dir, read-only   (host: ~/data/cerberus2)
#   /out    writable results dir     (host: cerberus_2/results/<run>)
#
# Env:
#   BAG=/data/mill19_trail/230628-mil19-trot-07-039-wild1445.bag
#   CONFIG=<pkg>/config/lecture/mill19_trail.yaml
#   START=          seconds to skip at the start of the bag (rosbag play -s)
#   DURATION=       seconds of bag to play; empty = whole bag
#   RATE=1.0        rosbag play rate
#   RVIZ=false      true => rviz on the host X display, plus screenshots
#   RVIZ_CONFIG=    alternative .rviz file (default: config/rviz/lecture/cerberus2_vilo.rviz)
#   RVIZ_DISTANCE=  orbit camera distance in metres; the shipped config is framed
#                   for a ~100 m outdoor path, so a 3 m indoor square needs ~8
#   RVIZ_FOCAL=     orbit focal point, "x y z"; the shipped one is offset to suit
#                   a long outdoor path, an indoor run wants "0 0 0"
#   RVIZ_PITCH=     orbit pitch, radians (0 = horizon, 1.57 = straight down)
#   SHOT_AT=90      seconds into playback to take the mid-run screenshot
#   FUSION_TYPE=    override vilo_fusion_type: 0 = VIO + MIPO baselines,
#                   1 = fuse leg-odometry velocity (Cerberus 2.0 proper),
#                   2 = tightly-coupled leg factor
#   KF_TYPE=        override kf_type: 0 = MIPO (multi-IMU), 1 = SIPO (single IMU)
#   OVERRIDES=      comma-separated scalar yaml overrides, e.g.
#                   "estimate_extrinsic=0,init_base_height=0.05". Escape hatch for
#                   any single-value field in the sequence config.
set -uo pipefail

CATKIN_WS=/home/EstimationUser/estimation_ws
BAG="${BAG:-/data/mill19_trail/230628-mil19-trot-07-039-wild1445.bag}"
CONFIG="${CONFIG:-$CATKIN_WS/src/cerberus2/config/lecture/mill19_trail.yaml}"
START="${START:-}"
DURATION="${DURATION:-}"
RATE="${RATE:-1.0}"
RVIZ="${RVIZ:-false}"
SHOT_AT="${SHOT_AT:-90}"
RVIZ_DISTANCE="${RVIZ_DISTANCE:-}"
RVIZ_FOCAL="${RVIZ_FOCAL:-}"
RVIZ_PITCH="${RVIZ_PITCH:-}"
FUSION_TYPE="${FUSION_TYPE:-}"
KF_TYPE="${KF_TYPE:-}"
OVERRIDES="${OVERRIDES:-}"
OUT=/out

# Export before sourcing setup.bash: ROS's own profile.d/10.roslaunch.sh reads
# $ROS_MASTER_URI, and under `set -u` an unset one aborts the script before
# anything runs. Only bites when ros_entrypoint.sh is bypassed, which is exactly
# what happens on `podman run ... bash /opt/cerberus2_demo/run_demo.sh`.
export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1

source /opt/ros/noetic/setup.bash
source $CATKIN_WS/devel/setup.bash

mkdir -p "$OUT"
test -f "$BAG" || { echo "[run] no such bag: $BAG"; exit 1; }
test -f "$CONFIG" || { echo "[run] no such config: $CONFIG"; exit 1; }

# Estimator-variant override. parameters.cpp reads kf_type / vilo_fusion_type
# from the yaml only -- there is no rosparam for either -- so switching variant
# means editing a copy of the config. The copy has to sit in a directory that
# also holds the two camera calib files, because parameters.cpp resolves
# cam0_calib/cam1_calib relative to the config file's own directory. This is the
# same trick upstream's launch/vilo_autotest.sh uses with /tmp/temp_run.yaml.
if [ -n "$FUSION_TYPE" ] || [ -n "$KF_TYPE" ] || [ -n "$OVERRIDES" ]; then
    RUNDIR=/tmp/cerberus2_run
    mkdir -p "$RUNDIR"
    cp "$(dirname "$CONFIG")"/go1_realsense_*.yaml "$RUNDIR"/
    cp "$CONFIG" "$RUNDIR/run.yaml"
    [ -n "$FUSION_TYPE" ] && sed -i "s/^vilo_fusion_type: *[0-9]/vilo_fusion_type: $FUSION_TYPE/" "$RUNDIR/run.yaml"
    [ -n "$KF_TYPE" ] && sed -i "s/^kf_type: *[0-9]/kf_type: $KF_TYPE/" "$RUNDIR/run.yaml"
    if [ -n "$OVERRIDES" ]; then
        IFS=',' read -ra KVS <<< "$OVERRIDES"
        for kv in "${KVS[@]}"; do
            k=${kv%%=*}; v=${kv#*=}
            grep -qE "^${k}: " "$RUNDIR/run.yaml" || { echo "[run] no such config key: $k"; exit 1; }
            sed -i -E "s|^(${k}: *)[-0-9.]+|\1${v}|" "$RUNDIR/run.yaml"
        done
    fi
    CONFIG="$RUNDIR/run.yaml"
    echo "[run] config overrides:"
    grep -E '^(kf_type|vilo_fusion_type|init_base_height|estimate_extrinsic|estimate_td|estimate_kinematic|td):' \
        "$RUNDIR/run.yaml" | sed 's/ *#.*//;s/^/[run]   /'
fi

echo "[run] bag    : $BAG"
echo "[run] config : $CONFIG"
echo "[run] rviz   : $RVIZ   rate: $RATE   duration: ${DURATION:-whole bag}"

roscore >"$OUT/roscore.log" 2>&1 &
ROSCORE_PID=$!
until rostopic list >/dev/null 2>&1; do sleep 0.5; done
echo "[run] roscore up"

roslaunch cerberus2 cerberus2_bag.launch \
    config:="$CONFIG" --wait >"$OUT/cerberus2.log" 2>&1 &
LAUNCH_PID=$!

# cerberus2_main builds its casadi symbolic Jacobians in the MIPOEstimator
# constructor before it subscribes to anything. Playing the bag into it before
# then throws away the initialisation window the estimator needs.
echo "[run] waiting for the estimator to advertise"
for _ in $(seq 1 300); do
    rostopic list 2>/dev/null | grep -q '^/vilo/estimate_pose$' && break
    kill -0 $LAUNCH_PID 2>/dev/null || { echo "[run] launch died early"; tail -40 "$OUT/cerberus2.log"; exit 1; }
    sleep 1
done
rostopic list 2>/dev/null | grep -q '^/vilo/estimate_pose$' || {
    echo "[run] estimator never advertised /vilo/estimate_pose"; tail -40 "$OUT/cerberus2.log"; exit 1; }
sleep 3
echo "[run] estimator up"

# --hz=2000: both estimator loops poll ros::Time::now() (the PO loop targets
# 400 Hz) and divide by elapsed simulated time. rosbag play's default 100 Hz
# /clock quantises that to 10 ms steps, so most iterations see dt_ros == 0 and
# `continue` out. Upstream's own launch files pass --hz=2000 for this reason.
# --queue=1000 matches the estimator's subscriber queues; the default 100 drops
# messages on the 200 Hz IMU topics.
PLAY_ARGS=(--clock --hz=2000 --queue=1000 --quiet -r "$RATE")
[ -n "$START" ] && PLAY_ARGS+=(-s "$START")
[ -n "$DURATION" ] && PLAY_ARGS+=(-u "$DURATION")
PLAY_ARGS+=("$BAG")

rviz_window() {
    # rviz creates a dozen X windows; the one to capture is the top-level frame,
    # whose title is "<config>.rviz - RViz". The bare "rviz" ones are child
    # widgets and capturing those yields a slice of the toolbar.
    xdotool search --all --name ' - RViz$' 2>/dev/null | head -1
}

grab_rviz() {
    local dest="$1" wid
    wid=$(rviz_window)
    [ -n "$wid" ] && import -window "$wid" "$dest" 2>/dev/null && echo "[run] screenshot $dest"
}

RVIZ_PID=""
if [ "$RVIZ" = "true" ]; then
    # Qt writes its socket into XDG_RUNTIME_DIR; if the directory does not exist
    # rviz still starts but logs a QStandardPaths warning on every run.
    mkdir -p "${XDG_RUNTIME_DIR:-/tmp/runtime-root}" && chmod 700 "${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
    RVIZ_CONFIG="${RVIZ_CONFIG:-$CATKIN_WS/src/cerberus2/config/rviz/lecture/cerberus2_vilo.rviz}"
    if [ -n "$RVIZ_DISTANCE$RVIZ_FOCAL$RVIZ_PITCH" ]; then
        # rviz has no command-line way to set the orbit view, and the saved one is
        # framed for a ~100 m outdoor path -- point it at an indoor 3 m square and
        # you get an empty grid cell. Edit a copy. "Distance:"/"Pitch:" appear only
        # in the Views block, and "Focal Point:" only once, so these are unambiguous.
        cp "$RVIZ_CONFIG" /tmp/run.rviz
        [ -n "$RVIZ_DISTANCE" ] && sed -i -E "s|^( +Distance: ).*|\\1$RVIZ_DISTANCE|" /tmp/run.rviz
        [ -n "$RVIZ_PITCH" ] && sed -i -E "s|^( +Pitch: ).*|\\1$RVIZ_PITCH|" /tmp/run.rviz
        if [ -n "$RVIZ_FOCAL" ]; then
            read -r fx fy fz <<< "$RVIZ_FOCAL"
            sed -i "/Focal Point:/{n;s/X: .*/X: ${fx:-0}/;n;s/Y: .*/Y: ${fy:-0}/;n;s/Z: .*/Z: ${fz:-0}/}" /tmp/run.rviz
        fi
        RVIZ_CONFIG=/tmp/run.rviz
        echo "[run] rviz view -> distance=${RVIZ_DISTANCE:-cfg} focal=${RVIZ_FOCAL:-cfg} pitch=${RVIZ_PITCH:-cfg}"
    fi
    echo "[run] rviz -d $RVIZ_CONFIG"
    rviz -d "$RVIZ_CONFIG" >"$OUT/rviz.log" 2>&1 &
    RVIZ_PID=$!
    for _ in $(seq 1 60); do
        [ -n "$(rviz_window)" ] && break
        sleep 1
    done
    if [ -n "$(rviz_window)" ]; then
        echo "[run] rviz window up: $(xdotool getwindowname "$(rviz_window)" 2>/dev/null)"
    else
        echo "[run] WARNING: rviz never mapped a window; see $OUT/rviz.log"
    fi
    # Grab SHOT_AT seconds in, so the screenshot shows a trajectory in progress
    # rather than an empty grid. Must be less than the playback length.
    ( sleep "$SHOT_AT"; grab_rviz "$OUT/rviz_mid.png" ) &
fi

echo "[run] rosbag play ${PLAY_ARGS[*]}"
SECONDS=0
rosbag play "${PLAY_ARGS[@]}" >"$OUT/rosbag_play.log" 2>&1
echo "[run] playback wall-clock: ${SECONDS} s"

echo "[run] draining 10 s"
sleep 10

[ "$RVIZ" = "true" ] && grab_rviz "$OUT/rviz_final.png"

echo "[run] SIGINT -> nodes"
kill -INT $LAUNCH_PID 2>/dev/null
sleep 10
kill -KILL $LAUNCH_PID 2>/dev/null
[ -n "$RVIZ_PID" ] && kill -INT $RVIZ_PID 2>/dev/null
kill -INT $ROSCORE_PID 2>/dev/null
sleep 2

echo "[run] results in $OUT:"
ls -la "$OUT"

# Which CSVs appear depends on the variant, because parameters.cpp derives the
# file names from kf_type/vilo_fusion_type:
#   vilo_fusion_type 1, kf_type 0  ->  vilo-m-<dataset>.csv
#   vilo_fusion_type 0             ->  vio-<dataset>.csv AND mipo|sipo-<dataset>.csv
# The CSVs are appended line-by-line during the run, so they are complete even if
# the estimator had to be killed.
# gt-*.csv is included when non-empty: plot_trajectory.py then rigidly aligns
# each estimate to it and prints an ATE. Outdoor sequences carry no pose topic, so
# there the file exists but is zero bytes and drops out here.
mapfile -t CSVS < <(ls "$OUT"/vilo-*.csv "$OUT"/vio-*.csv "$OUT"/mipo-*.csv "$OUT"/sipo-*.csv "$OUT"/gt-*.csv 2>/dev/null | while read -r f; do [ -s "$f" ] && echo "$f"; done)
if [ "${#CSVS[@]}" -eq 0 ]; then
    echo "[run] no non-empty trajectory CSV produced -- see $OUT/cerberus2.log"
    exit 1
fi
echo "[run] plot_trajectory.py ${CSVS[*]}"
python3 /opt/cerberus2_demo/plot_trajectory.py --out "$OUT/trajectory.png" "${CSVS[@]}"
