#!/bin/bash
# Run MASt3R-SLAM with its own OpenGL viewer on a virtual X display (Xvfb +
# Mesa llvmpipe), screenshot the viewer while it runs and once SLAM is done,
# then score the keyframe trajectory against groundtruth.txt with evo.
#
# usage: run_headless_viz.sh <dataset_dir> [config] [out_dir]
#   e.g. run_headless_viz.sh datasets/tum/rgbd_dataset_freiburg1_room config/calib.yaml /out
set -u
DS=${1:?dataset dir}
CFG=${2:-config/calib.yaml}
OUT=${3:-/out}
SHOT_EVERY=${SHOT_EVERY:-10}   # seconds between progress screenshots
MAX_SHOTS=${MAX_SHOTS:-12}
SEQ=$(basename "${DS%/}")
mkdir -p "$OUT"
cd /MASt3R-SLAM
export PYTHONUNBUFFERED=1       # otherwise "done" sits in the stdout buffer until exit

Xvfb :99 -screen 0 1960x1080x24 +extension GLX >/dev/null 2>&1 &
XVFB=$!
export DISPLAY=:99
sleep 2

t0=$(date +%s)
python main.py --dataset "$DS" --config "$CFG" --save-as "$OUT" >"$OUT/run.log" 2>&1 &
PID=$!

n=0; last=$t0
while kill -0 $PID 2>/dev/null && ! grep -q '^done$' "$OUT/run.log"; do
    sleep 1
    now=$(date +%s)
    if [ $((now - last)) -ge "$SHOT_EVERY" ] && [ $n -lt "$MAX_SHOTS" ]; then
        n=$((n + 1)); last=$now
        import -window root "$OUT/viewer_progress_$(printf %02d $n).png" 2>/dev/null
    fi
done
t1=$(date +%s)
echo "wall time until 'done': $((t1 - t0)) s (includes ~model load and result saving)" | tee -a "$OUT/run.log"
grep '^FPS' "$OUT/run.log" | tail -1

# SLAM finished; the viewer keeps the final map up until its window closes
sleep 8
import -window root "$OUT/mast3r_viewer_${SEQ}.png"
kill $PID 2>/dev/null; sleep 2; pkill -f main.py 2>/dev/null; kill $XVFB 2>/dev/null

if [ -f "$OUT/$SEQ.txt" ]; then
    python - "$OUT/$SEQ.txt" <<'EOF'
import sys, numpy as np
t = np.loadtxt(sys.argv[1])
print(f"keyframes: {len(t)}, keyframe path length (unscaled): "
      f"{np.linalg.norm(np.diff(t[:, 1:4], axis=0), axis=1).sum():.3f}")
EOF
fi
if [ -f "$DS/groundtruth.txt" ] && [ -f "$OUT/$SEQ.txt" ]; then
    evo_ape tum "$DS/groundtruth.txt" "$OUT/$SEQ.txt" -as | tee "$OUT/ate_${SEQ}.txt"
fi
ls -la "$OUT"
