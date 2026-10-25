#!/usr/bin/env bash
# Run cuVSLAM + nvblox over the demo's Humanoid Everyday episodes, one after another.
#
#   run_all.sh [data-root] [output-dir] [extra cuvslam_nvblox.py flags...]
#
# data-root defaults to /data (mount ~/data/humanoid_everyday there); every
# <task>/episode_N directory of the tasks below is processed, in sorted order. Each one gets
# its own tracker, map and rerun recording; results land in
# <output-dir>/<task>/episode_N/ and the table of all of them in <output-dir>/summary.md.
#
# Environment knobs (all optional):
#   TASKS="..."     space-separated task names (default: the three below,
#                   ../download_humanoid_everyday.py's default; "all" = every task on disk)
#   EPISODES=""     space-separated episode names, e.g. "episode_0" (default: all)
set -euo pipefail

ROOT="${1:-/data}"
OUT="${2:-/results/cuvslam}"
if [[ $# -gt 1 ]]; then shift 2; elif [[ $# -gt 0 ]]; then shift 1; fi
TASKS="${TASKS:-walk_towards_chair_and_rotate_the_chair walk_towards_outside_chair_and_pull_it_out walk_towards_trashbin_and_throw_trash_inside}"
[[ "$TASKS" == all ]] && TASKS=""
EPISODES="${EPISODES:-}"

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

list=()
for ep in $(find "$ROOT" -mindepth 2 -maxdepth 2 -type d -name 'episode_*' | sort); do
  task="$(basename "$(dirname "$ep")")"
  [[ -n "$TASKS" && " $TASKS " != *" $task "* ]] && continue
  [[ -n "$EPISODES" && " $EPISODES " != *" $(basename "$ep") "* ]] && continue
  [[ -f "$ep/robot_data.jsonl" || -f "$ep/data.json" ]] || continue
  list+=("$ep")
done
[[ ${#list[@]} -gt 0 ]] || { echo "[run_all] no episodes under $ROOT"; exit 1; }

echo "[run_all] ${#list[@]} episodes -> $OUT"
mkdir -p "$OUT"
/opt/slam_venv/bin/python -u "$SCRIPTS/cuvslam_nvblox.py" "${list[@]}" --out "$OUT" "$@" \
  2>&1 | tee "$OUT/run_all.log"
