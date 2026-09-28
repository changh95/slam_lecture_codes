#!/bin/bash
# ConceptFusion end to end: SAM masks + OpenCLIP pixel-aligned features -> gradslam
# PointFusion map -> text queries.
#   /data     ICL-NUIM root (holds living_room_traj2_frei_png/)
#   /weights  checkpoints from scripts/download_weights.sh; intermediates go to /weights/run
#   /out      PNG renders + query_summary.json
set -euo pipefail
SEQ=${SEQ:-living_room_traj2_frei_png}
END=${END:-880}          # livingRoom2n.gt.sim has 880 poses for 881 frames
STRIDE=${STRIDE:-20}
QUERIES=${QUERIES:-"sofa table"}
OUT=${OUT:-/out}
WORK=${WORK:-/weights/run}   # saved-feat (~2 GB), saved-map (~1 GB), PLYs, .rrd
cd /opt/concept-fusion/examples
mkdir -p "$OUT" "$WORK"

t0=$(date +%s)
python extract_conceptfusion_features.py \
  --checkpoint-path "$SAM_CKPT" --data-dir /data --sequence "$SEQ" \
  --end-idx "$END" --stride "$STRIDE" --save-dir "$WORK/saved-feat"
t1=$(date +%s)
python run_feature_fusion_and_save_map.py \
  --dataset-path /data --sequence "$SEQ" --frame-end "$END" --stride "$STRIDE" \
  --feat-dir "$WORK/saved-feat" --dir-to-save-map "$WORK/saved-map"
t2=$(date +%s)
# shellcheck disable=SC2086
/opt/cf_tools/query.sh --load-path "$WORK/saved-map" --out-dir "$OUT" --work-dir "$WORK" \
  --queries $QUERIES --feat-dir "$WORK/saved-feat" --data-dir /data --sequence "$SEQ" \
  --frame-end "$END" --stride "$STRIDE" "$@"
t3=$(date +%s)
echo "Timing: features $((t1-t0)) s, fusion $((t2-t1)) s, query+render $((t3-t2)) s, total $((t3-t0)) s"
