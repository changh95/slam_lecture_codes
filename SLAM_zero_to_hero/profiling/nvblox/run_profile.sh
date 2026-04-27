#!/bin/bash
# Inside-container entrypoint: run fuse_replica against a Replica scene and
# convert nvblox's native timing report into the shared analyze_profiler CSV.
#
# Environment overrides:
#   DATASET_PATH       default /data/Replica/office0
#   NUM_FRAMES         default 500 (negative = all)
#   VOXEL_SIZE         default 0.05 (metres, only used if --voxel_size flag
#                      is propagated; nvblox picks from mapper config by default)
#   OUTPUT_TIMING      default /output/nvblox_timing.txt
#   OUTPUT_MESH        default /output/nvblox_mesh.ply
#   OUTPUT_CSV         default /output/nvblox_stats.csv
set -e

DATASET_PATH="${DATASET_PATH:-/data/Replica/office0}"
NUM_FRAMES="${NUM_FRAMES:-500}"
OUTPUT_TIMING="${OUTPUT_TIMING:-/output/nvblox_timing.txt}"
OUTPUT_MESH="${OUTPUT_MESH:-/output/nvblox_mesh.ply}"
OUTPUT_CSV="${OUTPUT_CSV:-/output/nvblox_stats.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/nvblox.json}"

mkdir -p "$(dirname "$OUTPUT_TIMING")"

if [ ! -d "$DATASET_PATH" ]; then
  echo "ERROR: dataset not found at $DATASET_PATH" >&2
  echo "Mount a Replica scene at /data/Replica/<scene> or set DATASET_PATH" >&2
  exit 1
fi

echo "[nvblox-profiler] dataset   : $DATASET_PATH"
echo "[nvblox-profiler] num_frames: $NUM_FRAMES"
echo "[nvblox-profiler] timing    : $OUTPUT_TIMING"
echo "[nvblox-profiler] csv       : $OUTPUT_CSV"

/usr/local/bin/nvblox/fuse_replica "$DATASET_PATH" \
  --num_frames="$NUM_FRAMES" \
  --timing_output_path="$OUTPUT_TIMING" \
  --mesh_output_path="$OUTPUT_MESH"

python3 /app/parse_nvblox_timing.py "$OUTPUT_TIMING" -o "$OUTPUT_CSV"
python3 /app/nvblox_timing_to_json.py "$OUTPUT_TIMING" -o "$OUTPUT_JSON"
echo "[nvblox-profiler] done."
