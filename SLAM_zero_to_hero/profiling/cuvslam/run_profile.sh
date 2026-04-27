#!/bin/bash
# Inside-container entrypoint: run cuVSLAM stereo tracker on a KITTI sequence
# with py_profiler instrumentation, dumping an easy_profiler-compatible JSON.
#
# Environment overrides:
#   SEQUENCE_PATH  default /data  (must contain calib.txt, image_0/, image_1/, times.txt)
#   OUTPUT_JSON    default /output/cuvslam.json
set -e

SEQUENCE_PATH="${SEQUENCE_PATH:-/data}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/cuvslam.json}"

mkdir -p "$(dirname "$OUTPUT_JSON")"

if [ ! -f "$SEQUENCE_PATH/calib.txt" ]; then
  echo "ERROR: $SEQUENCE_PATH does not look like a KITTI sequence (no calib.txt)" >&2
  exit 1
fi

echo "[cuvslam-profiler] sequence: $SEQUENCE_PATH"
echo "[cuvslam-profiler] output  : $OUTPUT_JSON"

python3 /app/track_kitti_profiled.py "$SEQUENCE_PATH" "$OUTPUT_JSON"
