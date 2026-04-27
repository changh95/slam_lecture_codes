#!/bin/bash
# Inside-container entrypoint: run Kimera-VIO stereo+IMU on an EuRoC sequence
# with easy_profiler instrumentation, dumping kimera_profile.prof and
# converting it to kimera.json for analyze_profiler.py.
#
# The dataset path must be the parent of mav0/ (i.e. mount the MH_01_easy/
# directory itself, NOT MH_01_easy/mav0/) — Kimera's EurocDataProvider
# walks down to mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}/ on its own.
#
# Visualizer is force-disabled (it needs an X display and crashes otherwise).
#
# Environment overrides:
#   DATASET_PATH    default /data
#   PARAMS_PATH     default /Kimera-VIO/params/Euroc
#   OUTPUT_PROF     default /output/kimera_profile.prof
#   OUTPUT_JSON     default /output/kimera.json
set -e

DATASET_PATH="${DATASET_PATH:-/data}"
PARAMS_PATH="${PARAMS_PATH:-/Kimera-VIO/params/Euroc}"
OUTPUT_PROF="${OUTPUT_PROF:-/output/kimera_profile.prof}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/kimera.json}"

mkdir -p "$(dirname "$OUTPUT_PROF")"

if [ ! -d "$DATASET_PATH/mav0" ]; then
  echo "ERROR: $DATASET_PATH does not contain mav0/ (point at parent of mav0)" >&2
  exit 1
fi

echo "[kimera-profiler] dataset : $DATASET_PATH"
echo "[kimera-profiler] params  : $PARAMS_PATH"
echo "[kimera-profiler] prof    : $OUTPUT_PROF"
echo "[kimera-profiler] json    : $OUTPUT_JSON"

/Kimera-VIO/build/stereoVIOEuroc \
    --flagfile="$PARAMS_PATH/flags/stereoVIOEuroc.flags" \
    --logtostderr=1 \
    --output_path="$(dirname "$OUTPUT_PROF")" \
    --dataset_path="$DATASET_PATH" \
    --dataset_type=0 \
    --params_folder_path="$PARAMS_PATH" \
    --visualize=false \
    --visualize_frontend_images=false

# Kimera's profiler dump is hard-coded to /output/kimera_profile.prof in the
# patched main(); rename only if the caller asked for a different path.
if [ "$OUTPUT_PROF" != "/output/kimera_profile.prof" ] && [ -f /output/kimera_profile.prof ]; then
  mv /output/kimera_profile.prof "$OUTPUT_PROF"
fi

profiler_converter "$OUTPUT_PROF" "$OUTPUT_JSON"
echo "[kimera-profiler] done."
