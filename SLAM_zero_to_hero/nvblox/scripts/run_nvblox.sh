#!/usr/bin/env bash
# Convert one Humanoid Everyday episode and fuse it with nvblox, end to end.
#
#   run_nvblox.sh <episode-dir> [output-dir] [extra fuse_replica flags...]
#
# Environment knobs (all optional):
#   VOXEL=0.015      TSDF voxel size in metres
#   POSES=icp        pose source: icp (depth frame-to-model ICP) or odom
#   FRAMES=0         number of frames to convert, 0 = whole episode
#   STRIDE=1         take every Nth frame
#   START=0          first frame index
#   ROBOT=g1         g1 or h1, picks the intrinsics
#   FRAME=colour     camera frame for the RGB-D pair: colour or depth
#   WORLD=gravity    world frame: gravity (IMU-anchored, z up) or camera
#   DEPTH_MAX=6.0    drop depth beyond this many metres
#   MAX_INT=5.0      nvblox max integration distance in metres
#   RECONVERT=0      set to 1 to redo the conversion even if it is cached
set -euo pipefail

EPISODE="${1:?usage: run_nvblox.sh <episode-dir> [output-dir] [fuse_replica flags...]}"
EPISODE="${EPISODE%/}"
OUT="${2:-/results/$(basename "$(dirname "$EPISODE")")_$(basename "$EPISODE")}"
if [[ $# -gt 1 ]]; then shift 2; else shift 1; fi

VOXEL="${VOXEL:-0.015}"
POSES="${POSES:-icp}"
FRAMES="${FRAMES:-0}"
STRIDE="${STRIDE:-1}"
START="${START:-0}"
ROBOT="${ROBOT:-g1}"
FRAME="${FRAME:-colour}"
WORLD="${WORLD:-gravity}"
DEPTH_MAX="${DEPTH_MAX:-6.0}"
MAX_INT="${MAX_INT:-5.0}"
RECONVERT="${RECONVERT:-0}"

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$OUT/dataset"

mkdir -p "$OUT"

if [[ "$RECONVERT" == "1" || ! -f "$DATASET/seq/traj.txt" ]]; then
  rm -rf "$DATASET"
  python3 "$SCRIPTS/he_to_nvblox.py" "$EPISODE" "$DATASET" \
      --robot "$ROBOT" --poses "$POSES" --frame "$FRAME" --world "$WORLD" \
      --start "$START" --frames "$FRAMES" --stride "$STRIDE" \
      --depth-max "$DEPTH_MAX" 2>&1 | tee "$OUT/convert.log"
else
  echo "[run] reusing cached dataset at $DATASET (RECONVERT=1 to redo)"
fi

echo "[run] fusing with nvblox, voxel=${VOXEL} m"
# Ground plane estimation is what makes ground_aligned_mesh useful: it RANSACs a
# floor out of the TSDF and rotates the mesh so that floor lands on z=0. Without
# it the map is upright only by accident, because the world frame is the first
# camera pose and that camera is pitched 56 deg into the floor.
fuse_replica "$DATASET/seq" \
    --voxel_size="$VOXEL" \
    --projective_integrator_max_integration_distance_m="$MAX_INT" \
    --mesh_output_path="$OUT/mesh.ply" \
    --ground_aligned_mesh_output_path="$OUT/mesh_ground_aligned.ply" \
    --ground_plane_output_path="$OUT/ground_plane.yaml" \
    --esdf_output_path="$OUT/esdf.ply" \
    --tsdf_output_path="$OUT/tsdf.ply" \
    --map_output_path="$OUT/map.nvblx" \
    --timing_output_path="$OUT/timings.txt" \
    --experimental_use_ground_plane_estimation=true \
    "$@" 2>&1 | tee "$OUT/nvblox.log"

echo
echo "[run] outputs in $OUT:"
ls -la "$OUT" | sed 's/^/    /'
echo
echo "[run] nvblox timings (per-call means, from timings.txt):"
grep -E "integrate|mesh|esdf|FullRun" "$OUT/timings.txt" 2>/dev/null | head -20 | sed 's/^/    /' || true
