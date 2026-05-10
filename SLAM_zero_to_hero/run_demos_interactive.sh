#!/bin/bash
# Walk through every part1/2/3 demo with X11 forwarded.
# Press Enter to advance. Press 's' to skip the current demo.
# Press 'q' to quit.
set -u

# Make sure X server accepts connections from this user
xhost +local:root 2>/dev/null
xhost +local:    2>/dev/null

# X11 forwarding flags shared by all containers
X11_FLAGS=(
  --net=host
  -e DISPLAY=${DISPLAY:-:0}
  -e XAUTHORITY=${XAUTHORITY:-}
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro
)
if [ -n "${XAUTHORITY:-}" ] && [ -f "$XAUTHORITY" ]; then
  X11_FLAGS+=( -v "$XAUTHORITY:$XAUTHORITY:ro" )
fi

# (demo:::run-command-inside-container:::label)
# Separator is `:::` (three colons) so single `|` inside command pipes is fine.
DEMOS=(
  "part1_ch02_02:::cd /workspace/part1_ch02_02/build && for b in for_loop while_loop vector map unordered_map template_function template_class smart_pointers; do echo '== '\$b' =='; ./\$b 2>&1 | head -20; done:::Part1 ch02_02 - Basic C++ programming"
  "part1_ch02_03:::cd /workspace/part1_ch02_03/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -20; fi; done:::Part1 ch02_03 - Modern C++"
  "part1_ch02_04:::cd /workspace/part1_ch02_04/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -10; fi; done:::Part1 ch02_04 - Profiling (easy_profiler)"
  "part1_ch02_05:::cd /workspace/part1_ch02_05/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -10; fi; done:::Part1 ch02_05 - Memory tooling (Valgrind)"
  "part1_ch02_08:::cd /workspace/part1_ch02_08/build && python3 -c 'import sys; sys.path.insert(0, \".\"); import slam_bindings as m; print(\"Module:\", m); print(\"Members:\", [x for x in dir(m) if not x.startswith(\"_\")])' 2>&1:::Part1 ch02_08 - nanobind Python bindings"
  "part1_ch03_05:::cd /workspace/part1_ch03_05/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -8; fi; done:::Part1 ch03_05 - Math/Linear algebra"
  "part1_ch04_08:::echo '(ROS noetic image - skipping interactive run; image builds and contains ROS toolchain)':::Part1 ch04_08 - ROS noetic"
  "part2_ch01_03:::cd /workspace/part2_ch01_03/build && ./feature_matching ../data/1.jpg ../data/2.jpg:::Part2 ch01_03 - Feature Matching (1.jpg/2.jpg) [WINDOW: matches visualization]"
  "part2_ch01_07:::cd /workspace/part2_ch01_07/build && ./dense_optical_flow ../data/frame_0.png ../data/frame_1.png:::Part2 ch01_07 - Dense Optical Flow on TUM [WINDOW: flow visualization]"
  "part2_ch01_09:::cd /workspace/part2_ch01_09/build && ./vocabulary_training ../data:::Part2 ch01_09 - DBoW2 Vocabulary Training on TUM (no GUI)"
  "part2_ch01_10:::cd /workspace/part2_ch01_10/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -10; fi; done:::Part2 ch01_10 - Learning-based features"
  "part2_ch02_02:::cd /workspace/part2_ch02_02/build && for b in epipolar_visualization essential_fundamental_demo pose_recovery relpose_poselib; do echo '== '\$b' =='; ./\$b ../data/left.png ../data/right.png 2>&1 | head -10; done:::Part2 ch02_02 - Essential/Fundamental Matrix [4 binaries]"
  "part2_ch02_04:::cd /workspace/part2_ch02_04/build && ./homography_demo ../data/000024.png ../data/000025.png:::Part2 ch02_04 - Homography on KITTI [WINDOW]"
  "part2_ch02_05:::cd /workspace/part2_ch02_05/build && ./monocular_vo_demo ../data/kitti:::Part2 ch02_05 - Monocular VO on KITTI [WINDOW: tracking]"
  "part2_ch02_07:::cd /workspace/part2_ch02_07/build && ./triangulation_demo:::Part2 ch02_07 - Triangulation (synthetic stereo)"
  "part2_ch02_09:::cd /workspace/part2_ch02_09/build && ./charuco_calibration:::Part2 ch02_09 - PnP / ChArUco Calibration [WINDOW]"
  "part2_ch02_12:::cd /workspace/part2_ch02_12/build && ./ransac_fundamental ../data/000024.png ../data/000025.png:::Part2 ch02_12 - RANSAC Fundamental on EuRoC"
  "part2_ch03_04:::cd /workspace/part2_ch03_04/build && for b in basic_io filtering features registration visualization; do echo '== '\$b' =='; ./\$b ../data/000000.bin 2>&1 | head -10; done:::Part2 ch03_04 - PCL Tutorial [several binaries]"
  "part2_ch03_06:::cd /workspace/part2_ch03_06/build && ./icp_visualization ../data/scene.pcd ../data/scene.pcd --step:::Part2 ch03_06 - ICP Visualization [WINDOW: VTK 3D viewer]"
  "part2_ch03_07:::cd /workspace/part2_ch03_07/build && ./teaser_demo:::Part2 ch03_07 - TEASER++ / KISS-ICP"
  "part2_ch03_08:::cd /workspace/part2_ch03_08/build && ./comparison ../data/000000.bin:::Part2 ch03_08 - Octree/Octomap/Bonxai comparison"
  "part3_ch01_13:::cd /workspace/part3_ch01_13/build && ./g2o_bundle_adjustment:::Part3 ch01_13 - g2o BAL Bundle Adjustment"
  "part3_ch01_14:::cd /workspace/part3_ch01_14/build && ./gtsam_bundle_adjustment:::Part3 ch01_14 - GTSAM BAL Bundle Adjustment"
  "part3_ch01_15:::cd /workspace/part3_ch01_15/build && ./ceres_bundle_adjustment:::Part3 ch01_15 - Ceres BAL Bundle Adjustment"
  "part3_ch01_17:::/workspace/build/rpgo_basics; /workspace/build/rpgo_outlier_rejection:::Part3 ch01_17 - Kimera-RPGO"
)

TOTAL=${#DEMOS[@]}
i=0
for entry in "${DEMOS[@]}"; do
  i=$((i+1))
  # Split on `:::` (three colons). Use awk because read with multi-char IFS isn't portable.
  demo=$(awk -F':::' '{print $1}' <<< "$entry")
  cmd=$(awk -F':::' '{$1=""; $NF=""; sub(/^:::/, ""); sub(/:::$/, ""); print}' <<< "$entry")
  # Simpler: extract first/last/middle by sed
  demo="${entry%%:::*}"
  rest="${entry#*:::}"
  cmd="${rest%:::*}"
  label="${rest##*:::}"

  TAG="slam_zero_to_hero:${demo}"

  echo ""
  echo "================================================================"
  echo "[$i/$TOTAL] $label"
  echo "================================================================"
  echo "Demo: $demo"
  echo "Image: $TAG"
  echo "Cmd: $cmd"
  echo ""
  echo "Press Enter to RUN, 's' to SKIP, 'q' to QUIT:"
  read -r ans
  case "$ans" in
    q|Q)
      echo "Quitting."
      exit 0 ;;
    s|S)
      echo "Skipped."
      continue ;;
  esac

  # Mount the demo's data if it exists, in case the binary writes outputs to it
  EXTRA_MOUNTS=()
  if [ -d "/home/deepgadget/slam_lecture_codes/SLAM_zero_to_hero/${demo}/data" ]; then
    EXTRA_MOUNTS+=( -v "/home/deepgadget/slam_lecture_codes/SLAM_zero_to_hero/${demo}/data:/workspace/${demo}/data:rw" )
  fi

  echo "--- starting (close any GUI windows when done viewing) ---"
  podman run --rm -it \
    "${X11_FLAGS[@]}" \
    "${EXTRA_MOUNTS[@]}" \
    "$TAG" \
    bash -c "$cmd; echo; echo '[demo finished -- press Enter in this terminal to advance]'; read"
done

echo ""
echo "================================================================"
echo "All demos walked through."
echo "================================================================"
