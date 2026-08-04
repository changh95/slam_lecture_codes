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

# Each entry: demo:::cmd:::label
# Labels mark CLI-only ([CLI]) vs window ([GUI]) demos so you know what to expect.
DEMOS=(
  "part1_ch02_02:::cd /workspace/part1_ch02_02/build && for b in for_loop while_loop vector map unordered_map template_function template_class smart_pointers; do echo '== '\$b' =='; ./\$b 2>&1 | head -20; done:::[CLI] Part1 ch02_02 - Basic C++ programming"
  "part1_ch02_03:::cd /workspace/part1_ch02_03/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -20; fi; done:::[CLI] Part1 ch02_03 - Modern C++"
  "part1_ch02_04:::cd /workspace/part1_ch02_04/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -10; fi; done:::[CLI] Part1 ch02_04 - Profiling (easy_profiler)"
  "part1_ch02_05:::cd /workspace/part1_ch02_05/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -10; fi; done:::[CLI] Part1 ch02_05 - Memory tooling (Valgrind)"
  "part1_ch02_08:::cd /workspace/part1_ch02_08/build && python3 -c 'import sys; sys.path.insert(0, \".\"); import slam_bindings as m; print(\"Module:\", m); print(\"Members:\", [x for x in dir(m) if not x.startswith(\"_\")])' 2>&1:::[CLI] Part1 ch02_08 - nanobind Python bindings"
  "part1_ch03_05:::cd /workspace/part1_ch03_05/build && for b in *; do if [ -x \"\$b\" ] && [ -f \"\$b\" ]; then echo '== '\$b' =='; ./\$b 2>&1 | head -8; fi; done:::[CLI] Part1 ch03_05 - Math/Linear algebra"
  "part1_ch04_08:::source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && echo '=== ROS Noetic + Kalibr camera/IMU calibration toolchain ===' && echo '' && echo 'ROS distro:' \$ROS_DISTRO && echo 'Catkin pkgs:' && rospack list 2>/dev/null | awk '{print \$1}' | grep -iE 'kalibr|allan|aslam|apriltag|ethz' | head -15 && echo '' && echo 'Kalibr executables:' && ls /catkin_ws/devel/.private/kalibr/lib/kalibr/ 2>/dev/null | head -15 && echo '' && echo 'allan_variance_ros executables:' && ls /catkin_ws/devel/.private/allan_variance_ros/lib/allan_variance_ros/ 2>/dev/null | head:::[CLI] Part1 ch04_08 - Kalibr camera/IMU calibration (ROS noetic)"
  "part2_ch01_03:::cd /workspace/part2_ch01_03/build && ./feature_matching ../data/1.jpg ../data/2.jpg:::[GUI] Part2 ch01_03 - Feature Matching (1.jpg/2.jpg) - opens match visualization"
  "part2_ch01_07:::cd /workspace/part2_ch01_07/build && ./dense_optical_flow ../data/frame_0.png ../data/frame_1.png:::[CLI] Part2 ch01_07 - Dense vs Sparse Optical Flow benchmark on TUM"
  "part2_ch01_09:::cd /workspace/part2_ch01_09/build && ./vocabulary_training ../data:::[CLI] Part2 ch01_09 - DBoW2 Vocabulary Training on TUM"
  "part2_ch01_10:::ldconfig; cd /VPR_Tutorial && echo '=== VPR Tutorial: cross-time-of-day place recognition (RTX 5090) ===' && for desc in AlexNet NetVLAD PatchNetVLAD CosPlace EigenPlaces SAD; do echo; echo '------- '\$desc' -------'; python3 demo.py --descriptor \$desc --dataset GardensPoint 2>&1 | grep -E 'Using|R@|AUC|R@100P' | head -5; done:::[GPU] Part2 ch01_10 - VPR Tutorial (6 deep descriptors on GardensPoint day-vs-night)"
  "part2_ch02_02:::cd /workspace/part2_ch02_02/build && ./epipolar_visualization ../data/left.png ../data/right.png:::[GUI] Part2 ch02_02 - Epipolar Visualization on stereo pair"
  "part2_ch02_04:::cd /workspace/part2_ch02_04/build && ./image_stitching ../data/000024.png ../data/000025.png:::[GUI] Part2 ch02_04 - Image Stitching on KITTI pair"
  "part2_ch02_05:::cd /workspace/part2_ch02_05/build && ./run_vo_kitti ../data/kitti ../data/poses.txt:::[GUI] Part2 ch02_05 - Monocular VO on KITTI seq 00 (30 frames + ground-truth poses)"
  "part2_ch02_07:::cd /workspace/part2_ch02_07/build && ./triangulation_demo:::[CLI] Part2 ch02_07 - Triangulation (synthetic stereo benchmark)"
  "part2_ch02_09:::cd /workspace/part2_ch02_09/build && ./charuco_calibration:::[GUI] Part2 ch02_09 - ChArUco Calibration window"
  "part2_ch02_12:::cd /workspace/part2_ch02_12/build && ./ransac_fundamental ../data/000024.png ../data/000025.png:::[CLI] Part2 ch02_12 - RANSAC Fundamental on EuRoC (5 estimators benchmarked)"
  "part2_ch03_04:::cd /workspace/part2_ch03_04 && for b in passthrough downsampling sor kdtree plane_det normal_estimation; do echo '== '\$b' =='; ./build/\$b; done:::[GUI] Part2 ch03_04 - PCL preprocessing on KITTI scan (6 windows: passthrough, voxel, SOR, k-d tree, RANSAC plane, normals; close each to advance)"
  "part2_ch03_06:::cd /workspace/part2_ch03_06/build && ./icp_visualization ../data/scene.pcd ../data/scene.pcd --step:::[GUI] Part2 ch03_06 - ICP Visualization (VTK 3D viewer)"
  "part2_ch03_07:::cd /workspace/part2_ch03_07/build && ./teaser_demo:::[CLI] Part2 ch03_07 - TEASER++ / Advanced ICP benchmark"
  "part2_ch03_08:::cd /workspace/part2_ch03_08/build && ./comparison ../data/000000.bin:::[CLI] Part2 ch03_08 - Octree/Octomap/Bonxai comparison on KITTI scan"
  "part3_ch01_13:::cd /workspace/part3_ch01_13/build && ./g2o_bundle_adjustment:::[CLI] Part3 ch01_13 - g2o BAL Bundle Adjustment"
  "part3_ch01_14:::cd /workspace/part3_ch01_14/build && ./gtsam_bundle_adjustment:::[CLI] Part3 ch01_14 - GTSAM BAL Bundle Adjustment"
  "part3_ch01_15:::cd /workspace/part3_ch01_15/build && ./ceres_bundle_adjustment:::[CLI] Part3 ch01_15 - Ceres BAL Bundle Adjustment"
  "part3_ch01_17:::/workspace/build/rpgo_basics; /workspace/build/rpgo_outlier_rejection:::[CLI] Part3 ch01_17 - Kimera-RPGO basics + outlier rejection"
)

TOTAL=${#DEMOS[@]}
i=0
for entry in "${DEMOS[@]}"; do
  i=$((i+1))
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
  if [ -d "/home/deepgadget/slam_lecture_codes/SLAM_zero_to_hero/${demo}/weights" ]; then
    EXTRA_MOUNTS+=( -v "/home/deepgadget/slam_lecture_codes/SLAM_zero_to_hero/${demo}/weights:/workspace/${demo}/weights:rw" )
  fi

  # GPU passthrough for demos that need CUDA (currently just part2_ch01_10
  # for the VPR Tutorial running PyTorch on the RTX 5090). Mounts the host
  # nvidia driver libs into /usr/lib/x86_64-linux-gnu/ so libcuda.so.1
  # resolves without nvidia-container-toolkit.
  GPU_FLAGS=()
  if [[ "$demo" == "part2_ch01_10" ]]; then
    for d in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
      [ -e "$d" ] && GPU_FLAGS+=( --device "$d" )
    done
    for f in /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
             /usr/lib/x86_64-linux-gnu/libcuda.so.580.126.18 \
             /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 \
             /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.126.18; do
      [ -f "$f" ] && GPU_FLAGS+=( -v "$f:$f:ro" )
    done
  fi

  echo "--- starting (close any GUI windows when done viewing) ---"
  podman run --rm -it --shm-size=4g \
    "${X11_FLAGS[@]}" \
    "${EXTRA_MOUNTS[@]}" \
    "${GPU_FLAGS[@]}" \
    "$TAG" \
    bash -c "$cmd; echo; echo '[demo finished -- press Enter in this terminal to advance]'; read"
done

echo ""
echo "================================================================"
echo "All demos walked through."
echo "================================================================"
