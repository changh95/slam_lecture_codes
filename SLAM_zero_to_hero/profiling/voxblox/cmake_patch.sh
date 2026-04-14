#!/bin/bash
# cmake_patch.sh - Add WITH_PROFILER option to BOTH voxblox CMakeLists.
#
# The instrumented sources live in two packages:
#   - voxblox_ros (tsdf_server.cc — captures SLAM/FrameProcess)
#   - voxblox     (tsdf_integrator.cc, esdf_integrator.cc, mesh_integrator.h
#                  — captures Preprocessing, TsdfIntegration, EsdfIntegration,
#                  MeshGeneration)
#
# If we only patch voxblox_ros, BUILD_WITH_EASY_PROFILER is never defined when
# compiling the inner voxblox library, so all the integrator EASY_BLOCKs
# silently compile to nothing and only SLAM/FrameProcess shows up in the dump.
set -e

patch_one() {
  local cmakefile="$1"
  local target_name="$2"
  if [ ! -f "$cmakefile" ]; then
    echo "[cmake_patch] $cmakefile not found, skipping."
    return
  fi
  if grep -q "WITH_PROFILER" "$cmakefile"; then
    echo "[cmake_patch] $cmakefile already patched, skipping."
    return
  fi
  echo "[cmake_patch] Patching $cmakefile..."

  sed -i '/cmake_minimum_required/a \
\
option(WITH_PROFILER "Build with easy_profiler instrumentation" OFF)\
if(WITH_PROFILER)\
  find_package(easy_profiler REQUIRED)\
  add_definitions(-DBUILD_WITH_EASY_PROFILER)\
  message(STATUS "easy_profiler enabled in '"$target_name"'")\
endif()' "$cmakefile"

  cat >> "$cmakefile" << EOF

if(WITH_PROFILER)
  if(TARGET ${target_name})
    target_link_libraries(${target_name} easy_profiler)
  endif()
endif()
EOF
}

# Inner library: contains the integrators + mesh header
patch_one "/catkin_ws/src/voxblox/voxblox/CMakeLists.txt"     voxblox
# Outer ROS wrapper: contains tsdf_server.cc and the node main
patch_one "/catkin_ws/src/voxblox/voxblox_ros/CMakeLists.txt" voxblox_ros

# voxblox_ros also builds an executable (voxblox_node); link it too.
cat >> /catkin_ws/src/voxblox/voxblox_ros/CMakeLists.txt << 'EOF'

if(WITH_PROFILER)
  if(TARGET voxblox_node)
    target_link_libraries(voxblox_node easy_profiler)
  endif()
endif()
EOF

echo "[cmake_patch] Done."
