#!/bin/bash
# cmake_patch.sh - Patch voxblox_ros/CMakeLists.txt to add WITH_PROFILER option.
# Run after cloning voxblox, before building.
set -e

CMAKEFILE="/catkin_ws/src/voxblox/voxblox_ros/CMakeLists.txt"

if grep -q "WITH_PROFILER" "$CMAKEFILE"; then
  echo "[cmake_patch] WITH_PROFILER already present, skipping."
  exit 0
fi

echo "[cmake_patch] Patching ${CMAKEFILE}..."

# Insert option block after cmake_minimum_required line
sed -i '/cmake_minimum_required/a \
\
option(WITH_PROFILER "Build with easy_profiler instrumentation" OFF)\
if(WITH_PROFILER)\
  find_package(easy_profiler REQUIRED)\
  add_definitions(-DBUILD_WITH_EASY_PROFILER)\
  message(STATUS "easy_profiler enabled")\
endif()' "$CMAKEFILE"

# Append easy_profiler link at the end of the file (before last endif if any)
cat >> "$CMAKEFILE" << 'EOF'

if(WITH_PROFILER)
  if(TARGET voxblox_node)
    target_link_libraries(voxblox_node easy_profiler)
  endif()
  if(TARGET voxblox_ros)
    target_link_libraries(voxblox_ros easy_profiler)
  endif()
endif()
EOF

echo "[cmake_patch] CMakeLists.txt patched successfully."
