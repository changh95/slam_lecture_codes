#!/bin/bash
# profiler_patch.sh - Apply easy_profiler instrumentation to Kimera-VIO source
# Run from inside the Kimera-VIO repository root.
set -e

KIMERA_SRC="${1:-/Kimera-VIO}"
PATCH_DIR="$(cd "$(dirname "$0")/src" && pwd)"

# ---------------------------------------------------------------------------
# Helper: insert profiler guard after the last #include in a .cpp file
# ---------------------------------------------------------------------------
insert_profiler_guard() {
  local file="$1"
  python3 - "$file" << 'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
if "BUILD_WITH_EASY_PROFILER" in src:
    print(f"[skip] {path} already patched")
    sys.exit(0)
guard = '''
#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#else
#define EASY_FUNCTION(...)
#define EASY_BLOCK(...)
#define EASY_END_BLOCK
#define EASY_PROFILER_ENABLE
#endif
'''
# Insert right at the top of the file so it precedes any use of EASY_BLOCK
with open(path, "w") as f:
    f.write(guard + src)
print(f"[patched] {path}")
PY
}

# ---------------------------------------------------------------------------
# Helper: add EASY_BLOCK at entry of a function by matching its signature
# ---------------------------------------------------------------------------
insert_easy_block() {
  local file="$1"
  local pattern="$2"   # grep pattern matching the function signature line
  local block_name="$3"
  local color="${4:-profiler::colors::Blue}"

  if grep -q "EASY_BLOCK(\"${block_name}\"" "$file"; then
    echo "[skip] $block_name already in $file"
    return
  fi
  # Find opening brace line after the signature
  local sig_line
  sig_line=$(grep -n "$pattern" "$file" | head -1 | cut -d: -f1)
  if [ -z "$sig_line" ]; then
    echo "[warn] Pattern '$pattern' not found in $file"
    return
  fi
  # Find first { on or after sig_line
  local brace_line
  brace_line=$(awk "NR>=${sig_line} && /\{/{print NR; exit}" "$file")
  if [ -z "$brace_line" ]; then
    echo "[warn] No opening brace found for '$pattern' in $file"
    return
  fi
  sed -i "${brace_line}a\\  EASY_BLOCK(\"${block_name}\", ${color});" "$file"
  echo "[patched] $file: added EASY_BLOCK(\"$block_name\") after line $brace_line"
}

# ---------------------------------------------------------------------------
# Patch CMakeLists.txt
# ---------------------------------------------------------------------------
CMAKE_FILE="${KIMERA_SRC}/CMakeLists.txt"
if ! grep -q "WITH_PROFILER" "$CMAKE_FILE"; then
  # Insert after the project() line
  sed -i '/^project(/a\
\
option(WITH_PROFILER "Build with easy_profiler instrumentation" OFF)\
if(WITH_PROFILER)\
  find_package(easy_profiler REQUIRED)\
  message(STATUS "easy_profiler found: ${easy_profiler_VERSION}")\
  add_compile_definitions(BUILD_WITH_EASY_PROFILER)\
else()\
  message(STATUS "Building without easy_profiler")\
endif()' "$CMAKE_FILE"

  # Append an explicit conditional target_link_libraries at the END of
  # CMakeLists.txt. This is more robust than trying to splice into the existing
  # target_link_libraries(kimera_vio ...) block which may span multiple lines
  # or use variables for the target name.
  cat >> "$CMAKE_FILE" << 'APPEND'

# easy_profiler link (added by profiler_patch.sh)
if(WITH_PROFILER)
  target_link_libraries(kimera_vio PUBLIC easy_profiler)
endif()
APPEND

  echo "[patched] CMakeLists.txt"
else
  echo "[skip] CMakeLists.txt already has WITH_PROFILER"
fi

# ---------------------------------------------------------------------------
# Patch frontend/Tracker.cpp
# ---------------------------------------------------------------------------
TRACKER="${KIMERA_SRC}/src/frontend/Tracker.cpp"
if [ -f "$TRACKER" ]; then
  insert_profiler_guard "$TRACKER"
  insert_easy_block "$TRACKER" "Tracker::featureTracking" "SLAM/FeatureTracking" "profiler::colors::Cyan"
  insert_easy_block "$TRACKER" "Tracker::featureDetection\b" "SLAM/FeatureExtraction" "profiler::colors::Green"
  insert_easy_block "$TRACKER" "geometricOutlierRejection" "SLAM/RANSAC" "profiler::colors::Red"
else
  echo "[warn] Tracker.cpp not found at $TRACKER"
fi

# ---------------------------------------------------------------------------
# Patch backend/VioBackend.cpp
# ---------------------------------------------------------------------------
VIO_BACKEND="${KIMERA_SRC}/src/backend/VioBackend.cpp"
if [ -f "$VIO_BACKEND" ]; then
  insert_profiler_guard "$VIO_BACKEND"
  insert_easy_block "$VIO_BACKEND" "addVisualInertialState" "SLAM/IMUIntegration" "profiler::colors::Orange"
  insert_easy_block "$VIO_BACKEND" "VioBackend::optimize" "SLAM/VIOOptimization" "profiler::colors::Magenta"
else
  echo "[warn] VioBackend.cpp not found at $VIO_BACKEND"
fi

# ---------------------------------------------------------------------------
# Patch frontend/StereoMatcher.cpp
# ---------------------------------------------------------------------------
STEREO="${KIMERA_SRC}/src/frontend/StereoMatcher.cpp"
if [ -f "$STEREO" ]; then
  insert_profiler_guard "$STEREO"
  insert_easy_block "$STEREO" "sparseStereoReconstruction" "SLAM/StereoMatching" "profiler::colors::Yellow"
else
  echo "[warn] StereoMatcher.cpp not found at $STEREO"
fi

# ---------------------------------------------------------------------------
# Patch pipeline/StereoImuPipeline.cpp (frame process loop)
# ---------------------------------------------------------------------------
PIPELINE="${KIMERA_SRC}/src/pipeline/StereoImuPipeline.cpp"
if [ -f "$PIPELINE" ]; then
  insert_profiler_guard "$PIPELINE"
  insert_easy_block "$PIPELINE" "spinOnce\b" "SLAM/FrameProcess" "profiler::colors::Blue"
else
  # Try MonoImuPipeline
  PIPELINE="${KIMERA_SRC}/src/pipeline/MonoImuPipeline.cpp"
  if [ -f "$PIPELINE" ]; then
    insert_profiler_guard "$PIPELINE"
    insert_easy_block "$PIPELINE" "spinOnce\b" "SLAM/FrameProcess" "profiler::colors::Blue"
  else
    echo "[warn] Neither StereoImuPipeline.cpp nor MonoImuPipeline.cpp found"
  fi
fi

# ---------------------------------------------------------------------------
# Patch examples/KimeraVIO.cpp (main entry: enable profiler + dump on exit)
# ---------------------------------------------------------------------------
MAIN="${KIMERA_SRC}/examples/KimeraVIO.cpp"
if [ -f "$MAIN" ]; then
  insert_profiler_guard "$MAIN"
  # Insert EASY_PROFILER_ENABLE after opening brace of main()
  if ! grep -q "EASY_PROFILER_ENABLE" "$MAIN"; then
    local_line=$(grep -n "^int main\b" "$MAIN" | head -1 | cut -d: -f1)
    if [ -n "$local_line" ]; then
      brace_line=$(awk "NR>=${local_line} && /\{/{print NR; exit}" "$MAIN")
      sed -i "${brace_line}a\\  EASY_PROFILER_ENABLE;" "$MAIN"
      echo "[patched] $MAIN: EASY_PROFILER_ENABLE added"
    fi
  fi
  # Insert dump before return 0
  if ! grep -q "dumpBlocksToFile" "$MAIN"; then
    ret_line=$(grep -n "return 0;" "$MAIN" | tail -1 | cut -d: -f1)
    if [ -n "$ret_line" ]; then
      sed -i "${ret_line}i\\#ifdef BUILD_WITH_EASY_PROFILER\\n  profiler::dumpBlocksToFile(\"/output/kimera_profile.prof\");\\n  std::cout << \"[Profiler] Saved to /output/kimera_profile.prof\" << std::endl;\\n#endif" "$MAIN"
      echo "[patched] $MAIN: dumpBlocksToFile added before return"
    fi
  fi
else
  echo "[warn] KimeraVIO.cpp not found at $MAIN"
fi

echo ""
echo "=== Kimera-VIO profiler patch complete ==="
echo "Build with: cmake .. -DWITH_PROFILER=ON && make -j\$(nproc)"
