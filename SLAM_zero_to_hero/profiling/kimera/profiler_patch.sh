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
#
# Only featureTracking and geometricOutlierRejection live here. Feature
# detection has its own file (see FeatureDetector patch below). The
# `geometricOutlierRejection` pattern matches the first of the 2d2d/3d3d
# overloads; that's fine, sed injects into the first match only.
# ---------------------------------------------------------------------------
TRACKER="${KIMERA_SRC}/src/frontend/Tracker.cpp"
if [ -f "$TRACKER" ]; then
  insert_profiler_guard "$TRACKER"
  insert_easy_block "$TRACKER" "void Tracker::featureTracking" "SLAM/FeatureTracking" "profiler::colors::Cyan"
  insert_easy_block "$TRACKER" "Tracker::geometricOutlierRejection2d2d" "SLAM/RANSAC" "profiler::colors::Red"
else
  echo "[warn] Tracker.cpp not found at $TRACKER"
fi

# ---------------------------------------------------------------------------
# Patch frontend/feature-detector/FeatureDetector.cpp
#
# Upstream moved detection into its own class; the old patch searched
# Tracker.cpp for `Tracker::featureDetection` which no longer exists and
# silently dropped the FeatureExtraction block.
# ---------------------------------------------------------------------------
FEAT_DET="${KIMERA_SRC}/src/frontend/feature-detector/FeatureDetector.cpp"
if [ -f "$FEAT_DET" ]; then
  insert_profiler_guard "$FEAT_DET"
  insert_easy_block "$FEAT_DET" "void FeatureDetector::featureDetection" "SLAM/FeatureExtraction" "profiler::colors::Green"
else
  echo "[warn] FeatureDetector.cpp not found at $FEAT_DET"
fi

# ---------------------------------------------------------------------------
# Patch backend/VioBackend.cpp
#
# The old patch pattern `addVisualInertialState` matched the *call site*
# inside spinOnce at line 167 and injected EASY_BLOCK into spinOnce's body
# instead of the real function definition. Anchor on the typed function
# signature so we hit the definition at line 296. `VioBackend::optimize`
# is unambiguous (only one definition).
# ---------------------------------------------------------------------------
VIO_BACKEND="${KIMERA_SRC}/src/backend/VioBackend.cpp"
if [ -f "$VIO_BACKEND" ]; then
  insert_profiler_guard "$VIO_BACKEND"
  # Two overloads exist: the first at ~line 296 is the bootstrap/init path
  # (called once), the second at ~line 430 is the per-keyframe entry point
  # called from VioBackend::spinOnce. Anchor on the BackendInput parameter so
  # we instrument the workhorse, not the bootstrap.
  insert_easy_block "$VIO_BACKEND" "addVisualInertialStateAndOptimize(const BackendInput" "SLAM/BackendUpdate" "profiler::colors::Orange"
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
# Patch frontend/StereoVisionImuFrontend.cpp (per-frame processing entry)
#
# The old patch targeted `spinOnce` in StereoImuPipeline.cpp, but that file
# only contains a `std::bind(&StereoImuPipeline::spinOnce, ...)` reference;
# the real spinOnce lives in Pipeline.cpp and its body is only a 2-line
# queue push. `StereoVisionImuFrontend::processStereoFrame` is the real
# per-frame work: feature detection, tracking, RANSAC, stereo matching.
# ---------------------------------------------------------------------------
FRONTEND="${KIMERA_SRC}/src/frontend/StereoVisionImuFrontend.cpp"
if [ -f "$FRONTEND" ]; then
  insert_profiler_guard "$FRONTEND"
  insert_easy_block "$FRONTEND" "StereoVisionImuFrontend::processStereoFrame" "SLAM/FrameProcess" "profiler::colors::Blue"
else
  FRONTEND="${KIMERA_SRC}/src/frontend/MonoVisionImuFrontend.cpp"
  if [ -f "$FRONTEND" ]; then
    insert_profiler_guard "$FRONTEND"
    insert_easy_block "$FRONTEND" "MonoVisionImuFrontend::processFrame" "SLAM/FrameProcess" "profiler::colors::Blue"
  else
    echo "[warn] Neither StereoVisionImuFrontend.cpp nor MonoVisionImuFrontend.cpp found"
  fi
fi

# ---------------------------------------------------------------------------
# Patch examples/KimeraVIO.cpp (main entry: enable profiler + dump on exit)
#
# Uses a Python helper because:
#   1. `grep -q EASY_PROFILER_ENABLE` false-matches the stub `#define
#      EASY_PROFILER_ENABLE` line that `insert_profiler_guard` just added,
#      which makes the previous `sed -i` path silently skip the insertion.
#   2. Kimera-VIO's main() ends with
#          return is_pipeline_successful ? EXIT_SUCCESS : EXIT_FAILURE;
#      so searching for "return 0;" finds nothing and the dump insertion
#      is silently skipped. Match any terminal return statement instead.
# ---------------------------------------------------------------------------
MAIN="${KIMERA_SRC}/examples/KimeraVIO.cpp"
if [ -f "$MAIN" ]; then
  insert_profiler_guard "$MAIN"
  python3 - "$MAIN" << 'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()

# Ensure easy/profiler.h is included inside the guard block (the stub-only
# guard from insert_profiler_guard does not include it).
if "easy/profiler.h" not in s:
    s = s.replace(
        "#ifdef BUILD_WITH_EASY_PROFILER\n",
        "#ifdef BUILD_WITH_EASY_PROFILER\n#include <easy/profiler.h>\n",
        1,
    )

# Enable the profiler at the top of main(). profiler::setEnabled(true) is
# the library call that EASY_PROFILER_ENABLE expands to; calling it directly
# avoids ambiguity with the stub #define of the same name.
if "profiler::setEnabled" not in s:
    s = re.sub(
        r"(int\s+main\s*\([^)]*\)\s*\{)",
        lambda m: m.group(0)
        + "\n#ifdef BUILD_WITH_EASY_PROFILER\n  profiler::setEnabled(true);\n#endif",
        s,
        count=1,
    )

# Insert dumpBlocksToFile before the LAST return statement in the file.
# Using chr(34) avoids quote-escaping issues inside the heredoc.
if "dumpBlocksToFile" not in s:
    Q = chr(34)
    DUMP = (
        "#ifdef BUILD_WITH_EASY_PROFILER\n"
        "  profiler::dumpBlocksToFile(" + Q + "/output/kimera_profile.prof" + Q + ");\n"
        "  fprintf(stderr, " + Q + "[easy_profiler] wrote /output/kimera_profile.prof\\n" + Q + ");\n"
        "#endif\n"
    )
    matches = list(re.finditer(r"^[ \t]*return [^;]*;", s, flags=re.MULTILINE))
    if not matches:
        print("[warn] no return statement found in main", file=sys.stderr)
    else:
        last = matches[-1]
        s = s[: last.start()] + DUMP + s[last.start():]

open(p, "w").write(s)
print("[patched] KimeraVIO.cpp: profiler::setEnabled + dumpBlocksToFile")
PY
else
  echo "[warn] KimeraVIO.cpp not found at $MAIN"
fi

echo ""
echo "=== Kimera-VIO profiler patch complete ==="
echo "Build with: cmake .. -DWITH_PROFILER=ON && make -j\$(nproc)"
