#!/bin/bash
# Patch a fresh clone of glim v1.0.0/v1.1.0 to add easy_profiler instrumentation
# at the major SLAM component boundaries. Run from inside the /glim clone.
#
# Bug history (fixed in this version):
#   - The previous sed-based EASY_BLOCK insertion used a greedy `\(.*{\)`
#     pattern that matched the LAST `{` on a 2-line pattern space, including
#     `{` inside `logger->info("foo {} bar")` format strings. That split log
#     calls in half and broke compilation. Replaced with a Python helper that
#     anchors on the function-signature line and inserts after the immediately
#     following bare `{` (at start of line, possibly with whitespace).
#   - The CMakeLists.txt patch used to insert `option(BUILD_WITH_EASY_PROFILER)`
#     after the last `find_package(...)` call. In glim's CMakeLists that
#     happens to be `find_package(catkin REQUIRED)` -- which is itself nested
#     inside `if(DEFINED ENV{ROS_VERSION}) elseif(ROS_VERSION EQUAL 1)`.
#     The Dockerfile builds glim without sourcing ROS, so the entire ROS
#     conditional is skipped at configure time and the option (plus the
#     `-DBUILD_WITH_EASY_PROFILER` define) never reaches glim's compile
#     commands -> all EASY_BLOCK macros expand to no-ops. Replaced with a
#     direct insert before `add_library(glim` so the option lives at top-level.
set -e

GLIM_DIR="${1:-/glim}"
cd "$GLIM_DIR"

INCLUDE_GUARD='
#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#else
#define EASY_FUNCTION(...)
#define EASY_BLOCK(...)
#define EASY_END_BLOCK
#define EASY_PROFILER_ENABLE
#endif
'

inject_guard() {
    local file="$1"
    if grep -q "BUILD_WITH_EASY_PROFILER" "$file"; then
        return  # already patched
    fi
    awk -v guard="$INCLUDE_GUARD" '
        /^#include/ { print; last_include=NR; next }
        NR == last_include+1 && !guard_done { print guard; guard_done=1 }
        { print }
        END { if (!guard_done) print guard }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# Insert "EASY_BLOCK(\"NAME\", COLOR);" right after the function-body opening
# brace. Anchors on a signature regex; finds the first line that is a bare `{`
# (or `{` followed by trailing whitespace) at or after the signature line.
#
# Args:  $1=file  $2=signature_grep_pattern  $3=block_name  $4=color
inject_block_py() {
  local file="$1"
  local sig_pattern="$2"
  local block_name="$3"
  local color="${4:-profiler::colors::Blue}"

  if grep -q "EASY_BLOCK(\"${block_name}\"" "$file"; then
    echo "[skip] $block_name already in $file"
    return
  fi

  python3 - "$file" "$sig_pattern" "$block_name" "$color" <<'PYEOF'
import sys, re
path, sig_pat, name, color = sys.argv[1:5]
with open(path) as f:
    lines = f.readlines()

# Find signature line (regex match anywhere in the line)
sig_re = re.compile(sig_pat)
sig_idx = None
for i, ln in enumerate(lines):
    if sig_re.search(ln):
        sig_idx = i
        break
if sig_idx is None:
    print(f"[warn] {path}: pattern not found: {sig_pat!r}", file=sys.stderr)
    sys.exit(0)

# Walk forward to find the function-body opening `{`. We accept either
# (a) the signature line itself ending with `{` (possibly with trailing space),
# or (b) a subsequent line that is exactly `{` (with optional leading/trailing whitespace).
# Multi-line signatures (like Optimizer::LocalBundleAdjustment) have `{` on
# the line after the param list closing `)`.
brace_idx = None
for j in range(sig_idx, min(sig_idx + 10, len(lines))):
    s = lines[j].rstrip()
    if s.endswith("{") or s == "{":
        brace_idx = j
        break
if brace_idx is None:
    print(f"[warn] {path}: no opening brace within 10 lines of signature {sig_pat!r}", file=sys.stderr)
    sys.exit(0)

block_line = f'  EASY_BLOCK("{name}", {color});\n'
lines.insert(brace_idx + 1, block_line)

with open(path, "w") as f:
    f.writelines(lines)
print(f"[patched] {path}: {name} after line {brace_idx+1}")
PYEOF
}

# 1. cloud_preprocessor.cpp: SLAM/Preprocessing inside preprocess()
F=src/glim/preprocess/cloud_preprocessor.cpp
inject_guard "$F"
inject_block_py "$F" "^PreprocessedFrame::Ptr CloudPreprocessor::preprocess\\(" \
    "SLAM/Preprocessing" "profiler::colors::Yellow"

# 2. odometry_estimation_imu.cpp: SLAM/FrameProcess inside insert_frame()
F=src/glim/odometry/odometry_estimation_imu.cpp
inject_guard "$F"
inject_block_py "$F" "^EstimationFrame::ConstPtr OdometryEstimationIMU::insert_frame\\(" \
    "SLAM/FrameProcess" "profiler::colors::Red"

# 3. sub_mapping.cpp: SLAM/LocalMapping inside SubMapping::insert_frame
F=src/glim/mapping/sub_mapping.cpp
inject_guard "$F"
inject_block_py "$F" "^void SubMapping::insert_frame\\(" \
    "SLAM/LocalMapping" "profiler::colors::Yellow"

# 4. global_mapping.cpp: SLAM/GlobalMapping inside insert_submap (single-arg)
#    + SLAM/GlobalMapping/Optimize inside optimize().
F=src/glim/mapping/global_mapping.cpp
inject_guard "$F"
inject_block_py "$F" "^void GlobalMapping::insert_submap\\(const SubMap::Ptr&" \
    "SLAM/GlobalMapping" "profiler::colors::Purple"
inject_block_py "$F" "^void GlobalMapping::optimize\\(\\)" \
    "SLAM/GlobalMapping/Optimize" "profiler::colors::Red"

# 5. CMakeLists.txt: insert option() and find_package() at top-level (NOT
#    inside the ROS conditional where the previous version put it). Place it
#    just before `add_library(glim` so the -DBUILD_WITH_EASY_PROFILER define
#    is set before glim's sources are compiled.
if ! grep -q "BUILD_WITH_EASY_PROFILER" CMakeLists.txt; then
  python3 - CMakeLists.txt <<'PYEOF'
import sys, re
p = sys.argv[1]
with open(p) as f:
    s = f.read()

block = (
    "\n"
    "# easy_profiler (must be at top-level, NOT inside ROS_VERSION conditional,\n"
    "# so the define reaches glim's add_library compile commands)\n"
    'option(BUILD_WITH_EASY_PROFILER "Build with easy_profiler instrumentation" OFF)\n'
    "if(BUILD_WITH_EASY_PROFILER)\n"
    "  find_package(easy_profiler REQUIRED)\n"
    "  add_definitions(-DBUILD_WITH_EASY_PROFILER)\n"
    "endif()\n\n"
)

# Insert just before `add_library(glim`
s2, n = re.subn(r"(\nadd_library\(glim\b)", block + r"\1", s, count=1)
if n == 0:
    print("[warn] add_library(glim ...) not found; appending option block at end of CMakeLists.txt", file=sys.stderr)
    s2 = s + "\n" + block

with open(p, "w") as f:
    f.write(s2)
print("[patched] CMakeLists.txt: top-level BUILD_WITH_EASY_PROFILER block inserted before add_library(glim)")
PYEOF
fi

# Append `easy_profiler` to target_link_libraries(glim ...) via generator
# expression. This was already correct in the previous version of the script.
if ! grep -q '<\$<BOOL:\${BUILD_WITH_EASY_PROFILER}>:easy_profiler>' CMakeLists.txt; then
  sed -i '/^target_link_libraries(glim$/,/^)/{
      /^)/i\
  $<$<BOOL:${BUILD_WITH_EASY_PROFILER}>:easy_profiler>
  }' CMakeLists.txt
fi

echo "Patched glim with easy_profiler instrumentation"
grep -l "EASY_BLOCK" src/glim/ -r 2>/dev/null | sort
