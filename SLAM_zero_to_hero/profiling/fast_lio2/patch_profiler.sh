#!/bin/bash
# Patch a fresh clone of FAST-LIO2 to add easy_profiler instrumentation
# at the major SLAM component boundaries. Run from the profiling/fast_lio2
# directory, pointing at the FAST-LIO2 source root (the folder that
# contains CMakeLists.txt and src/).
set -e

FASTLIO_DIR="${1:-/fast_lio2}"
cd "$FASTLIO_DIR"

# ---------------------------------------------------------------------------
# Helper: inject the #ifdef BUILD_WITH_EASY_PROFILER guard after the last
# #include line in a file.
# ---------------------------------------------------------------------------
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
    # Insert include guard right after the last #include line
    awk -v guard="$INCLUDE_GUARD" '
        /^#include/ { print; last_include=NR; next }
        NR == last_include+1 && !guard_done { print guard; guard_done=1 }
        { print }
        END { if (!guard_done) print guard }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# ---------------------------------------------------------------------------
# 1. src/preprocess.cpp
#    SLAM/Preprocessing – instrument both overloads of Preprocess::process()
# ---------------------------------------------------------------------------
F=src/preprocess.cpp
inject_guard "$F"

# Livox overload: void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, ...)
sed -i '/^void Preprocess::process(const livox_ros_driver/,/^}/{
    /^void Preprocess::process(const livox_ros_driver/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/Preprocessing", profiler::colors::Yellow);|
    }
}' "$F"

# Standard PCL overload: void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, ...)
sed -i '/^void Preprocess::process(const sensor_msgs/,/^}/{
    /^void Preprocess::process(const sensor_msgs/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/Preprocessing", profiler::colors::Yellow);|
    }
}' "$F"

# ---------------------------------------------------------------------------
# 2. src/IMU_Processing.hpp
#    SLAM/IMUProcessing/Undistort – instrument UndistortPcl()
#    SLAM/IMUProcessing           – instrument ImuProcess::Process()
# ---------------------------------------------------------------------------
F=src/IMU_Processing.hpp
inject_guard "$F"

# UndistortPcl – private method
sed -i '/^void ImuProcess::UndistortPcl/,/^}/{
    /^void ImuProcess::UndistortPcl/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/IMUProcessing/Undistort", profiler::colors::Orange);|
    }
}' "$F"

# Process – public method
sed -i '/^void ImuProcess::Process/,/^}/{
    /^void ImuProcess::Process/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/IMUProcessing", profiler::colors::Orange);|
    }
}' "$F"

# ---------------------------------------------------------------------------
# 3. src/laserMapping.cpp
#    Multiple blocks:
#      SLAM/FrameProcess    – outer sync_packages branch in main loop
#      SLAM/FOVSegment      – lasermap_fov_segment() body
#      SLAM/ICP             – h_share_model() body
#      SLAM/MapUpdate       – map_incremental() body
#      SLAM/EKFUpdate       – kf.update_iterated_dyn_share_modified call site
#      SLAM/PointCloudPublish – publish_* call cluster
# ---------------------------------------------------------------------------
F=src/laserMapping.cpp
inject_guard "$F"

# SLAM/FOVSegment: lasermap_fov_segment()
sed -i '/^void lasermap_fov_segment/,/^}/{
    /^void lasermap_fov_segment/{
        N; s|\(.*{\)|\1\n    EASY_BLOCK("SLAM/FOVSegment", profiler::colors::Cyan);|
    }
}' "$F"

# SLAM/ICP: h_share_model()
sed -i '/^void h_share_model/,/^}/{
    /^void h_share_model/{
        N; s|\(.*{\)|\1\n    EASY_BLOCK("SLAM/ICP", profiler::colors::Blue);|
    }
}' "$F"

# SLAM/MapUpdate: map_incremental()
sed -i '/^void map_incremental/,/^}/{
    /^void map_incremental/{
        N; s|\(.*{\)|\1\n    EASY_BLOCK("SLAM/MapUpdate", profiler::colors::Green);|
    }
}' "$F"

# SLAM/FrameProcess: wrap the entire if(sync_packages(Measures)) body.
# We insert EASY_BLOCK right after the opening brace of that if-block.
# The line is:        if(sync_packages(Measures))
# followed by        {
# We match two lines together and insert after the '{'.
python3 - "$F" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as fh:
    src = fh.read()

marker = 'if(sync_packages(Measures)) \n        {'
replacement = 'if(sync_packages(Measures)) \n        {\n            EASY_BLOCK("SLAM/FrameProcess", profiler::colors::Red);'

if marker in src:
    src = src.replace(marker, replacement, 1)
    with open(path, 'w') as fh:
        fh.write(src)
    print("Inserted SLAM/FrameProcess block")
else:
    print("WARNING: SLAM/FrameProcess marker not found – check whitespace", file=sys.stderr)
PYEOF

# SLAM/EKFUpdate: wrap kf.update_iterated_dyn_share_modified call
# Insert EASY_BLOCK before the call and EASY_END_BLOCK after it.
python3 - "$F" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as fh:
    lines = fh.readlines()

out = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'kf.update_iterated_dyn_share_modified(' in line and 'EASY_BLOCK' not in line:
        # Find the indentation
        indent = len(line) - len(line.lstrip())
        pad = ' ' * indent
        out.append(pad + 'EASY_BLOCK("SLAM/EKFUpdate", profiler::colors::Orange);\n')
        out.append(line)
        out.append(pad + 'EASY_END_BLOCK;\n')
    else:
        out.append(line)
    i += 1

with open(path, 'w') as fh:
    fh.writelines(out)
print("Inserted SLAM/EKFUpdate block")
PYEOF

# SLAM/PointCloudPublish: wrap the publish_* cluster
# The cluster starts at "if (path_en)  publish_path" and ends after the last publish call.
python3 - "$F" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as fh:
    lines = fh.readlines()

out = []
i = 0
while i < len(lines):
    line = lines[i]
    # Detect start of publish cluster
    if 'if (path_en)' in line and 'publish_path' in line and 'EASY_BLOCK' not in line:
        indent = len(line) - len(line.lstrip())
        pad = ' ' * indent
        out.append(pad + 'EASY_BLOCK("SLAM/PointCloudPublish", profiler::colors::Yellow);\n')
        out.append(line)
        # Consume the next two publish lines too
        i += 1
        while i < len(lines):
            next_line = lines[i]
            out.append(next_line)
            i += 1
            # After the last publish line (scan_body_pub_en line), close the block
            if 'scan_body_pub_en' in next_line and 'publish_frame_body' in next_line:
                out.append(pad + 'EASY_END_BLOCK;\n')
                break
        continue
    out.append(line)
    i += 1

with open(path, 'w') as fh:
    fh.writelines(out)
print("Inserted SLAM/PointCloudPublish block")
PYEOF

# ---------------------------------------------------------------------------
# 4. Add profiler enable + SIGINT dump logic to main()
#    Insert EASY_PROFILER_ENABLE after ros::init, register dump in SigHandle.
# ---------------------------------------------------------------------------
python3 - "$F" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as fh:
    src = fh.read()

# --- Enable profiler at startup (after ros::init line) ---
enable_marker = '    ros::init(argc, argv, "laserMapping");\n    ros::NodeHandle nh;\n\n'
enable_replacement = (
    '    ros::init(argc, argv, "laserMapping");\n'
    '    ros::NodeHandle nh;\n\n'
    '#ifdef BUILD_WITH_EASY_PROFILER\n'
    '    EASY_PROFILER_ENABLE;\n'
    '#endif\n\n'
)
if enable_marker in src and 'EASY_PROFILER_ENABLE;\n' not in src:
    src = src.replace(enable_marker, enable_replacement, 1)
    print("Inserted EASY_PROFILER_ENABLE")
else:
    print("WARNING: profiler enable marker not found or already present", file=sys.stderr)

# --- Dump on SIGINT: patch SigHandle ---
sighandle_marker = 'void SigHandle(int sig)\n{\n    flg_exit = true;\n    ROS_WARN("catch sig %d", sig);\n    sig_buffer.notify_all();\n}'
sighandle_replacement = (
    'void SigHandle(int sig)\n'
    '{\n'
    '    flg_exit = true;\n'
    '    ROS_WARN("catch sig %d", sig);\n'
    '    sig_buffer.notify_all();\n'
    '#ifdef BUILD_WITH_EASY_PROFILER\n'
    '    profiler::dumpBlocksToFile("/output/fast_lio2_profiler.prof");\n'
    '    ROS_INFO("easy_profiler dump written to /output/fast_lio2_profiler.prof");\n'
    '#endif\n'
    '}'
)
if sighandle_marker in src:
    src = src.replace(sighandle_marker, sighandle_replacement, 1)
    print("Patched SigHandle with profiler dump")
else:
    print("WARNING: SigHandle marker not found", file=sys.stderr)

with open(path, 'w') as fh:
    fh.write(src)
PYEOF

# ---------------------------------------------------------------------------
# 5. CMakeLists.txt: add WITH_PROFILER option, find_package, compile def,
#    and link easy_profiler to the fastlio_mapping target.
# ---------------------------------------------------------------------------
# Insert option block after the last find_package(...) line
awk '
    /find_package/ { last_fp = NR }
    { lines[NR] = $0 }
    END {
        for (i = 1; i <= NR; i++) {
            print lines[i]
            if (i == last_fp) {
                print ""
                print "option(BUILD_WITH_EASY_PROFILER \"Build with easy_profiler instrumentation\" OFF)"
                print "if(BUILD_WITH_EASY_PROFILER)"
                print "  find_package(easy_profiler REQUIRED)"
                print "  add_definitions(-DBUILD_WITH_EASY_PROFILER)"
                print "endif()"
            }
        }
    }
' CMakeLists.txt > CMakeLists.txt.tmp && mv CMakeLists.txt.tmp CMakeLists.txt

# Append easy_profiler to target_link_libraries for fastlio_mapping
sed -i 's|target_link_libraries(fastlio_mapping \(.*\))|target_link_libraries(fastlio_mapping \1 $<$<BOOL:${BUILD_WITH_EASY_PROFILER}>:easy_profiler>)|' CMakeLists.txt

echo ""
echo "Patched FAST-LIO2 with easy_profiler instrumentation."
echo "Files modified:"
grep -rl "EASY_BLOCK" src/ | sort
echo "  CMakeLists.txt"
echo ""
echo "Build with profiling:"
echo "  catkin build fast_lio -DBUILD_WITH_EASY_PROFILER=ON"
