#!/bin/bash
# Patch a fresh clone of glim v1.1.0 to add easy_profiler instrumentation
# at the major SLAM component boundaries. Run from inside the /glim clone.
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
    # Insert include guard right after the last #include line
    awk -v guard="$INCLUDE_GUARD" '
        /^#include/ { print; last_include=NR; next }
        NR == last_include+1 && !guard_done { print guard; guard_done=1 }
        { print }
        END { if (!guard_done) print guard }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# 1. cloud_preprocessor.cpp: SLAM/Preprocessing inside preprocess()
F=src/glim/preprocess/cloud_preprocessor.cpp
inject_guard "$F"
sed -i '/^PreprocessedFrame::Ptr CloudPreprocessor::preprocess/,/^}/{ /^PreprocessedFrame::Ptr CloudPreprocessor::preprocess/{ N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/Preprocessing", profiler::colors::Yellow);| } }' "$F"

# 2. odometry_estimation_imu.cpp: SLAM/FrameProcess inside insert_frame()
#    (the variant with signature returning EstimationFrame::Ptr)
F=src/glim/odometry/odometry_estimation_imu.cpp
inject_guard "$F"
sed -i '/^EstimationFrame::ConstPtr OdometryEstimationIMU::insert_frame/,/^}/{
    /^EstimationFrame::ConstPtr OdometryEstimationIMU::insert_frame/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/FrameProcess", profiler::colors::Red);|
    }
}' "$F"

# 3. sub_mapping.cpp: SLAM/LocalMapping inside SubMapping::insert_frame
F=src/glim/mapping/sub_mapping.cpp
inject_guard "$F"
sed -i '/^void SubMapping::insert_frame/,/^}/{
    /^void SubMapping::insert_frame/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/LocalMapping", profiler::colors::Yellow);|
    }
}' "$F"

# 4. global_mapping.cpp: SLAM/GlobalMapping inside insert_submap + /Optimize inside optimize
F=src/glim/mapping/global_mapping.cpp
inject_guard "$F"
sed -i '/^void GlobalMapping::insert_submap/,/^}/{
    /^void GlobalMapping::insert_submap/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/GlobalMapping", profiler::colors::Purple);|
    }
}' "$F"
sed -i '/^void GlobalMapping::optimize/,/^}/{
    /^void GlobalMapping::optimize/{
        N; s|\(.*{\)|\1\n  EASY_BLOCK("SLAM/GlobalMapping/Optimize", profiler::colors::Red);|
    }
}' "$F"

# 5. CMakeLists.txt: add option + find_package + link
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

# Add easy_profiler to target_link_libraries(glim ...)
sed -i '/^target_link_libraries(glim$/,/^)/{
    /^)/i\
  $<$<BOOL:${BUILD_WITH_EASY_PROFILER}>:easy_profiler>
}' CMakeLists.txt

echo "Patched glim v1.1.0 with easy_profiler instrumentation"
grep -l "EASY_BLOCK" src/glim/ -r | sort
