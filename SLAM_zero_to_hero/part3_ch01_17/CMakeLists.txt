cmake_minimum_required(VERSION 3.10)
project(rpgo_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type: without this everything compiles at -O0, unlike every sibling
# chapter, which makes the timings meaningless.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(GTSAM REQUIRED)
find_package(KimeraRPGO REQUIRED)

include_directories(${GTSAM_INCLUDE_DIR})

# Example 1: pose graph construction and robust optimization with noisy odometry
add_executable(rpgo_basics examples/rpgo_basics.cpp)

# Example 2: PCM / GNC / no-rejection contrast on a graph with outlier loops
add_executable(rpgo_outlier_rejection examples/rpgo_outlier_rejection.cpp)

# Example 3: the shared 5-pose SE(2) pose graph, for diffing against the
# g2o / GTSAM / Ceres / SymForce chapters
add_executable(rpgo_pose_graph examples/rpgo_pose_graph.cpp)

set(ALL_TARGETS rpgo_basics rpgo_outlier_rejection rpgo_pose_graph)

foreach(target ${ALL_TARGETS})
    target_link_libraries(${target} gtsam KimeraRPGO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
endforeach()

# ============================================
# Optional: rerun C++ SDK (live visualization)
# ============================================
find_package(rerun_sdk QUIET)
if(rerun_sdk_FOUND)
    message(STATUS "rerun C++ SDK found - demos stream results to a rerun viewer")
    foreach(target ${ALL_TARGETS})
        target_link_libraries(${target} rerun_sdk)
        target_compile_definitions(${target} PRIVATE HAVE_RERUN)
    endforeach()
else()
    message(STATUS "rerun C++ SDK not found - demos run without live streaming")
endif()

message(STATUS "")
message(STATUS "=== Kimera-RPGO: Robust Pose Graph Optimization ===")
message(STATUS "Executables (run them from this build directory):")
message(STATUS "  ./rpgo_basics              3D square loop, noisy odometry")
message(STATUS "  ./rpgo_outlier_rejection   no rejection vs PCM vs GNC")
message(STATUS "  ./rpgo_pose_graph          shared 5-pose SE(2) problem")
message(STATUS "")
