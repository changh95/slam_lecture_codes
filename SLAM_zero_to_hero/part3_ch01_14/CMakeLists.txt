cmake_minimum_required(VERSION 3.10)
project(gtsam_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(GTSAM REQUIRED)
message(STATUS "GTSAM found: ${GTSAM_FOUND}")

add_executable(gtsam_curve_fitting examples/gtsam_curve_fitting.cpp)
target_link_libraries(gtsam_curve_fitting gtsam)

add_executable(gtsam_pose_graph examples/gtsam_pose_graph.cpp)
target_link_libraries(gtsam_pose_graph gtsam)

add_executable(gtsam_bundle_adjustment examples/gtsam_bundle_adjustment.cpp)
target_link_libraries(gtsam_bundle_adjustment gtsam)

set(ALL_TARGETS gtsam_curve_fitting gtsam_pose_graph gtsam_bundle_adjustment)

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
message(STATUS "=== GTSAM Tutorial ===")
message(STATUS "Executables (run from the build directory):")
message(STATUS "  ./gtsam_curve_fitting     - fit y=exp(a x^2+b x+c)")
message(STATUS "  ./gtsam_pose_graph        - 2D pose-graph optimization")
message(STATUS "  ./gtsam_bundle_adjustment problem-21-11315-pre.txt")
message(STATUS "")
