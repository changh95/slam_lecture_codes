cmake_minimum_required(VERSION 3.10)
project(ceres_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(Ceres REQUIRED)
message(STATUS "Ceres found: ${CERES_FOUND} (version ${Ceres_VERSION})")

add_executable(ceres_curve_fitting examples/ceres_curve_fitting.cpp)
add_executable(ceres_pose_graph examples/ceres_pose_graph.cpp)
add_executable(ceres_bundle_adjustment examples/ceres_bundle_adjustment.cpp)

set(ALL_TARGETS
    ceres_curve_fitting
    ceres_pose_graph
    ceres_bundle_adjustment
)

foreach(target ${ALL_TARGETS})
    target_link_libraries(${target} Ceres::ceres)
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
message(STATUS "=== Ceres-Solver Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  ceres_curve_fitting     - fit y=exp(a x^2+b x+c)")
message(STATUS "  ceres_pose_graph        - 2D pose-graph optimization")
message(STATUS "  ceres_bundle_adjustment - bundle adjustment with BAL dataset")
message(STATUS "")
