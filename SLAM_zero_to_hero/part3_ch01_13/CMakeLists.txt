cmake_minimum_required(VERSION 3.10)
project(g2o_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "g2o found: ${g2o_FOUND}")
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

# --- fmt, for g2o's spdlog logging -----------------------------------------
# The examples never mention spdlog, but g2o does: its headers emit spdlog calls
# into whatever includes them, and find_package(g2o) pulls spdlog::spdlog in as
# an interface dependency of g2o::stuff. spdlog is built here with
# SPDLOG_FMT_EXTERNAL, so both it and our objects need libfmt at link time - and
# on this base image the fmt headers in /usr/include/fmt (v8) are older than
# libfmt.so.9, so the shared library resolves nothing. The matching v8 static
# archive is /usr/lib/libfmt.a; link that explicitly and the whole
# g2o -> spdlog -> fmt chain resolves.
if(TARGET spdlog::spdlog)
    find_library(FMT_STATIC_LIBRARY NAMES libfmt.a PATHS /usr/local/lib /usr/lib)
    if(FMT_STATIC_LIBRARY)
        message(STATUS "fmt (for g2o/spdlog): ${FMT_STATIC_LIBRARY}")
        # Hang it off spdlog::spdlog rather than off each executable: static
        # archives are order-sensitive, and this is the only way to guarantee
        # fmt lands after libspdlog.a on the link line.
        set_property(TARGET spdlog::spdlog APPEND PROPERTY
                     INTERFACE_LINK_LIBRARIES "${FMT_STATIC_LIBRARY}")
    else()
        message(WARNING "g2o wants spdlog but no static libfmt.a was found - "
                        "linking may fail with undefined fmt:: references")
    endif()
endif()

# Curve fitting: custom vertex + unary edge, dense Levenberg-Marquardt.
add_executable(g2o_curve_fitting examples/g2o_curve_fitting.cpp)
target_link_libraries(g2o_curve_fitting
    g2o::core g2o::stuff g2o::solver_dense)

# Pose-graph optimization: built-in VertexSE2 / EdgeSE2.
add_executable(g2o_pose_graph examples/g2o_pose_graph.cpp)
target_link_libraries(g2o_pose_graph
    g2o::core g2o::stuff g2o::types_slam2d g2o::solver_dense)

# Bundle adjustment: custom BAL camera/point types, sparse Schur (Eigen).
add_executable(g2o_bundle_adjustment examples/g2o_bundle_adjustment.cpp)
target_link_libraries(g2o_bundle_adjustment
    g2o::core g2o::stuff g2o::solver_eigen)

set(ALL_TARGETS g2o_curve_fitting g2o_pose_graph g2o_bundle_adjustment)

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
message(STATUS "=== g2o Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  g2o_curve_fitting     - fit y=exp(a x^2+b x+c)")
message(STATUS "  g2o_pose_graph        - 2D pose-graph optimization")
message(STATUS "  g2o_bundle_adjustment - bundle adjustment with BAL dataset")
message(STATUS "")
