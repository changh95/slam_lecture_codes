cmake_minimum_required(VERSION 3.10)
project(g2o_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)

message(STATUS "g2o found: ${g2o_FOUND}")
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

# Curve fitting: custom vertex + unary edge, dense Gauss-Newton.
add_executable(g2o_curve_fitting examples/g2o_curve_fitting.cpp)
target_link_libraries(g2o_curve_fitting
    g2o::core g2o::stuff g2o::solver_dense spdlog::spdlog)

# Pose-graph optimization: built-in VertexSE2 / EdgeSE2.
add_executable(g2o_pose_graph examples/g2o_pose_graph.cpp)
target_link_libraries(g2o_pose_graph
    g2o::core g2o::stuff g2o::types_slam2d g2o::solver_dense spdlog::spdlog)

# Bundle adjustment: custom BAL camera/point types, sparse Schur (Eigen).
add_executable(g2o_bundle_adjustment examples/g2o_bundle_adjustment.cpp)
target_link_libraries(g2o_bundle_adjustment
    g2o::core g2o::stuff g2o::solver_eigen spdlog::spdlog)

message(STATUS "")
message(STATUS "=== g2o Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  g2o_curve_fitting     - fit y=exp(a x^2+b x+c)")
message(STATUS "  g2o_pose_graph        - 2D pose-graph optimization")
message(STATUS "  g2o_bundle_adjustment - bundle adjustment with BAL dataset")
message(STATUS "")
