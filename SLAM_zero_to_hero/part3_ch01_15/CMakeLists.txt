cmake_minimum_required(VERSION 3.10)
project(ceres_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
message(STATUS "Ceres found: ${CERES_FOUND}")

add_executable(ceres_curve_fitting examples/ceres_curve_fitting.cpp)
target_link_libraries(ceres_curve_fitting Ceres::ceres)

add_executable(ceres_pose_graph examples/ceres_pose_graph.cpp)
target_link_libraries(ceres_pose_graph Ceres::ceres)

add_executable(ceres_bundle_adjustment examples/ceres_bundle_adjustment.cpp)
target_link_libraries(ceres_bundle_adjustment Ceres::ceres)

message(STATUS "")
message(STATUS "=== Ceres-Solver Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  ceres_curve_fitting     - fit y=exp(a x^2+b x+c)")
message(STATUS "  ceres_pose_graph        - 2D pose-graph optimization")
message(STATUS "  ceres_bundle_adjustment - bundle adjustment with BAL dataset")
message(STATUS "")
