cmake_minimum_required(VERSION 3.10)
project(gtsam_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(GTSAM REQUIRED)
message(STATUS "GTSAM found: ${GTSAM_FOUND}")

add_executable(gtsam_curve_fitting examples/gtsam_curve_fitting.cpp)
target_link_libraries(gtsam_curve_fitting gtsam)

add_executable(gtsam_pose_graph examples/gtsam_pose_graph.cpp)
target_link_libraries(gtsam_pose_graph gtsam)

add_executable(gtsam_bundle_adjustment examples/gtsam_bundle_adjustment.cpp)
target_link_libraries(gtsam_bundle_adjustment gtsam)

message(STATUS "")
message(STATUS "=== GTSAM Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  gtsam_curve_fitting     - fit y=exp(a x^2+b x+c)")
message(STATUS "  gtsam_pose_graph        - 2D pose-graph optimization")
message(STATUS "  gtsam_bundle_adjustment - bundle adjustment with BAL dataset")
message(STATUS "")
