cmake_minimum_required(VERSION 3.10)
project(g2o_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find g2o
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "g2o found: ${g2o_FOUND}")
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# 2D Pose Graph example
add_executable(g2o_basics examples/g2o_basics.cpp)
target_link_libraries(g2o_basics
    g2o::core
    g2o::stuff
    g2o::types_slam2d
    g2o::solver_dense
    fmt
)

# Bundle Adjustment example
add_executable(g2o_bundle_adjustment examples/g2o_bundle_adjustment.cpp)
target_link_libraries(g2o_bundle_adjustment
    g2o::core
    g2o::stuff
    g2o::types_sba
    g2o::solver_dense
    g2o::solver_eigen
    fmt
)

message(STATUS "")
message(STATUS "=== g2o Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  g2o_basics           - 2D pose graph optimization")
message(STATUS "  g2o_bundle_adjustment - Bundle adjustment with BAL dataset")
message(STATUS "")
message(STATUS "Download BAL datasets from:")
message(STATUS "  https://grail.cs.washington.edu/projects/bal/")
message(STATUS "")
