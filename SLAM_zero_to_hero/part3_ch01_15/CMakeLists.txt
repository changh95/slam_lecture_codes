cmake_minimum_required(VERSION 3.10)
project(ceres_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Ceres
find_package(Ceres REQUIRED)

message(STATUS "Ceres found: ${CERES_FOUND}")

# Basic examples
add_executable(ceres_basics examples/ceres_basics.cpp)
target_link_libraries(ceres_basics Ceres::ceres)

# Bundle Adjustment
add_executable(ceres_bundle_adjustment examples/ceres_bundle_adjustment.cpp)
target_link_libraries(ceres_bundle_adjustment Ceres::ceres)

message(STATUS "")
message(STATUS "=== Ceres-Solver Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  ceres_basics            - Curve fitting and 2D pose graph")
message(STATUS "  ceres_bundle_adjustment - Bundle adjustment with BAL dataset")
message(STATUS "")
message(STATUS "Download BAL datasets from:")
message(STATUS "  https://grail.cs.washington.edu/projects/bal/")
message(STATUS "")
