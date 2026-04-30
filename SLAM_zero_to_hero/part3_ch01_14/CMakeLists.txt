cmake_minimum_required(VERSION 3.10)
project(gtsam_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find GTSAM
find_package(GTSAM REQUIRED)

message(STATUS "GTSAM found: ${GTSAM_FOUND}")

# 2D Pose Graph example
add_executable(gtsam_basics examples/gtsam_basics.cpp)
target_link_libraries(gtsam_basics gtsam)

# Bundle Adjustment example
add_executable(gtsam_bundle_adjustment examples/gtsam_bundle_adjustment.cpp)
target_link_libraries(gtsam_bundle_adjustment gtsam)

message(STATUS "")
message(STATUS "=== GTSAM Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  gtsam_basics           - 2D pose graph optimization")
message(STATUS "  gtsam_bundle_adjustment - Bundle adjustment with BAL dataset")
message(STATUS "")
message(STATUS "Download BAL datasets from:")
message(STATUS "  https://grail.cs.washington.edu/projects/bal/")
message(STATUS "")
