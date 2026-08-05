cmake_minimum_required(VERSION 3.10)
project(ICP_PCL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# MPI is required by VTK (used by PCL visualization)
find_package(MPI REQUIRED)

# Find PCL (with registration component)
find_package(PCL 1.10 REQUIRED COMPONENTS
    common
    io
    filters
    registration
    features
    visualization
    kdtree
    search
)

message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Basic Point-to-Point ICP executable
add_executable(icp_basic
    examples/icp_basic.cpp
)

target_link_libraries(icp_basic
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

# Point-to-Plane ICP executable
add_executable(icp_point_to_plane
    examples/icp_point_to_plane.cpp
)

target_link_libraries(icp_point_to_plane
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

# ICP Visualization executable
add_executable(icp_visualization
    examples/icp_visualization.cpp
)

target_link_libraries(icp_visualization
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

# LiDAR Odometry executable
add_executable(lidar_odometry
    examples/lidar_odometry.cpp
)

target_link_libraries(lidar_odometry
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(WARNING_FLAGS -Wall -Wextra -Wpedantic)
    target_compile_options(icp_basic PRIVATE ${WARNING_FLAGS})
    target_compile_options(icp_point_to_plane PRIVATE ${WARNING_FLAGS})
    target_compile_options(icp_visualization PRIVATE ${WARNING_FLAGS})
    target_compile_options(lidar_odometry PRIVATE ${WARNING_FLAGS})
endif()

# Optimization flags for Release build.
# Note: -march=native must NOT be used here — it changes Eigen's alignment
# (AVX-512 -> 64-byte) and breaks ABI with the system libpcl_registration,
# which segfaults inside IterativeClosestPoint::computeTransformation.
if(CMAKE_BUILD_TYPE MATCHES "Release")
    set(OPTIMIZATION_FLAGS -O3)
    target_compile_options(icp_basic PRIVATE ${OPTIMIZATION_FLAGS})
    target_compile_options(icp_point_to_plane PRIVATE ${OPTIMIZATION_FLAGS})
    target_compile_options(icp_visualization PRIVATE ${OPTIMIZATION_FLAGS})
    target_compile_options(lidar_odometry PRIVATE ${OPTIMIZATION_FLAGS})
endif()

# Print build info
message(STATUS "")
message(STATUS "=== ICP using PCL Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  - icp_basic: Point-to-point ICP registration")
message(STATUS "  - icp_point_to_plane: Point-to-plane ICP with normals")
message(STATUS "  - icp_visualization: Visualize ICP alignment process")
message(STATUS "  - lidar_odometry: Sequential scan registration for LiDAR odometry")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./icp_basic                  # Stanford bunny (data/bun_zipper_res3.ply)")
message(STATUS "  ./icp_point_to_plane         # Stanford bunny")
message(STATUS "  ./icp_visualization [--step] # Stanford bunny")
message(STATUS "  ./lidar_odometry <kitti_sequence_dir>")
message(STATUS "")
message(STATUS "Pass --help to any demo for the full option list.")
message(STATUS "")
