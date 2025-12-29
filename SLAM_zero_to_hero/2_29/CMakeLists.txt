cmake_minimum_required(VERSION 3.10)
project(ICP_PCL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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

# Optimization flags for Release build
if(CMAKE_BUILD_TYPE MATCHES "Release")
    set(OPTIMIZATION_FLAGS -O3 -march=native)
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
message(STATUS "  ./icp_basic source.pcd target.pcd")
message(STATUS "  ./icp_point_to_plane source.pcd target.pcd")
message(STATUS "  ./icp_visualization source.pcd target.pcd")
message(STATUS "  ./lidar_odometry /path/to/velodyne/")
message(STATUS "")
