cmake_minimum_required(VERSION 3.10)
project(pcl_point_cloud_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find PCL with required components
find_package(PCL 1.10 REQUIRED COMPONENTS
    common
    io
    filters
    features
    segmentation
    visualization
)

message(STATUS "PCL found: ${PCL_FOUND}")
message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Basic I/O - Reading/writing PCD and PLY files
add_executable(basic_io examples/basic_io.cpp)
target_link_libraries(basic_io ${PCL_LIBRARIES})

# Filtering - VoxelGrid, StatisticalOutlierRemoval, PassThrough
add_executable(filtering examples/filtering.cpp)
target_link_libraries(filtering ${PCL_LIBRARIES})

# Normal Estimation - Computing surface normals
add_executable(normal_estimation examples/normal_estimation.cpp)
target_link_libraries(normal_estimation ${PCL_LIBRARIES})

# Segmentation - RANSAC plane segmentation and Euclidean clustering
add_executable(segmentation examples/segmentation.cpp)
target_link_libraries(segmentation ${PCL_LIBRARIES})

# Visualization - PCL visualizer
add_executable(visualization examples/visualization.cpp)
target_link_libraries(visualization ${PCL_LIBRARIES})

message(STATUS "")
message(STATUS "=== Point Cloud Processing Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  basic_io          - Read/write PCD and PLY files")
message(STATUS "  filtering         - VoxelGrid, SOR, PassThrough filters")
message(STATUS "  normal_estimation - Compute surface normals")
message(STATUS "  segmentation      - RANSAC plane, Euclidean clustering")
message(STATUS "  visualization     - PCL visualization demo")
message(STATUS "")
