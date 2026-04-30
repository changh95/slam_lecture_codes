cmake_minimum_required(VERSION 3.10)
project(MONOCULAR_VO LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find required packages
find_package(OpenCV 4.0 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
    video
    calib3d
    features2d
)
find_package(Eigen3 QUIET)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "Eigen3 not found (optional)")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# ============================================================================
# MonocularVO Library
# ============================================================================
add_library(monocular_vo STATIC
    src/monocular_vo.cpp
)

target_link_libraries(monocular_vo
    ${OpenCV_LIBS}
)

target_include_directories(monocular_vo PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================================
# Executables
# ============================================================================

# Feature Tracking Demo
add_executable(feature_tracking_demo
    examples/feature_tracking_demo.cpp
)

target_link_libraries(feature_tracking_demo
    ${OpenCV_LIBS}
)

# Monocular VO (standalone)
add_executable(monocular_vo_demo
    examples/monocular_vo.cpp
)

target_link_libraries(monocular_vo_demo
    ${OpenCV_LIBS}
)

# Run VO on KITTI (uses library)
add_executable(run_vo_kitti
    examples/run_vo_kitti.cpp
)

target_link_libraries(run_vo_kitti
    monocular_vo
    ${OpenCV_LIBS}
)

# ============================================================================
# Compiler flags
# ============================================================================
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(monocular_vo PRIVATE -O3)
    target_compile_options(feature_tracking_demo PRIVATE -O3)
    target_compile_options(monocular_vo_demo PRIVATE -O3)
    target_compile_options(run_vo_kitti PRIVATE -O3)
endif()

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Monocular Visual Odometry Tutorial ===")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  - feature_tracking_demo : Feature detection and KLT tracking demo")
message(STATUS "  - monocular_vo_demo     : Standalone monocular VO pipeline")
message(STATUS "  - run_vo_kitti          : VO on KITTI dataset with ground truth")
message(STATUS "")
message(STATUS "After building, run:")
message(STATUS "  ./feature_tracking_demo <image_dir> [num_frames]")
message(STATUS "  ./monocular_vo_demo <image_dir> [focal] [cx] [cy]")
message(STATUS "  ./run_vo_kitti <image_dir> <poses_file> [max_frames]")
message(STATUS "")
message(STATUS "Example with KITTI sequence 00:")
message(STATUS "  ./run_vo_kitti /data/sequences/00/image_0 /data/poses/00.txt")
message(STATUS "")
