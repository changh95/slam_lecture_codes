cmake_minimum_required(VERSION 3.10)
project(homography_bev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    features2d
    calib3d
    highgui
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Eigen3
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found")

# PoseLib (minimal pose estimation solvers)
find_package(PoseLib QUIET)
if(PoseLib_FOUND)
    message(STATUS "PoseLib found")
else()
    message(STATUS "PoseLib not found - homography_poselib will not be built")
endif()

# Homography Estimation Demo
add_executable(homography_demo examples/homography_demo.cpp)
target_link_libraries(homography_demo ${OpenCV_LIBS})

# BEV Projection (Inverse Perspective Mapping)
add_executable(bev_projection examples/bev_projection.cpp)
target_link_libraries(bev_projection ${OpenCV_LIBS})

# Lane Detection from BEV
add_executable(bev_lane_detection examples/bev_lane_detection.cpp)
target_link_libraries(bev_lane_detection ${OpenCV_LIBS})

# PoseLib Homography Demo
if(PoseLib_FOUND)
    add_executable(homography_poselib examples/homography_poselib.cpp)
    target_link_libraries(homography_poselib
        ${OpenCV_LIBS}
        Eigen3::Eigen
        PoseLib::PoseLib
    )
endif()

# Print build summary
message(STATUS "")
message(STATUS "=== Homography and BEV Projection Tutorial ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  homography_demo     - Homography estimation and decomposition")
message(STATUS "  bev_projection      - Bird's Eye View (IPM) transformation")
message(STATUS "  bev_lane_detection  - Lane detection from BEV")
if(PoseLib_FOUND)
    message(STATUS "  homography_poselib  - 4-point homography using PoseLib")
endif()
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./homography_demo")
message(STATUS "  ./bev_projection [input_image]")
message(STATUS "  ./bev_lane_detection [input_image]")
message(STATUS "")
