cmake_minimum_required(VERSION 3.10)
project(homography_vslam LANGUAGES CXX)

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
    message(STATUS "PoseLib not found - PoseLib examples will not be built")
endif()

# =============================================================================
# OpenCV Examples
# =============================================================================

# Homography Demo (basic estimation and decomposition)
add_executable(homography_demo examples/homography_demo.cpp)
target_link_libraries(homography_demo ${OpenCV_LIBS})

# H/F Model Selection (ORB-SLAM style)
add_executable(hf_model_selection examples/hf_model_selection.cpp)
target_link_libraries(hf_model_selection ${OpenCV_LIBS})

# Image Stitching (panorama)
add_executable(image_stitching examples/image_stitching.cpp)
target_link_libraries(image_stitching ${OpenCV_LIBS})

# =============================================================================
# PoseLib Examples
# =============================================================================

if(PoseLib_FOUND)
    # H/F Model Selection with PoseLib
    add_executable(hf_model_selection_poselib examples/hf_model_selection_poselib.cpp)
    target_link_libraries(hf_model_selection_poselib
        ${OpenCV_LIBS}
        Eigen3::Eigen
        PoseLib::PoseLib
    )

    # Image Stitching with PoseLib
    add_executable(image_stitching_poselib examples/image_stitching_poselib.cpp)
    target_link_libraries(image_stitching_poselib
        ${OpenCV_LIBS}
        Eigen3::Eigen
        PoseLib::PoseLib
    )
endif()

# =============================================================================
# Build Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Homography for Visual SLAM ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "OpenCV Executables:")
message(STATUS "  homography_demo      - Homography estimation and decomposition")
message(STATUS "  hf_model_selection   - H/F model selection (ORB-SLAM style)")
message(STATUS "  image_stitching      - Image stitching / panorama")
if(PoseLib_FOUND)
    message(STATUS "")
    message(STATUS "PoseLib Executables:")
    message(STATUS "  hf_model_selection_poselib  - H/F model selection with PoseLib")
    message(STATUS "  image_stitching_poselib     - Image stitching with PoseLib")
endif()
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./homography_demo")
message(STATUS "  ./hf_model_selection")
message(STATUS "  ./image_stitching [img1.jpg img2.jpg]")
message(STATUS "")
