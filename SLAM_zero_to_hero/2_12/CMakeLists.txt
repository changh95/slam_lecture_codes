cmake_minimum_required(VERSION 3.10)
project(epipolar_geometry LANGUAGES CXX)

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
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# Find Eigen3 (required for PoseLib and OpenGV)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(-DHAVE_EIGEN)

# PoseLib (minimal pose estimation solvers)
find_package(PoseLib QUIET)
if(PoseLib_FOUND)
    message(STATUS "PoseLib found")
else()
    message(STATUS "PoseLib not found - relpose_poselib will not be built")
endif()

# OpenGV (geometric vision algorithms)
find_package(opengv QUIET)
if(opengv_FOUND)
    message(STATUS "OpenGV found")
else()
    message(STATUS "OpenGV not found - relpose_opengv will not be built")
endif()

# Essential and Fundamental Matrix Estimation
add_executable(essential_fundamental_demo examples/essential_fundamental_demo.cpp)
target_link_libraries(essential_fundamental_demo ${OpenCV_LIBS})

# Pose Recovery from Essential Matrix
add_executable(pose_recovery examples/pose_recovery.cpp)
target_link_libraries(pose_recovery ${OpenCV_LIBS})

# Epipolar Line Visualization
add_executable(epipolar_visualization examples/epipolar_visualization.cpp)
target_link_libraries(epipolar_visualization ${OpenCV_LIBS})

# PoseLib 5-point relative pose solver
if(PoseLib_FOUND)
    add_executable(relpose_poselib examples/relpose_poselib.cpp)
    target_link_libraries(relpose_poselib
        ${OpenCV_LIBS}
        Eigen3::Eigen
        PoseLib::PoseLib
    )
endif()

# OpenGV 5-point relative pose solvers
if(opengv_FOUND)
    add_executable(relpose_opengv examples/relpose_opengv.cpp)
    target_link_libraries(relpose_opengv
        ${OpenCV_LIBS}
        Eigen3::Eigen
        opengv
    )
endif()

# Print build summary
message(STATUS "")
message(STATUS "=== Epipolar Geometry Tutorial ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  essential_fundamental_demo - Essential & Fundamental matrix estimation")
message(STATUS "  pose_recovery             - Recover rotation and translation from E")
message(STATUS "  epipolar_visualization    - Visualize epipolar lines")
if(PoseLib_FOUND)
    message(STATUS "  relpose_poselib           - 5-point solver using PoseLib")
endif()
if(opengv_FOUND)
    message(STATUS "  relpose_opengv            - 5-point solver using OpenGV")
endif()
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./essential_fundamental_demo")
message(STATUS "  ./pose_recovery image1.jpg image2.jpg")
message(STATUS "  ./epipolar_visualization image1.jpg image2.jpg")
message(STATUS "")
