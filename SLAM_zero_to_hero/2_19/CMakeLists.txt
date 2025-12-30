cmake_minimum_required(VERSION 3.10)
project(PNP_AND_MARKER_TRACKING LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build type: Release (default)")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "")
message(STATUS "=== PnP and Fiducial Marker Tracking ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# ============================================================
# DEPENDENCIES
# ============================================================

# OpenCV (aruco is part of objdetect in OpenCV 4.7+)
find_package(OpenCV 4.0 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    videoio
    highgui
    calib3d
    objdetect
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Eigen3 (for pose transformations)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# ============================================================
# EXAMPLES - Main tutorial examples
# ============================================================

# 1. PnP Demo - Compare different PnP algorithms (P3P, EPnP, DLS, etc.)
add_executable(pnp_demo examples/pnp_demo.cpp)
target_link_libraries(pnp_demo PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
target_include_directories(pnp_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# 2. PnP RANSAC Demo - Robust PnP with outliers
add_executable(pnp_ransac_demo examples/pnp_ransac_demo.cpp)
target_link_libraries(pnp_ransac_demo PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
target_include_directories(pnp_ransac_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# 3. Marker Detection - ArUco marker detection and generation
add_executable(marker_detection examples/marker_detection.cpp)
target_link_libraries(marker_detection PRIVATE ${OpenCV_LIBS})
target_include_directories(marker_detection PRIVATE ${OpenCV_INCLUDE_DIRS})

# 4. Marker Pose Estimation - 6-DoF pose from ArUco markers
add_executable(marker_pose_estimation examples/marker_pose_estimation.cpp)
target_link_libraries(marker_pose_estimation PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
target_include_directories(marker_pose_estimation PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# 5. ChArUco Calibration - Camera calibration using ChArUco boards
add_executable(charuco_calibration examples/charuco_calibration.cpp)
target_link_libraries(charuco_calibration PRIVATE ${OpenCV_LIBS})
target_include_directories(charuco_calibration PRIVATE ${OpenCV_INCLUDE_DIRS})

# ============================================================
# SRC - Additional utilities (from original implementation)
# ============================================================

# Check if src files exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/marker_generator.cpp")
    add_executable(marker_generator src/marker_generator.cpp)
    target_link_libraries(marker_generator PRIVATE ${OpenCV_LIBS})
    target_include_directories(marker_generator PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/marker_detector.cpp")
    add_executable(marker_detector src/marker_detector.cpp)
    target_link_libraries(marker_detector PRIVATE ${OpenCV_LIBS})
    target_include_directories(marker_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/pose_estimator.cpp")
    add_executable(pose_estimator src/pose_estimator.cpp)
    target_link_libraries(pose_estimator PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
    )
    target_include_directories(pose_estimator PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/charuco_calibration.cpp")
    add_executable(charuco_calibration_src src/charuco_calibration.cpp)
    target_link_libraries(charuco_calibration_src PRIVATE ${OpenCV_LIBS})
    target_include_directories(charuco_calibration_src PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# ============================================================
# SUMMARY
# ============================================================

message(STATUS "")
message(STATUS "=== Executables ===")
message(STATUS "")
message(STATUS "PnP Examples:")
message(STATUS "  pnp_demo              - Compare PnP algorithms (P3P, EPnP, DLS, SQPnP)")
message(STATUS "  pnp_ransac_demo       - Robust PnP estimation with RANSAC")
message(STATUS "")
message(STATUS "Marker Examples:")
message(STATUS "  marker_detection      - ArUco marker detection and generation")
message(STATUS "  marker_pose_estimation- 6-DoF pose estimation from markers")
message(STATUS "  charuco_calibration   - Camera calibration with ChArUco board")
message(STATUS "")
