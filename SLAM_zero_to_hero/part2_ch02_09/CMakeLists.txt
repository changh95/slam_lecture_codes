cmake_minimum_required(VERSION 3.10)
project(APRILTAG_PNP_LOCALIZATION LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build type: Release (default)")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "")
message(STATUS "=== AprilTag Detection + PnP Camera Localization ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================
# DEPENDENCIES
# ============================================================

find_package(OpenCV 4.0 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    videoio
    calib3d
)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# AprilTag (https://github.com/AprilRobotics/apriltag)
find_package(apriltag REQUIRED)
if(TARGET apriltag::apriltag)
    set(APRILTAG_TARGET apriltag::apriltag)
else()
    set(APRILTAG_TARGET apriltag)
endif()
message(STATUS "AprilTag found (target: ${APRILTAG_TARGET})")

# PoseLib (optional - pnp_poselib is built only when found)
find_package(PoseLib QUIET)
if(PoseLib_FOUND)
    message(STATUS "PoseLib found")
else()
    message(STATUS "PoseLib not found - pnp_poselib will not be built")
endif()

# OpenGV (optional - pnp_opengv is built only when found)
find_package(opengv QUIET)
if(opengv_FOUND)
    message(STATUS "OpenGV found")
else()
    message(STATUS "OpenGV not found - pnp_opengv will not be built")
endif()

# ============================================================
# COMMON LIBRARY - shared AprilTag + PnP pipeline
# ============================================================

add_library(apriltag_pnp_common STATIC src/apriltag_pnp.cpp)
target_include_directories(apriltag_pnp_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(apriltag_pnp_common PUBLIC
    ${OpenCV_LIBS}
    ${APRILTAG_TARGET}
)

# ============================================================
# EXAMPLES - one executable per PnP backend
# ============================================================

# 1. OpenCV: solvePnPRansac (EPnP) + solvePnPRefineLM
add_executable(pnp_opencv examples/pnp_opencv.cpp)
target_link_libraries(pnp_opencv PRIVATE apriltag_pnp_common)

# 2. PoseLib: estimate_absolute_pose (P3P LO-RANSAC + refinement)
if(PoseLib_FOUND)
    add_executable(pnp_poselib examples/pnp_poselib.cpp)
    target_link_libraries(pnp_poselib PRIVATE
        apriltag_pnp_common
        PoseLib::PoseLib
        Eigen3::Eigen
    )
endif()

# 3. OpenGV: Kneip P3P RANSAC + optimize_nonlinear
if(opengv_FOUND)
    add_executable(pnp_opengv examples/pnp_opengv.cpp)
    target_link_libraries(pnp_opengv PRIVATE
        apriltag_pnp_common
        opengv
        Eigen3::Eigen
    )
    # OpenGV compiles itself with -march=native; build this translation unit
    # with the same flags so Eigen object alignment matches across the
    # library boundary (mismatch crashes inside p3p_kneip).
    target_compile_options(pnp_opengv PRIVATE -O3 -march=native)
endif()

# ============================================================
# SUMMARY
# ============================================================

message(STATUS "")
message(STATUS "=== Executables ===")
message(STATUS "  pnp_opencv   - AprilTag + OpenCV solvePnPRansac/RefineLM")
if(PoseLib_FOUND)
    message(STATUS "  pnp_poselib  - AprilTag + PoseLib estimate_absolute_pose")
endif()
if(opengv_FOUND)
    message(STATUS "  pnp_opengv   - AprilTag + OpenGV Kneip P3P RANSAC")
endif()
message(STATUS "")
