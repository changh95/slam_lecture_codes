cmake_minimum_required(VERSION 3.14)
project(superpointglue)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

# Suppress warnings for cleaner output
add_definitions(-w)

# ==============================================================================
# Find Required Packages
# ==============================================================================

# OpenCV - for image I/O and visualization
find_package(OpenCV 4.2 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Eigen3 - for matrix operations
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include dirs: ${EIGEN3_INCLUDE_DIRS}")

# CUDA - for GPU operations
find_package(CUDA REQUIRED)
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")

# yaml-cpp - for configuration file parsing
find_package(yaml-cpp REQUIRED)
message(STATUS "yaml-cpp found")

# ==============================================================================
# TensorRT Configuration
# ==============================================================================

# TensorRT is typically installed in /usr/lib/x86_64-linux-gnu or via NVIDIA container
# Set TensorRT path if not found automatically
if(NOT DEFINED TENSORRT_ROOT)
    set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
endif()

# TensorRT include directories
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES include
    PATHS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include /usr/include
)

# TensorRT libraries
find_library(NVINFER_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib
)

find_library(NVONNXPARSER_LIBRARY nvonnxparser
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib
)

if(TENSORRT_INCLUDE_DIR AND NVINFER_LIBRARY AND NVONNXPARSER_LIBRARY)
    message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT nvinfer library: ${NVINFER_LIBRARY}")
    message(STATUS "TensorRT nvonnxparser library: ${NVONNXPARSER_LIBRARY}")
else()
    message(FATAL_ERROR "TensorRT not found. Please install TensorRT or set TENSORRT_ROOT.")
endif()

# ==============================================================================
# Third-party Libraries
# ==============================================================================

# TensorRT buffer management utility library
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)

# ==============================================================================
# Include Directories
# ==============================================================================

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
)

# ==============================================================================
# SuperPoint + SuperGlue Library
# ==============================================================================

add_library(${PROJECT_NAME}_lib SHARED
    src/super_point.cpp
    src/super_glue.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
    ${NVINFER_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    yaml-cpp
    tensorrtbuffer
)

# ==============================================================================
# Executables
# ==============================================================================

# Single image pair inference
add_executable(${PROJECT_NAME}_image examples/inference_image.cpp)
target_link_libraries(${PROJECT_NAME}_image ${PROJECT_NAME}_lib)

# Sequence/video inference
add_executable(${PROJECT_NAME}_sequence examples/inference_sequence.cpp)
target_link_libraries(${PROJECT_NAME}_sequence ${PROJECT_NAME}_lib)

# ==============================================================================
# Installation (optional)
# ==============================================================================

install(TARGETS ${PROJECT_NAME}_lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS ${PROJECT_NAME}_image ${PROJECT_NAME}_sequence
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
