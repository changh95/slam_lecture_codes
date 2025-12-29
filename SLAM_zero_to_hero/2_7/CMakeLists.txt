cmake_minimum_required(VERSION 3.10)
project(OPTICAL_FLOW LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find required packages
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Common include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Sparse optical flow example (Lucas-Kanade feature tracking)
add_executable(sparse_optical_flow examples/sparse_optical_flow.cpp)
target_link_libraries(sparse_optical_flow ${OpenCV_LIBS})

# Dense optical flow example (Farneback method)
add_executable(dense_optical_flow examples/dense_optical_flow.cpp)
target_link_libraries(dense_optical_flow ${OpenCV_LIBS})

# Compiler flags for better performance
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(sparse_optical_flow PRIVATE -O3)
    target_compile_options(dense_optical_flow PRIVATE -O3)
endif()

message(STATUS "")
message(STATUS "=== Optical Flow Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  - sparse_optical_flow : Lucas-Kanade feature tracking")
message(STATUS "  - dense_optical_flow  : Farneback dense flow")
message(STATUS "")
