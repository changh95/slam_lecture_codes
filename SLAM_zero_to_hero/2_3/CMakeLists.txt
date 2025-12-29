cmake_minimum_required(VERSION 3.10)
project(LOCAL_FEATURES LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV 4.x")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Feature Detection Example
add_executable(feature_detection examples/feature_detection.cpp)
target_link_libraries(feature_detection ${OpenCV_LIBS})

# Feature Matching Example
add_executable(feature_matching examples/feature_matching.cpp)
target_link_libraries(feature_matching ${OpenCV_LIBS})

# Compiler optimizations for Release build
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(feature_detection PRIVATE -O3)
    target_compile_options(feature_matching PRIVATE -O3)
endif()

# Print summary
message(STATUS "")
message(STATUS "=== Local Features Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  - feature_detection: FAST, ORB, SIFT detection demo")
message(STATUS "  - feature_matching: BFMatcher, FLANN matching demo")
message(STATUS "")
message(STATUS "Build with: mkdir build && cd build && cmake .. && make -j")
message(STATUS "")
