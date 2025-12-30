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

# Find easy_profiler (for feature_profiling)
find_package(easy_profiler REQUIRED)
message(STATUS "easy_profiler found: ${easy_profiler_FOUND}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Feature Detection Example
add_executable(feature_detection examples/feature_detection.cpp)
target_link_libraries(feature_detection ${OpenCV_LIBS})

# Feature Matching Example
add_executable(feature_matching examples/feature_matching.cpp)
target_link_libraries(feature_matching ${OpenCV_LIBS})

# Feature Profiling Example (ORB vs SIFT vs FAST+TEBLID)
add_executable(feature_profiling examples/feature_profiling.cpp)
target_link_libraries(feature_profiling ${OpenCV_LIBS} easy_profiler)

# Compiler optimizations for Release build
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(feature_detection PRIVATE -O3)
    target_compile_options(feature_matching PRIVATE -O3)
    # Use -O2 -g for profiling to keep symbols
    target_compile_options(feature_profiling PRIVATE -O2 -g)
endif()

# Print summary
message(STATUS "")
message(STATUS "=== Local Features Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  - feature_detection: FAST, ORB, SIFT detection demo")
message(STATUS "  - feature_matching: BFMatcher, FLANN matching demo")
message(STATUS "  - feature_profiling: ORB vs SIFT vs FAST+TEBLID profiling (easy_profiler)")
message(STATUS "")
message(STATUS "Build with: mkdir build && cd build && cmake .. && make -j")
message(STATUS "View profiling: profiler_gui feature_profiling.prof")
message(STATUS "")
