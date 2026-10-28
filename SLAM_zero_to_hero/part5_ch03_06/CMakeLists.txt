cmake_minimum_required(VERSION 3.10)
project(SIMD_ACCELERATION LANGUAGES C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# komrad36's kernels are hand-written x86 intrinsics (AVX2, SSE4.1, BMI1)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(FATAL_ERROR "This exercise needs an x86-64 CPU with AVX2 (found ${CMAKE_SYSTEM_PROCESSOR})")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV 4.x")
endif()

# Find easy_profiler (timing) and threads (the FastThreadPool workers of the _MT variants)
find_package(easy_profiler REQUIRED)
find_package(Threads REQUIRED)

# Plain C/C++ baselines (per-pixel BGR->gray, Rosten's FAST-9, ORB's IC_Angle loop).
# The same code is compiled twice, so the tables show what the compiler alone does with it:
#   plain_o3     -O3                 portable x86-64 Release build: the reference (the -O3 column)
#   plain_native -O3 -march=native   GCC may use every extension of the build CPU (AVX2, AVX-512, ...)
# Rosten's functions get a per-copy prefix (include/rosten_prefix.h) so both copies link together.
set(PLAIN_FLAGS_o3 -O3)
set(PLAIN_FLAGS_native -O3 -march=native)
foreach(variant o3 native)
    add_library(plain_${variant} STATIC
        src/plain_kernels.cpp
        third_party/komrad36/KFAST/Rosten/fast_9.c
        third_party/komrad36/KFAST/Rosten/nonmax.c
    )
    target_include_directories(plain_${variant} PRIVATE include third_party/komrad36 ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(plain_${variant} PRIVATE PLAIN_NAMESPACE=plain_${variant} ROSTEN_PREFIX=plain_${variant}_)
    target_compile_options(plain_${variant} PRIVATE ${PLAIN_FLAGS_${variant}}
        "$<$<COMPILE_LANGUAGE:C>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/include/rosten_prefix.h>")
endforeach()

# Instruction sets komrad36's kernels use. -mavx2 implies SSE4.1/SSSE3 (FeatureAngle);
# -mbmi enables _tzcnt_u32/_blsr_u32 (KFAST).
set(SIMD_FLAGS -mavx2 -mbmi)

set(EXAMPLES rgb2y_benchmark kfast_benchmark feature_angle_benchmark)
foreach(example ${EXAMPLES})
    add_executable(${example} examples/${example}.cpp)
    target_include_directories(${example} PRIVATE include third_party/komrad36 ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(${example} PRIVATE SIMD_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    target_link_libraries(${example} plain_o3 plain_native ${OpenCV_LIBS} easy_profiler Threads::Threads)
    if(CMAKE_BUILD_TYPE MATCHES "Release")
        target_compile_options(${example} PRIVATE -O3 ${SIMD_FLAGS})
    else()
        target_compile_options(${example} PRIVATE ${SIMD_FLAGS})
    endif()
endforeach()

# Re-print the comparison tables of any .prof file (e.g. one recorded on another machine)
add_executable(prof_summary examples/prof_summary.cpp)
target_include_directories(prof_summary PRIVATE include)
target_link_libraries(prof_summary easy_profiler)

# Print summary
message(STATUS "")
message(STATUS "=== SIMD Acceleration Hands-on (plain C/C++ vs OpenCV vs komrad36) ===")
message(STATUS "Executables:")
message(STATUS "  - rgb2y_benchmark: BGR->gray, plain loop vs cv::cvtColor vs RGB2Y (AVX2)")
message(STATUS "  - kfast_benchmark: FAST-9 + NMS, Rosten (plain C) vs cv::FAST vs KFAST (AVX2)")
message(STATUS "  - feature_angle_benchmark: keypoint angle, ORB IC_Angle loop vs FeatureAngle (SSE)")
message(STATUS "  - prof_summary: comparison tables from any .prof file")
message(STATUS "")
message(STATUS "Build with: mkdir build && cd build && cmake .. && make -j")
message(STATUS "")
