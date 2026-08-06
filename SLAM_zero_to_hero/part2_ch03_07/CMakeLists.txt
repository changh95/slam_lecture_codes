cmake_minimum_required(VERSION 3.18)
project(AdvancedICP LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# MPI is required by VTK (used by PCL visualization)
find_package(MPI REQUIRED)

# Find PCL with required components
find_package(PCL 1.10 REQUIRED COMPONENTS
    common
    io
    filters
    features
    registration
    visualization
)

message(STATUS "PCL found: ${PCL_FOUND}")
message(STATUS "PCL version: ${PCL_VERSION}")

# Include directories for PCL
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP)

# ============================================
# Demos
# ============================================
# Three experiments, three demos:
#   gicp_demo    ICP vs GICP, and PCL vs small_gicp vs CUDA for the same GICP
#   ndt_demo     PCL NDT vs fast_gicp's CUDA NDT
#   teaser_demo  global registration where the local methods have no chance
add_executable(gicp_demo examples/gicp_demo.cpp)
target_link_libraries(gicp_demo ${PCL_LIBRARIES})

add_executable(ndt_demo examples/ndt_demo.cpp)
target_link_libraries(ndt_demo ${PCL_LIBRARIES})

# ============================================
# TEASER++ (required by teaser_demo)
# ============================================
# teaser_demo links the real TEASER++ solver and has no fallback path, so the
# target is simply not built when the library is missing - a stand-in
# implementation would only invite mistaking its numbers for TEASER++'s.
option(USE_TEASER "Build teaser_demo against TEASER++" ON)

if(USE_TEASER)
    find_package(teaserpp QUIET)
    if(teaserpp_FOUND)
        message(STATUS "TEASER++ found - building teaser_demo")
        add_executable(teaser_demo examples/teaser_demo.cpp)
        target_link_libraries(teaser_demo
            ${PCL_LIBRARIES}
            teaserpp::teaser_registration
        )
        # PCL's *EstimationOMP feature estimators pull in libgomp
        if(OpenMP_CXX_FOUND)
            target_link_libraries(teaser_demo OpenMP::OpenMP_CXX)
        endif()
    else()
        message(WARNING "TEASER++ not found - teaser_demo will NOT be built")
        message(STATUS "To install TEASER++:")
        message(STATUS "  git clone https://github.com/MIT-SPARK/TEASER-plusplus.git")
        message(STATUS "  cd TEASER-plusplus")
        message(STATUS "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF")
        message(STATUS "  cmake --build build -j8 --target install")
    endif()
else()
    message(STATUS "TEASER++ support disabled - teaser_demo will not be built")
endif()

set(ALL_TARGETS gicp_demo ndt_demo teaser_demo)

# ============================================
# small_gicp (CPU GICP)
# ============================================
find_package(small_gicp REQUIRED)
message(STATUS "small_gicp found: ${small_gicp_DIR}")

foreach(target ${ALL_TARGETS})
    if(TARGET ${target})
        target_link_libraries(${target} small_gicp::small_gicp)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${target} OpenMP::OpenMP_CXX)
        endif()
    endif()
endforeach()

# ============================================
# fast_gicp CUDA backends (NDTCuda, FastVGICPCuda)
# ============================================
# NDTCuda is the only NDT either library offers - small_gicp has none at all -
# and it is unconditionally CUDA-gated upstream, so the GPU NDT and a GPU build
# are the same decision.
#
# fast_gicp installs no CMake package config outside catkin/ament, so the library
# and headers are located by hand rather than with find_package.
option(USE_CUDA "Build the fast_gicp CUDA backends (NDTCuda, FastVGICPCuda)" ON)

# The flag exists to state the intent, not to make the GPU optional. The demos
# carry no CPU fallback for these backends - and NDTCuda is the only NDT either
# library provides, so experiment 2 has nothing at all to run without it.
if(NOT USE_CUDA)
    message(FATAL_ERROR
        "USE_CUDA=OFF is not supported: the demos have no CPU fallback for the "
        "fast_gicp CUDA backends, and NDTCuda is the only NDT available.")
endif()

# If any of these are missing, see the Dockerfile: upstream fast_gicp needs two
# patches to build against CUDA 12.x at all, and to target the right GPU.
find_library(FAST_GICP_LIB fast_gicp REQUIRED)
find_library(FAST_VGICP_CUDA_LIB fast_vgicp_cuda REQUIRED)
find_path(FAST_GICP_INCLUDE_DIR fast_gicp/ndt/ndt_cuda.hpp REQUIRED)
find_package(CUDAToolkit REQUIRED)

message(STATUS "fast_gicp CUDA found:")
message(STATUS "  ${FAST_GICP_LIB}")
message(STATUS "  ${FAST_VGICP_CUDA_LIB}")

foreach(target ${ALL_TARGETS})
    if(TARGET ${target})
        target_include_directories(${target} PRIVATE ${FAST_GICP_INCLUDE_DIR})
        target_link_libraries(${target}
            ${FAST_GICP_LIB}
            ${FAST_VGICP_CUDA_LIB}
            CUDA::cudart
        )
    endif()
endforeach()

# ============================================
# Compiler flags
# ============================================
foreach(target ${ALL_TARGETS})
    if(TARGET ${target})
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -Wall -Wextra)
        endif()
        if(CMAKE_BUILD_TYPE MATCHES "Release")
            target_compile_options(${target} PRIVATE -O3)
        endif()
    endif()
endforeach()

# ============================================
# Optional: rerun C++ SDK (live visualization)
# ============================================
find_package(rerun_sdk QUIET)
if(rerun_sdk_FOUND)
    message(STATUS "rerun C++ SDK found - demos stream results to a rerun viewer")
    foreach(target ${ALL_TARGETS})
        if(TARGET ${target})
            target_link_libraries(${target} rerun_sdk)
            target_compile_definitions(${target} PRIVATE HAVE_RERUN)
        endif()
    endforeach()
else()
    message(STATUS "rerun C++ SDK not found - demos run without live streaming")
endif()

# ============================================
# Build summary
# ============================================
message(STATUS "")
message(STATUS "=== Advanced ICP Methods Tutorial (KITTI) ===")
message(STATUS "Executables:")
message(STATUS "  - gicp_demo:    ICP vs GICP across PCL / small_gicp / CUDA")
message(STATUS "  - ndt_demo:     PCL NDT vs fast_gicp CUDA NDT")
if(TARGET teaser_demo)
    message(STATUS "  - teaser_demo:  TEASER++ global registration")
endif()
message(STATUS "")
message(STATUS "Usage (all demos read KITTI velodyne .bin scans):")
message(STATUS "  ./gicp_demo [source.bin target.bin]")
message(STATUS "  ./ndt_demo [source.bin target.bin [resolution]]")
message(STATUS "  ./teaser_demo [source.bin target.bin]")
message(STATUS "")
