cmake_minimum_required(VERSION 3.10)
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
message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")

# Include directories for PCL
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ============================================
# Core Examples (PCL only - always built)
# ============================================

# GICP Demo - Generalized ICP
add_executable(gicp_demo examples/gicp_demo.cpp)
target_link_libraries(gicp_demo ${PCL_LIBRARIES})

# NDT Demo - Normal Distributions Transform
add_executable(ndt_demo examples/ndt_demo.cpp)
target_link_libraries(ndt_demo ${PCL_LIBRARIES})

# Method Comparison - ICP vs GICP vs NDT
add_executable(method_comparison examples/method_comparison.cpp)
target_link_libraries(method_comparison ${PCL_LIBRARIES})

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
        find_package(OpenMP)
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

# ============================================
# Compiler flags
# ============================================
set(ALL_TARGETS gicp_demo ndt_demo method_comparison teaser_demo)

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
message(STATUS "  - gicp_demo:          Generalized ICP (plane-to-plane)")
message(STATUS "  - ndt_demo:           Normal Distributions Transform")
message(STATUS "  - method_comparison:  Compare ICP vs GICP vs NDT")
if(TARGET teaser_demo)
    message(STATUS "  - teaser_demo:        TEASER++ global registration")
endif()
message(STATUS "")
message(STATUS "Usage (all demos read KITTI velodyne .bin scans):")
message(STATUS "  ./gicp_demo [source.bin target.bin]")
message(STATUS "  ./ndt_demo [source.bin target.bin [resolution]]")
message(STATUS "  ./method_comparison [source.bin target.bin]")
message(STATUS "  ./teaser_demo [source.bin target.bin]")
message(STATUS "")
