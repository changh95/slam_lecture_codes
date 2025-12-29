cmake_minimum_required(VERSION 3.10)
project(AdvancedICP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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
# Optional: TEASER++ support
# ============================================
option(USE_TEASER "Build with TEASER++ support" OFF)

if(USE_TEASER)
    find_package(teaserpp QUIET)
    if(teaserpp_FOUND)
        message(STATUS "TEASER++ found - building teaser_demo")
        add_executable(teaser_demo examples/teaser_demo.cpp)
        target_link_libraries(teaser_demo
            ${PCL_LIBRARIES}
            teaserpp::teaser_registration
        )
        target_compile_definitions(teaser_demo PRIVATE WITH_TEASER)
    else()
        message(STATUS "TEASER++ not found - teaser_demo will not be built")
        message(STATUS "To install TEASER++:")
        message(STATUS "  git clone https://github.com/MIT-SPARK/TEASER-plusplus.git")
        message(STATUS "  cd TEASER-plusplus && mkdir build && cd build")
        message(STATUS "  cmake .. && make && sudo make install")
    endif()
else()
    message(STATUS "TEASER++ support disabled (enable with -DUSE_TEASER=ON)")
    # Build stub version without TEASER++ dependency
    add_executable(teaser_demo examples/teaser_demo.cpp)
    target_link_libraries(teaser_demo ${PCL_LIBRARIES})
endif()

# ============================================
# Optional: KISS-ICP support
# ============================================
option(USE_KISS_ICP "Build with KISS-ICP support" OFF)

if(USE_KISS_ICP)
    find_package(kiss_icp QUIET)
    if(kiss_icp_FOUND)
        message(STATUS "KISS-ICP found - building kiss_icp_demo")
        add_executable(kiss_icp_demo examples/kiss_icp_demo.cpp)
        target_link_libraries(kiss_icp_demo
            ${PCL_LIBRARIES}
            kiss_icp::kiss_icp_pipeline
        )
        target_compile_definitions(kiss_icp_demo PRIVATE WITH_KISS_ICP)
    else()
        message(STATUS "KISS-ICP not found - kiss_icp_demo will not be built")
        message(STATUS "To install KISS-ICP:")
        message(STATUS "  pip install kiss-icp")
        message(STATUS "  # Or from source:")
        message(STATUS "  git clone https://github.com/PRBonn/kiss-icp.git")
        message(STATUS "  cd kiss-icp/cpp && mkdir build && cd build")
        message(STATUS "  cmake .. && make && sudo make install")
    endif()
else()
    message(STATUS "KISS-ICP support disabled (enable with -DUSE_KISS_ICP=ON)")
    # Build stub version without KISS-ICP dependency
    add_executable(kiss_icp_demo examples/kiss_icp_demo.cpp)
    target_link_libraries(kiss_icp_demo ${PCL_LIBRARIES})
endif()

# ============================================
# Compiler flags
# ============================================
set(ALL_TARGETS gicp_demo ndt_demo method_comparison teaser_demo kiss_icp_demo)

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
# Build summary
# ============================================
message(STATUS "")
message(STATUS "=== Advanced ICP Methods Tutorial ===")
message(STATUS "Core executables (always built):")
message(STATUS "  - gicp_demo:          Generalized ICP (plane-to-plane)")
message(STATUS "  - ndt_demo:           Normal Distributions Transform")
message(STATUS "  - method_comparison:  Compare ICP vs GICP vs NDT")
message(STATUS "")
message(STATUS "Optional executables:")
message(STATUS "  - teaser_demo:        TEASER++ global registration (USE_TEASER=${USE_TEASER})")
message(STATUS "  - kiss_icp_demo:      KISS-ICP odometry (USE_KISS_ICP=${USE_KISS_ICP})")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./gicp_demo <source.pcd> <target.pcd>")
message(STATUS "  ./ndt_demo <source.pcd> <target.pcd> [resolution]")
message(STATUS "  ./method_comparison <source.pcd> <target.pcd>")
message(STATUS "  ./teaser_demo <source.pcd> <target.pcd>")
message(STATUS "  ./kiss_icp_demo <sequence_dir>")
message(STATUS "")
