cmake_minimum_required(VERSION 3.10)
project(spatial_data_structures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find PCL
find_package(PCL REQUIRED COMPONENTS common io octree visualization)
if(PCL_FOUND)
    message(STATUS "Found PCL: ${PCL_VERSION}")
    message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else()
    message(FATAL_ERROR "PCL not found!")
endif()

# Find OctoMap
find_package(octomap REQUIRED)
if(octomap_FOUND)
    message(STATUS "Found OctoMap: ${OCTOMAP_VERSION}")
    message(STATUS "OctoMap include dirs: ${OCTOMAP_INCLUDE_DIRS}")
    include_directories(${OCTOMAP_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OctoMap not found!")
endif()

# Bonxai is header-only
set(BONXAI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
if(EXISTS "${BONXAI_INCLUDE_DIR}/bonxai/bonxai.hpp")
    message(STATUS "Found Bonxai headers in: ${BONXAI_INCLUDE_DIR}")
    include_directories(${BONXAI_INCLUDE_DIR})
else()
    message(WARNING "Bonxai headers not found at ${BONXAI_INCLUDE_DIR}/bonxai/bonxai.hpp")
    message(WARNING "Bonxai demo will not be built. Clone Bonxai to 3rdparty/bonxai/")
endif()

# PCL Octree Demo
add_executable(octree_demo examples/octree_demo.cpp)
target_link_libraries(octree_demo
    ${PCL_LIBRARIES}
)

# OctoMap Demo
add_executable(octomap_demo examples/octomap_demo.cpp)
target_link_libraries(octomap_demo
    ${PCL_LIBRARIES}
    ${OCTOMAP_LIBRARIES}
)

# OctoMap Navigation Demo
add_executable(octomap_navigation examples/octomap_navigation.cpp)
target_link_libraries(octomap_navigation
    ${OCTOMAP_LIBRARIES}
)

# Bonxai Demo (only if headers exist)
if(EXISTS "${BONXAI_INCLUDE_DIR}/bonxai/bonxai.hpp")
    add_executable(bonxai_demo examples/bonxai_demo.cpp)
    target_link_libraries(bonxai_demo
        ${PCL_LIBRARIES}
    )

    # Comparison of all methods
    add_executable(comparison examples/comparison.cpp)
    target_link_libraries(comparison
        ${PCL_LIBRARIES}
        ${OCTOMAP_LIBRARIES}
    )
else()
    message(STATUS "Skipping bonxai_demo and comparison (Bonxai headers not found)")
endif()

# Print build summary
message(STATUS "")
message(STATUS "=== Spatial Data Structures Tutorial ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  octree_demo         - PCL Octree spatial search demo")
message(STATUS "  octomap_demo        - OctoMap probabilistic mapping")
message(STATUS "  octomap_navigation  - Path planning with OctoMap")
if(EXISTS "${BONXAI_INCLUDE_DIR}/bonxai/bonxai.hpp")
    message(STATUS "  bonxai_demo         - Bonxai high-performance voxels")
    message(STATUS "  comparison          - Compare all three methods")
endif()
message(STATUS "")
