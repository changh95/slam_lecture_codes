cmake_minimum_required(VERSION 3.14)
project("ransac_usac_tutorial" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (required)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Check for USAC support (OpenCV 4.5+)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.5.0")
    message(STATUS "OpenCV ${OpenCV_VERSION} supports USAC framework")
    add_definitions(-DHAVE_USAC)
else()
    message(WARNING "OpenCV ${OpenCV_VERSION} does not fully support USAC. Some examples may not work.")
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Example 1: Homography estimation with various RANSAC methods
add_executable(ransac_homography examples/ransac_homography.cpp)
target_link_libraries(ransac_homography ${OpenCV_LIBS})

# Example 2: Fundamental matrix estimation with USAC
add_executable(ransac_fundamental examples/ransac_fundamental.cpp)
target_link_libraries(ransac_fundamental ${OpenCV_LIBS})

# Example 3: Custom RANSAC implementation
add_executable(ransac_custom examples/ransac_custom.cpp)
target_link_libraries(ransac_custom ${OpenCV_LIBS})

# Install targets (optional)
install(TARGETS ransac_homography ransac_fundamental ransac_custom
        RUNTIME DESTINATION bin)
