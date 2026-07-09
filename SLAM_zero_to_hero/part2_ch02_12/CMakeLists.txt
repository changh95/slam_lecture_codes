cmake_minimum_required(VERSION 3.14)
project("ransac_usac_tutorial" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build: Eigen-heavy code is orders of magnitude
# slower at -O0, which skews the timing comparisons these demos print.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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

# Find Eigen (needed for RansacLib and MAGSAC)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# RansacLib (header-only RANSAC library)
find_path(RANSACLIB_INCLUDE_DIR RansacLib/ransac.h
    HINTS /usr/local/include ${CMAKE_INSTALL_PREFIX}/include)
if(RANSACLIB_INCLUDE_DIR AND Eigen3_FOUND)
    message(STATUS "Found RansacLib: ${RANSACLIB_INCLUDE_DIR}")
    add_executable(ransac_ransaclib examples/ransac_ransaclib.cpp)
    target_include_directories(ransac_ransaclib PRIVATE ${RANSACLIB_INCLUDE_DIR})
    target_link_libraries(ransac_ransaclib ${OpenCV_LIBS} Eigen3::Eigen)
else()
    message(STATUS "RansacLib not found, skipping ransac_ransaclib")
endif()

# MAGSAC (state-of-the-art threshold-free robust estimation).
# The Dockerfile builds it from source and installs headers to
# /usr/local/include/pymagsac/include and /usr/local/include/gcransac,
# libs (libMAGSAC.a, libGraphCutRANSAC.a) to /usr/local/lib.
find_path(MAGSAC_INCLUDE_DIR magsac.h
    HINTS /usr/local/include ${CMAKE_INSTALL_PREFIX}/include
    PATH_SUFFIXES pymagsac/include)
find_path(GCRANSAC_INCLUDE_DIR GCRANSAC.h
    HINTS /usr/local/include ${CMAKE_INSTALL_PREFIX}/include
    PATH_SUFFIXES gcransac)
find_library(MAGSAC_LIBRARY NAMES MAGSAC
    HINTS /usr/local/lib ${CMAKE_INSTALL_PREFIX}/lib)
find_library(GCRANSAC_LIBRARY NAMES GraphCutRANSAC graph_cut_ransac gcransac
    HINTS /usr/local/lib ${CMAKE_INSTALL_PREFIX}/lib)
find_package(OpenMP QUIET)
if(MAGSAC_INCLUDE_DIR AND GCRANSAC_INCLUDE_DIR AND MAGSAC_LIBRARY
   AND GCRANSAC_LIBRARY AND Eigen3_FOUND)
    message(STATUS "Found MAGSAC: ${MAGSAC_INCLUDE_DIR} (${MAGSAC_LIBRARY})")
    add_executable(ransac_magsac examples/ransac_magsac.cpp)
    target_include_directories(ransac_magsac PRIVATE
        ${MAGSAC_INCLUDE_DIR} ${GCRANSAC_INCLUDE_DIR})
    target_link_libraries(ransac_magsac ${OpenCV_LIBS} Eigen3::Eigen
        ${MAGSAC_LIBRARY} ${GCRANSAC_LIBRARY})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ransac_magsac OpenMP::OpenMP_CXX)
    endif()
else()
    message(STATUS "MAGSAC not found, skipping ransac_magsac")
endif()

# Install targets (optional)
install(TARGETS ransac_homography ransac_fundamental ransac_custom
        RUNTIME DESTINATION bin)
