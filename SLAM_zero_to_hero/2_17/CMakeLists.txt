cmake_minimum_required(VERSION 3.10)
project(Triangulation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION_STRING}")

# Triangulation demo executable
add_executable(triangulation_demo examples/triangulation_demo.cpp)

target_link_libraries(triangulation_demo
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

target_include_directories(triangulation_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# Compiler flags for better optimization
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(triangulation_demo PRIVATE -O3 -march=native)
endif()

# Enable all warnings
target_compile_options(triangulation_demo PRIVATE -Wall -Wextra -Wpedantic)

message(STATUS "")
message(STATUS "=== Triangulation Tutorial ===")
message(STATUS "Executable: triangulation_demo")
message(STATUS "Run './triangulation_demo' to see triangulation methods")
message(STATUS "")
