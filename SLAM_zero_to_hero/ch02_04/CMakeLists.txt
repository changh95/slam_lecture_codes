cmake_minimum_required(VERSION 3.10)
project(CPU_PROFILER LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(easy_profiler REQUIRED)
find_package(Threads REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "easy_profiler found: ${easy_profiler_FOUND}")

# Main profiling executable
add_executable(profile_features examples/profile_features.cpp)

target_link_libraries(profile_features
    ${OpenCV_LIBS}
    easy_profiler
    Threads::Threads
)

target_include_directories(profile_features PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# Compiler flags for better profiling
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(profile_features PRIVATE -O2 -g)
endif()

message(STATUS "")
message(STATUS "=== CPU Profiler Tutorial ===")
message(STATUS "Executable: profile_features")
message(STATUS "Run './profile_features' to generate profile.prof")
message(STATUS "View with: profiler_gui profile.prof")
message(STATUS "")
