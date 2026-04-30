cmake_minimum_required(VERSION 3.10)
project(BASIC_CPP_PROGRAMMING LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build type: Release (default)")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# Control flow examples
add_executable(for_loop examples/for_loop.cpp)
add_executable(while_loop examples/while_loop.cpp)

# STL container examples
add_executable(vector examples/vector.cpp)
add_executable(map examples/map.cpp)
add_executable(unordered_map examples/unordered_map.cpp)

# Template examples
add_executable(template_function examples/template_function.cpp)
add_executable(template_class examples/template_class.cpp)

# Memory management
add_executable(smart_pointers examples/smart_pointers.cpp)
