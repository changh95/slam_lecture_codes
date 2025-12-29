cmake_minimum_required(VERSION 3.15)
project(slam_nanobind LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Download nanobind if not found (using FetchContent)
include(FetchContent)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.0.0
)
FetchContent_MakeAvailable(nanobind)

# Find Eigen3
find_package(Eigen3 REQUIRED)

message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")

# Build the nanobind module
nanobind_add_module(
    slam_bindings
    STABLE_ABI
    src/slam_bindings.cpp
)

target_include_directories(slam_bindings PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)

# Install to current directory for easy testing
install(TARGETS slam_bindings DESTINATION ${CMAKE_SOURCE_DIR})

message(STATUS "")
message(STATUS "=== nanobind SLAM Bindings ===")
message(STATUS "After building, the module will be in build/")
message(STATUS "Copy slam_bindings*.so to your Python path or use:")
message(STATUS "  export PYTHONPATH=\${PYTHONPATH}:$(pwd)/build")
message(STATUS "Then: python3 -c 'import slam_bindings; print(slam_bindings.__doc__)'")
message(STATUS "")
