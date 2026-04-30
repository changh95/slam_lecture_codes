cmake_minimum_required(VERSION 3.14)
project(tensorrtbuffer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# Create library
add_library(tensorrtbuffer SHARED
    src/logger.cpp
)

target_link_libraries(tensorrtbuffer
    ${CUDA_LIBRARIES}
)
