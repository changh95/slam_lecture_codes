cmake_minimum_required(VERSION 3.10)
project(pcl_point_cloud_preprocessing LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Required: VTK 9.1 references MPI::MPI_C without looking MPI up itself.
find_package(MPI QUIET)

find_package(PCL 1.10 REQUIRED COMPONENTS
    common
    io
    kdtree
    search
    filters
    features
    segmentation
    visualization
)

message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(DEMOS
    passthrough
    downsampling
    sor
    kdtree
    plane_det
    normal_estimation
)

foreach(demo ${DEMOS})
    add_executable(${demo} examples/${demo}.cpp)
    target_include_directories(${demo} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(${demo} ${PCL_LIBRARIES})
endforeach()
