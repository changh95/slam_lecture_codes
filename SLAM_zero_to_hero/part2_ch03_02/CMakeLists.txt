cmake_minimum_required(VERSION 3.10)
project("Introduction to point cloud processing" LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# Ubuntu's VTK 9.1 declares VTK::mpi -> MPI::MPI_C but never looks MPI up
# itself, so find_package(PCL) below fails unless the target already exists.
# Harmless where VTK was built without MPI — nothing here links against it.
find_package(MPI QUIET)

find_package(PCL REQUIRED)
if (PCL_FOUND)
    message(STATUS "Found PCL library: " ${PCL_INCLUDE_DIRS})
    include_directories(${PCL_INCLUDE_DIRS})
endif (PCL_FOUND)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(visualization examples/visualization.cpp)
target_link_libraries(visualization ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})

add_executable(visualization_kitti examples/visualization_kitti.cpp)
target_link_libraries(visualization_kitti ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})