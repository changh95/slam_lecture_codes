cmake_minimum_required(VERSION 3.10)
project("Iterative Closest Point (ICP)")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH )

find_package(PCL REQUIRED)
if (PCL_FOUND)
	message(STATUS "Found PCL library: " ${PCL_INCLUDE_DIRS})
	include_directories(${PCL_INCLUDE_DIRS})
endif (PCL_FOUND)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcd_viewer examples/pcd_viewer.cpp)
target_link_libraries(pcd_viewer ${PCL_LIBRARIES})

