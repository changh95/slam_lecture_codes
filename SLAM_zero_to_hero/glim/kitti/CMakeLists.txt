cmake_minimum_required(VERSION 3.16)
project(glim_kitti LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# IMPORTANT ordering: glim's and gtsam_points' exported link interfaces refer to
# the target `GTSAM::GTSAM`, but GTSAM 4.2.0's own GTSAMConfig.cmake only exports
# a target called `gtsam`. The `GTSAM::GTSAM` wrapper is created by the
# FindGTSAM.cmake that *gtsam_points* installs next to its own config file.
# glim-config.cmake calls find_dependency(GTSAM) BEFORE find_dependency(gtsam_points),
# so GTSAM gets resolved in CONFIG mode first and GTSAM::GTSAM is never created
# -> link fails with "cannot find -lGTSAM::GTSAM".
# Finding gtsam_points first puts its FindGTSAM.cmake on CMAKE_MODULE_PATH.
find_package(gtsam_points REQUIRED)
if(NOT TARGET GTSAM::GTSAM)
  find_package(GTSAM REQUIRED)
  add_library(GTSAM::GTSAM INTERFACE IMPORTED GLOBAL)
  target_link_libraries(GTSAM::GTSAM INTERFACE gtsam)
endif()

# glim installs /usr/local/lib/cmake/glim/glim-config.cmake from its plain
# (non-ROS) CMake build, exporting the glim::glim target.
find_package(glim REQUIRED)

add_executable(glim_kitti glim_kitti.cpp)
target_link_libraries(glim_kitti glim::glim)

# gcc < 9.1 keeps std::filesystem in a separate library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  target_link_libraries(glim_kitti stdc++fs)
endif()

install(TARGETS glim_kitti RUNTIME DESTINATION bin)
