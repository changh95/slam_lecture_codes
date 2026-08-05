cmake_minimum_required(VERSION 3.5)
project(vikit_ros)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processor number: ${N}")
  if(N GREATER 4)
      add_definitions(-DMP_EN)
      add_definitions(-DMP_PROC_NUM=3)
      message("Cores for MP: 3")
  elseif(N GREATER 3)
      add_definitions(-DMP_EN)
      add_definitions(-DMP_PROC_NUM=2)
      message("Cores for MP: 2")
  else()
      add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vikit_common REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
set(dependencies
  rclcpp
  visualization_msgs
  tf2_ros
  tf2
  Eigen3
  tf2_geometry_msgs
  vikit_common
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(${dependencies})

# Link libraries
list(APPEND SOURCEFILES 
  src/output_helper.cpp
  src/camera_loader.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})
# Modify target_include_directories to include the project directory directly
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CURRENT_SOURCE_DIR}  # Add this line to include the project root directory
  ${Sophus_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${cpp_typesupport_target}
)

ament_target_dependencies(${PROJECT_NAME} ${dependencies})

# Install headers directly to the include directory without the project name prefix
install(DIRECTORY include/
  DESTINATION include
  COMPONENT ${PROJECT_NAME}
)

# Install library
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Export the include directory for the project
set_target_properties(${PROJECT_NAME} PROPERTIES
  PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Ensure the include directory is correctly set for downstream packages
ament_export_include_directories(include)

ament_package()