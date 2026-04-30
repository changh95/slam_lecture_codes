cmake_minimum_required(VERSION 3.8)
project(slam_tutorial)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# C++ executables
add_executable(publisher_cpp src/publisher.cpp)
ament_target_dependencies(publisher_cpp rclcpp geometry_msgs)

add_executable(subscriber_cpp src/subscriber.cpp)
ament_target_dependencies(subscriber_cpp rclcpp geometry_msgs)

# Install C++ executables
install(TARGETS
  publisher_cpp
  subscriber_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  slam_tutorial/publisher.py
  slam_tutorial/subscriber.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
