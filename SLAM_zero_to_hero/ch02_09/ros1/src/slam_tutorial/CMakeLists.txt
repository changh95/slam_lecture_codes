cmake_minimum_required(VERSION 3.0.2)
project(slam_tutorial)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

# C++ executables
add_executable(publisher_cpp src/publisher.cpp)
target_link_libraries(publisher_cpp ${catkin_LIBRARIES})

add_executable(subscriber_cpp src/subscriber.cpp)
target_link_libraries(subscriber_cpp ${catkin_LIBRARIES})

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/publisher.py
  scripts/subscriber.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
