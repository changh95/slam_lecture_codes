cmake_minimum_required(VERSION 3.5)
project(vikit_py)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# set dependencies
set(dependencies
rclpy
)

#install python modules
# ament_python_install_package(${PROJECT_NAME})

#install python executables
install(PROGRAMS
  src/vikit_py/align_trajectory.py
  src/vikit_py/cpu_info.py
  src/vikit_py/depthmap_utils.py
  src/vikit_py/math_utils.py
  src/vikit_py/ros_node.py
  src/vikit_py/transformations.py
  DESTINATION lib/${PROJECT_NAME}
)

#--------------------------------------------------------------------
# export dependencies
ament_export_dependencies(${dependencies})
ament_package()
