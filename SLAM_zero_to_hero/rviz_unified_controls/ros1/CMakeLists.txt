cmake_minimum_required(VERSION 3.0.2)
project(rviz_unified_controls)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS rviz pluginlib)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

# rviz exports only librviz; the stock view controllers we subclass live in
# librviz_default_plugin next to it.
find_library(RVIZ_DEFAULT_PLUGIN_LIB rviz_default_plugin
             HINTS ${rviz_DIR}/../../../lib REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/unified_view_controllers.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${RVIZ_DEFAULT_PLUGIN_LIB} Qt5::Widgets)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(FILES plugin_description.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
