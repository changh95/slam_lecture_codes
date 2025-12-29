FROM ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y build-essential git cmake python3-wstool python3-rosdep ninja-build stow ros-noetic-velodyne-pointcloud ros-noetic-pointcloud-to-laserscan python3-catkin-tools google-mock libgmock-dev

RUN mkdir -p /catkin_ws/src/cartographer
WORKDIR /catkin_ws/src/cartographer
COPY cartographer.tar.xz /catkin_ws/src/cartographer/cartographer.tar.xz
RUN tar -xvf cartographer.tar.xz && catkin init
COPY velodyne_kitti_2D.lua /catkin_ws/src/cartographer/cartographer_ros/configuration_files/velodyne_kitti_2D.lua
COPY velodyne_kitti_uamc.launch /catkin_ws/src/cartographer/cartographer_ros/launch/velodyne_kitti_uamc.launch
COPY velodyne_kitti_uamc.udrf /catkin_ws/src/cartographer/cartographer_ros/udrf/velodyne_kitti_uamc.udrf
RUN rosdep install --from-paths . --ignore-src -r -y
RUN cd ./cartographer/scripts && rm -rf abseil-cpp &&\
    sh ./install_abseil.sh
WORKDIR /catkin_ws/
RUN catkin config  --extend /opt/ros/noetic && catkin build

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh && catkin build

WORKDIR /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
