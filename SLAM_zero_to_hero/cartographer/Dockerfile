FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y build-essential git cmake python3-wstool python3-rosdep ninja-build stow \
      ros-noetic-velodyne-pointcloud ros-noetic-pointcloud-to-laserscan \
      python3-catkin-tools google-mock libgmock-dev && \
    apt-get clean

RUN mkdir -p /catkin_ws/src/cartographer
WORKDIR /catkin_ws/src/cartographer
COPY cartographer.tar.xz /catkin_ws/src/cartographer/cartographer.tar.xz
RUN tar -xf cartographer.tar.xz && catkin init

# Legacy 2D-on-KITTI demo (kept for backwards compat with earlier readers)
COPY velodyne_kitti_2D.lua /catkin_ws/src/cartographer/cartographer_ros/configuration_files/velodyne_kitti_2D.lua
COPY velodyne_kitti_uamc.launch /catkin_ws/src/cartographer/cartographer_ros/launch/velodyne_kitti_uamc.launch
COPY velodyne_kitti_uamc.udrf /catkin_ws/src/cartographer/cartographer_ros/udrf/velodyne_kitti_uamc.udrf

# Hilti 2022 config (Hesai PandarXT-32, exp14_basement_2.bag) — runs in 2D mode
# (file kept named *_3d.lua for backwards compatibility with earlier docs).
COPY hilti_3d.lua /catkin_ws/src/cartographer/cartographer_ros/configuration_files/hilti_3d.lua
COPY hilti_3d.launch /catkin_ws/src/cartographer/cartographer_ros/launch/hilti_3d.launch
COPY run_hilti_offline.sh /usr/local/bin/run_hilti_offline.sh

# rospack resolves cartographer_ros to the nested package dir; symlink the lua
# into that directory so cartographer_offline_node finds it on its default path.
RUN ln -sf /catkin_ws/src/cartographer/cartographer_ros/configuration_files/hilti_3d.lua \
            /catkin_ws/src/cartographer/cartographer_ros/cartographer_ros/configuration_files/hilti_3d.lua && \
    chmod +x /usr/local/bin/run_hilti_offline.sh

RUN rosdep install --from-paths . --ignore-src -r -y
RUN cd ./cartographer/scripts && rm -rf abseil-cpp && sh ./install_abseil.sh
WORKDIR /catkin_ws/
RUN catkin config --extend /opt/ros/noetic && catkin build

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh && catkin build

WORKDIR /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
