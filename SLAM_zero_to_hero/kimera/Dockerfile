# Kimera-Semantics: real-time metric-semantic 3D reconstruction (MIT SPARK)
# Repo: https://github.com/MIT-SPARK/Kimera-Semantics
#
# ROS 1 Noetic + catkin. Kimera-Semantics is a thin layer on voxblox: it fuses a
# depth image + a 2D semantic segmentation image (turned into a labelled point
# cloud by depth_image_proc) into a semantic TSDF and publishes a mesh coloured
# by class. No GPU needed. Every source is pinned to the commit in
# Kimera-Semantics/install/kimera_semantics_https.rosinstall's HEAD as of 2026-09.
FROM docker.io/osrf/ros:noetic-desktop-full

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates autoconf libtool \
      python3-wstool python3-catkin-tools \
      protobuf-compiler libprotobuf-dev libgoogle-glog-dev libgflags-dev \
      ros-noetic-cmake-modules ros-noetic-depth-image-proc ros-noetic-rviz \
      xvfb x11-apps imagemagick mesa-utils && \
    rm -rf /var/lib/apt/lists/*

# Noetic's PCL 1.10 refuses anything below C++14, but eigen_checks' exported
# cmake extras append -std=c++11 to CMAKE_CXX_FLAGS, so both Kimera packages get
# an explicit add_definitions(-std=c++14), which lands later on the command line.
# name  repo  commit  (the kimera_semantics_https.rosinstall set, minus the
# python-only minkindr_python / numpy_eigen / catkin_boost_python_buildtool)
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    clone() { git clone -q "https://github.com/$2.git" "$1" && git -C "$1" checkout -q "$3"; } && \
    clone catkin_simple         catkin/catkin_simple         0e62848b12da76c8cc58a1add42b4f894d1ac21e && \
    clone disparity_image_proc  ToniRV/disparity_image_proc  c35c2ff6da31eb5ceceb326e4997a7edf8589db0 && \
    clone eigen_catkin          ethz-asl/eigen_catkin        3323b388540fa95ec9da6f9cd887f70ead055edb && \
    clone eigen_checks          ethz-asl/eigen_checks        22a6247a3df11bc285d43d1a030f4e874a413997 && \
    clone gflags_catkin         ethz-asl/gflags_catkin       fc38fc525f7d48881aebb27a7b9978453556bbd4 && \
    clone glog_catkin           ethz-asl/glog_catkin         40a9edadd15c59f8b57dc947d0135b0a007ea10b && \
    clone minkindr              ethz-asl/minkindr            564f12639a8447d4d3e5e7707851424302941056 && \
    clone minkindr_ros          ethz-asl/minkindr_ros        5528b042124fe056a7cf53f96c8b39e1e32ec2b9 && \
    clone protobuf_catkin       ethz-asl/protobuf_catkin     721a6cc17e7e937e7accfb88a9967b26134db6f9 && \
    clone voxblox               ethz-asl/voxblox             c8066b04075d2fee509de295346b1c0b788c4f38 && \
    clone Kimera-Semantics      MIT-SPARK/Kimera-Semantics   1ed69c61f4d1d71a812336574ef298e6210ebcd2 && \
    touch minkindr/minkindr_python/CATKIN_IGNORE && \
    sed -i 's/^catkin_simple()/catkin_simple()\nadd_definitions(-std=c++14)/' \
      Kimera-Semantics/kimera_semantics/CMakeLists.txt \
      Kimera-Semantics/kimera_semantics_ros/CMakeLists.txt && \
    grep -c 'std=c++14' Kimera-Semantics/kimera_semantics_ros/CMakeLists.txt

WORKDIR /catkin_ws
RUN catkin init && \
    catkin config --extend /opt/ros/noetic --merge-devel \
      --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    catkin build -j$(nproc) --no-status kimera_semantics_ros

# Course-wide RViz mouse scheme: left drag rotates, right/middle drag pans, wheel zooms.
# The upstream kimera_semantics_ros/rviz/*.rviz files get the same controller
# (only the Views > Current > Class line; the "- Class:" saved views stay stock).
COPY --from=localhost/slam_zero_to_hero:rviz_unified_controls /rviz_unified_controls /opt/rviz_unified_controls
RUN /opt/rviz_unified_controls/install_ros1.sh /opt/rviz_unified_controls && \
    sed -i -E 's#^( +)Class: rviz/(Orbit|ThirdPersonFollower|TopDownOrtho)$#\1Class: slam_zero_to_hero/Unified\2#' \
      /catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/rviz/*.rviz && \
    grep -c 'Class: slam_zero_to_hero/Unified' /catkin_ws/src/Kimera-Semantics/kimera_semantics_ros/rviz/*.rviz

# Lecture launch file, RViz config and run/measure scripts
COPY launch/ /kimera/launch/
COPY config/ /kimera/config/
COPY scripts/ /kimera/scripts/
RUN chmod +x /kimera/scripts/* && \
    sed -i '/^exec "\$@"/i source "/catkin_ws/devel/setup.bash"' /ros_entrypoint.sh

# Noetic is EOL; its RViz otherwise opens a modal warning over the 3D view.
ENV DISABLE_ROS1_EOL_WARNINGS=1
WORKDIR /kimera
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
