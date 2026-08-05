# Cerberus 2.0: Visual-Inertial-Leg Odometry for legged robots (CMU RExLab)
# Repo: https://github.com/ShuoYangRobotics/Cerberus2.0
# Sensors: Unitree Go1 body IMU + 4 foot IMUs (WT901) + joint encoders
#          + forward RealSense stereo infra pair
#
# Upstream ships no standalone Dockerfile -- it ships a *devcontainer* that pulls
# a prebuilt 2.5 GB image (yyss2028/legged_state_estimation:2.0) and expects you
# to run `catkin build` by hand inside VSCode. This file reproduces that
# environment from source and bakes the build in, so a run is one `podman run`.
#
# Deliberately NOT reproduced from .devcontainer/Dockerfile:
#   * gram_savitzky_golay, OSQP, osqp-eigen -- no header from any of the three is
#     included anywhere in include/ or src/. They are leftovers from Cerberus 1.
#   * the elevation-mapping workspace (grid_map, kindr, elevation_mapping,
#     plane_segmentation) -- only launch/elev_map/* uses it, not the odometry.
#   * oh-my-zsh -- zsh itself is installed because upstream's launch files use
#     launch-prefix="zsh -c ...", but the interactive shell setup is noise here.
FROM docker.io/ros:noetic-perception-focal

ARG DEBIAN_FRONTEND=noninteractive

# Noetic went EOL in May 2025 and the base image prints a banner about it on every
# single container start, including every `rosrun`/`roslaunch` invocation inside
# run_demo.sh. Nothing here can move off Noetic -- Cerberus 2.0 is ROS 1 only --
# so the warning is pure noise.
ENV DISABLE_ROS1_EOL_WARNINGS=true

# Timezone matters: rosbag /clock and the estimator's ros::Time bookkeeping are
# UTC-based, but a container with no /etc/localtime makes every ROS log line
# claim 1970. Upstream hardcodes America/New_York; UTC is the neutral choice.
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# libceres-dev on focal IS 1.14.0 -- byte-for-byte the version upstream's
# Dockerfile clones and builds from source. One apt line replaces that build.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget unzip ca-certificates pkg-config \
      libeigen3-dev libatlas-base-dev libsuitesparse-dev \
      libgoogle-glog-dev libgflags-dev libceres-dev \
      libboost-all-dev liburdfdom-dev \
      python3-catkin-tools \
      ros-noetic-pinocchio \
      ros-noetic-cv-bridge ros-noetic-image-transport \
      ros-noetic-message-filters ros-noetic-tf \
      zsh && \
    rm -rf /var/lib/apt/lists/*

ENV CATKIN_WS=/home/EstimationUser/estimation_ws
ENV SUPPORT_WS=/home/EstimationUser/support_ws

# CMakeLists.txt hardcodes /home/EstimationUser/support_ws/libtorch in
# CMAKE_PREFIX_PATH and the build-time test target hardcodes
# /home/EstimationUser/estimation_ws/..., so the workspace keeps upstream's
# absolute paths even though this image has no EstimationUser account. Deviating
# would mean patching CMakeLists.txt for no gain.

# casadi 3.5.5. Every leg-kinematics Jacobian in the estimator is a casadi
# expression (include/utils/casadi_kino.hpp), so this is a hard dependency, and
# it is not packaged for focal.
RUN git clone https://github.com/casadi/casadi.git /tmp/casadi && \
    cd /tmp/casadi && git checkout tags/3.5.5 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DWITH_CPLEX=OFF -DWITH_KNITRO=OFF -DWITH_OOQP=OFF -DWITH_SNOPT=OFF && \
    make -j$(nproc) && make install && ldconfig && \
    test -f /usr/local/include/casadi/core/casadi_misc.hpp && \
    rm -rf /tmp/casadi

# libtorch, CPU, cxx11 ABI. find_package(Torch REQUIRED) in CMakeLists.txt gates
# configure on it. Pinned to 1.13.1 rather than upstream's
# "...-latest.zip" nightly for two reasons: the nightly URL is a moving target,
# and libtorch >= 2.0's TorchConfig.cmake forces CMAKE_CXX_STANDARD 17, which
# collides with this project's C++14.
RUN mkdir -p $SUPPORT_WS && cd $SUPPORT_WS && \
    wget -q "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip" \
      -O libtorch.zip && \
    unzip -q libtorch.zip && rm libtorch.zip && \
    test -f $SUPPORT_WS/libtorch/share/cmake/Torch/TorchConfig.cmake

# camera_models only. cerberus2 needs it as a catkin component (it supplies the
# pinhole/MEI/Kannala-Brandt models the feature tracker undistorts with), but the
# rest of upstream's VINS-Fusion fork is dead weight here: vins_estimator is an
# alternative estimator this demo does not run, and global_fusion would drag in
# GeographicLib. Same fork, same commit as upstream's Dockerfile clones.
RUN mkdir -p $CATKIN_WS/src && \
    git clone --depth 1 https://github.com/ShuoYangRobotics/VINS-Fusion.git /tmp/vins && \
    mv /tmp/vins/camera_models $CATKIN_WS/src/ && rm -rf /tmp/vins

# camera_models predates OpenCV 4: Chessboard.cc and CameraCalibration.cc use the
# C-API colour codes (CV_GRAY2BGR, CV_TERMCRIT_EPS), which OpenCV 4 moved out of
# opencv.hpp into imgproc/types_c.h. Force-including that header fixes all five
# uses. It has to be done by editing the CMakeLists line: the package *assigns*
# to CMAKE_CXX_FLAGS, so any -DCMAKE_CXX_FLAGS passed on the command line is
# overwritten before it reaches the compiler.
RUN cd $CATKIN_WS/src/camera_models && \
    sed -i 's|set(CMAKE_CXX_FLAGS "-std=c++11")|set(CMAKE_CXX_FLAGS "-std=c++11 -include opencv2/imgproc/types_c.h")|' CMakeLists.txt && \
    grep -q 'types_c.h' CMakeLists.txt

# Cerberus 2.0 itself, pinned to main@d81c394 (2024-01-17, current HEAD).
ARG CERBERUS2_COMMIT=d81c394fd58f7443f697ee53f8cc04199f008f9e
RUN git clone https://github.com/ShuoYangRobotics/Cerberus2.0.git $CATKIN_WS/src/cerberus2 && \
    cd $CATKIN_WS/src/cerberus2 && git checkout ${CERBERUS2_COMMIT}

# Publish the sliding-window local map AND the factor graph. Upstream computes the
# map in src/utils/visualization.cpp's pubPointCloud(), but that function -- and
# eleven other publishers with it -- is commented out, so a stock build shows you
# the feature tracks in 2D and nothing of the 3D structure the estimator solved
# for, let alone the graph it solved. Everything needed (feature_manager_, Ps, Rs,
# ric, tic) is private to VILOEstimator, so the patch adds two const accessors,
# getLocalMap() and getFactorGraph(), and publishes four topics from VILOFusion:
#   /vilo/point_cloud         landmark positions        (sensor_msgs/PointCloud)
#   /vilo/key_poses           keyframe nodes            (Marker, SPHERE_LIST)
#   /vilo/factor_graph_imu    keyframe-chain IMU preintegration factors (LINE_LIST)
#   /vilo/factor_graph_leg    keyframe-chain leg-odometry factors -- the ones that
#                             make this VILO and not VIO
#   /vilo/factor_graph_obs    landmark-to-keyframe reprojection factors
# Estimator maths is untouched: both accessors are read-only.
COPY patches/ /tmp/patches/
RUN cd $CATKIN_WS/src/cerberus2 && \
    git apply --verbose /tmp/patches/0001-publish-local-map-and-factor-graph.patch && \
    grep -q "getLocalMap" include/vilo/VILOEstimator.hpp

# The casadi header patch that upstream applies as a devcontainer postStartCommand.
# casadi's casadi_misc.hpp defines `operator<<(ostream&, const pair<T1,T2>&)`
# inside namespace casadi; libtorch's c10/util/Logging headers define the same
# overload for std::pair, and in a TU that pulls in both, every `<<` on a pair is
# ambiguous. misc/casadi_misc.hpp is casadi's own file with exactly that one
# overload commented out (the vector/set/map ones are untouched).
RUN cp $CATKIN_WS/src/cerberus2/misc/casadi_misc.hpp /usr/local/include/casadi/core/ && \
    grep -q '^// ostream& operator<<(ostream& stream, const pair<T1, T2>&' \
      /usr/local/include/casadi/core/casadi_misc.hpp

# Two directories that must exist *before* the build, both created by upstream's
# devcontainer postStartCommand:
#   devel/lib/cerberus2 -- CMakeLists.txt adds run_test_LOTightIntegration to
#     ALL, i.e. it *executes* test_LOTightIntegration during the build with this
#     as its WORKING_DIRECTORY. A missing directory fails the build.
#   bags/output -- that same test calls readParametersFile(), which opens the
#     result CSVs under output_path for truncation.
RUN mkdir -p $CATKIN_WS/devel/lib/cerberus2 $CATKIN_WS/src/cerberus2/bags/output

WORKDIR $CATKIN_WS
# --merge-devel, not catkin_tools' default --link-devel. With the linked layout
# every package gets a private devel space and the top-level devel/setup.sh only
# works from a *clean* environment, because _setup_util.py skips a workspace it
# already sees on CMAKE_PREFIX_PATH -- and this base image exports
# CMAKE_PREFIX_PATH=/opt/ros/noetic and ROS_PACKAGE_PATH=/opt/ros/noetic/share as
# image ENV. The result is a workspace that builds fine but whose setup leaves
# ROS_PACKAGE_PATH as build/catkin_tools_prebuild:/opt/ros/noetic/share, so
# `rospack find cerberus2` and every roslaunch by package name fail. A merged
# devel space behaves like catkin_make's and works in either sourcing order.
#
# Two sequential invocations, not one: cerberus2's CMakeLists.txt does
# find_package(catkin COMPONENTS ... camera_models ...) but its package.xml
# declares only roscpp, so catkin_tools sees no dependency between the two and
# schedules them in parallel -- cerberus2 then fails configure in under a second
# with "Could not find a package configuration file provided by camera_models".
# Upstream hits the same wall and works around it the same way (its Dockerfile
# runs `catkin build` twice).
RUN . /opt/ros/noetic/setup.sh && \
    catkin config --init --extend /opt/ros/noetic --merge-devel && \
    catkin build camera_models -j$(nproc) --no-status -DCMAKE_BUILD_TYPE=Release && \
    . $CATKIN_WS/devel/setup.sh && \
    catkin build cerberus2 -j$(nproc) --no-status -DCMAKE_BUILD_TYPE=Release

# catkin swallows some link failures into a non-fatal warning; assert the two
# binaries this demo actually launches, and that the package resolves by name in
# both sourcing orders a user might use.
RUN . /opt/ros/noetic/setup.sh && . $CATKIN_WS/devel/setup.sh && \
    rospack find cerberus2 && rospack find camera_models && \
    test -x $CATKIN_WS/devel/lib/cerberus2/cerberus2_main && \
    test -x $CATKIN_WS/devel/lib/cerberus2/mipo_main
RUN . $CATKIN_WS/devel/setup.sh && rospack find cerberus2 >/dev/null

RUN sed -i "6i source \"$CATKIN_WS/devel/setup.bash\"" /ros_entrypoint.sh

# Runtime-only extras, in their own layer after the build so adding one costs a
# cheap rebuild instead of a full recompile:
#   rviz                -- the GUI; upstream's own rviz configs need it
#   robot-state-publisher + tf2-ros -- turn /vis_joint_state into the leg TF
#                          chain so the legs are visible (see the launch file)
#   imagemagick + xdotool -- window capture for the README screenshot
#   matplotlib/scipy    -- plot_trajectory.py; scipy.io.loadmat reads the iPhone
#                          GPS ground truth that ships as a .mat next to the bag
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-rviz imagemagick xdotool x11-utils \
      ros-noetic-robot-state-publisher ros-noetic-tf2-ros \
      python3-matplotlib python3-scipy python3-numpy && \
    rm -rf /var/lib/apt/lists/*

# Lecture assets last, for the same cache reason. config/lecture/ holds the
# per-sequence yaml *and* the two camera calibs, because parameters.cpp resolves
# cam0_calib/cam1_calib relative to the directory of the config file it is given.
COPY config/ $CATKIN_WS/src/cerberus2/config/lecture/
COPY launch/ $CATKIN_WS/src/cerberus2/launch/lecture/
COPY rviz/ $CATKIN_WS/src/cerberus2/config/rviz/lecture/
COPY scripts/ /opt/cerberus2_demo/

# `rosrun`/roslaunch resolve type="pose_to_path.py" by walking the package's own
# directory tree, so the node has to be reachable from inside src/cerberus2. A
# symlink keeps /opt/cerberus2_demo the single editable copy -- bind-mounting a
# modified scripts/ over it still works.
RUN chmod +x /opt/cerberus2_demo/*.sh /opt/cerberus2_demo/*.py && \
    mkdir -p $CATKIN_WS/src/cerberus2/script/lecture && \
    ln -sf /opt/cerberus2_demo/pose_to_path.py \
      $CATKIN_WS/src/cerberus2/script/lecture/pose_to_path.py

# Assert the lecture assets are actually reachable by name: roslaunch --files
# parses the launch file and resolves every $(find cerberus2) inside it, which
# covers the config path, the rviz path and the pose_to_path.py node lookup.
RUN . $CATKIN_WS/devel/setup.sh && \
    roslaunch --files cerberus2 cerberus2_bag.launch > /dev/null && \
    rosrun --prefix echo cerberus2 pose_to_path.py > /dev/null && \
    test -f $CATKIN_WS/src/cerberus2/config/lecture/mill19_trail.yaml && \
    test -f $CATKIN_WS/src/cerberus2/config/lecture/cmu_garage.yaml && \
    test -f $CATKIN_WS/src/cerberus2/config/lecture/go1_realsense_left.yaml && \
    test -f $CATKIN_WS/src/cerberus2/config/rviz/lecture/cerberus2_vilo.rviz && \
    python3 -c "import matplotlib, numpy, scipy.io"

WORKDIR /data
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
