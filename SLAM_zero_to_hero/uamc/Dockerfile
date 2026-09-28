# UAMC course image: U-AMC's FAST-LIVO2-ROS2 on ROS 2 Humble, built from the trees
# vendored in this folder (FAST-LIVO2-ROS2/ and rpg_vikit_rational_polynomial/).
# Build context is uamc/, so the vendored vikit is COPYed instead of cloned.
#
#   podman build -t slam_zero_to_hero:uamc .
#
# CPU only; RViz renders with Mesa (LIBGL_ALWAYS_SOFTWARE) under Xvfb.
FROM docker.io/library/ros:humble-perception-jammy

ARG WS=/fast_livo_ws
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    WS=${WS} \
    CPATH=/opt/ros/humble/include

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git \
      libapr1-dev libboost-thread-dev libeigen3-dev libfmt-dev libopencv-dev libpcl-dev \
      python3-colcon-common-extensions python3-pip python3-numpy \
      ros-humble-cv-bridge ros-humble-image-transport ros-humble-compressed-image-transport \
      ros-humble-pcl-conversions ros-humble-pcl-ros \
      ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins ros-humble-ros2bag \
      ros-humble-rviz2 ros-humble-sophus ros-humble-tf2-ros ros-humble-tf2-geometry-msgs \
      ros-humble-demo-nodes-cpp \
      xvfb x11-utils imagemagick mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Livox-SDK2 and livox_ros_driver2, pinned (both HEAD on 2026-09-27).
ARG LIVOX_SDK2_SHA=c0796f04c143143899c87a773d9f6b7136453c0b
ARG LIVOX_DRIVER2_SHA=21445540f0d100dc86a7e6df312dd70bbdb4afdf
RUN git init -q /tmp/Livox-SDK2 && cd /tmp/Livox-SDK2 \
    && git fetch -q --depth 1 https://github.com/Livox-SDK/Livox-SDK2.git ${LIVOX_SDK2_SHA} \
    && git checkout -q FETCH_HEAD \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel "$(nproc)" && cmake --install build && ldconfig \
    && rm -rf /tmp/Livox-SDK2
RUN mkdir -p ${WS}/src && git init -q ${WS}/src/livox_ros_driver2 && cd ${WS}/src/livox_ros_driver2 \
    && git fetch -q --depth 1 https://github.com/Livox-SDK/livox_ros_driver2.git ${LIVOX_DRIVER2_SHA} \
    && git checkout -q FETCH_HEAD && rm -rf .git \
    && cp package_ROS2.xml package.xml && cp -r launch_ROS2 launch

# vikit from the vendored tree (U-AMC/rpg_vikit_rational_polynomial @ 6f213c7).
# vikit_common is a plain CMake package installed globally; colcon builds vikit_ros.
COPY rpg_vikit_rational_polynomial ${WS}/src/rpg_vikit
# patches/vikit_remote_camera_params.patch: upstream vikit reads the RationalPolynomial
# intrinsics with a local-only lookup that never finds them, so fastlivo_mapping dies with
# SIGFPE on the first image. See the patch header.
COPY patches /uamc_patches
RUN cd ${WS}/src/rpg_vikit && git apply -p1 --verbose /uamc_patches/vikit_remote_camera_params.patch
RUN . /opt/ros/humble/setup.sh \
    && cmake -S ${WS}/src/rpg_vikit/vikit_common -B /tmp/vikit_common_build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /tmp/vikit_common_build --parallel "$(nproc)" \
    && cmake --install /tmp/vikit_common_build \
    && mkdir -p /usr/local/lib/cmake/vikit_common \
    && printf '%s\n' \
      'set(vikit_common_SOURCE_DIR "/usr/local")' \
      'set(vikit_common_DIR "/usr/local/lib/cmake/vikit_common")' \
      'set(vikit_common_LIBRARY_DIR "/usr/local/lib")' \
      'set(vikit_common_LIBRARY_DIRS "/usr/local/lib")' \
      'set(vikit_common_INCLUDE_DIR "/usr/local/include")' \
      'set(vikit_common_INCLUDE_DIRS "/usr/local/include")' \
      'set(vikit_common_LIBRARY "/usr/local/lib/libvikit_common.so")' \
      'set(vikit_common_LIBRARIES "/usr/local/lib/libvikit_common.so")' \
      > /usr/local/lib/cmake/vikit_common/vikit_commonConfig.cmake \
    && ldconfig \
    && touch ${WS}/src/rpg_vikit/vikit_common/COLCON_IGNORE ${WS}/src/rpg_vikit/vikit_py/COLCON_IGNORE \
    && rm -rf /tmp/vikit_common_build

# FAST-LIVO2-ROS2 from the vendored tree (U-AMC/FAST-LIVO2-ROS2 @ a09599a).
COPY FAST-LIVO2-ROS2 ${WS}/src/fast_livo
RUN rm -rf ${WS}/src/fast_livo/Log && mkdir -p ${WS}/src/fast_livo/Log/result ${WS}/src/fast_livo/Log/pcd \
    && . /opt/ros/humble/setup.sh && cd ${WS} \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DROS_EDITION=ROS2 -DDISTRO_ROS=humble \
    && test -x ${WS}/install/fast_livo/lib/fast_livo/fastlivo_mapping

# Course-wide RViz mouse scheme: left drag rotates, right/middle drag pans, wheel zooms.
COPY --from=localhost/slam_zero_to_hero:rviz_unified_controls /rviz_unified_controls /opt/rviz_unified_controls
RUN /opt/rviz_unified_controls/install_ros2.sh /opt/rviz_unified_controls

# Upstream's ROS 2 rviz_cfg/*.rviz use the course mouse controller too (only the
# Views > Current > Class line; saved views untouched). hilti.rviz is a ROS 1 file that
# rviz2 cannot load anyway, so it is left alone. The install/share copies are symlinks here.
RUN sed -i -E 's#^( +)Class: rviz_default_plugins/(Orbit|ThirdPersonFollower|TopDownOrtho)$#\1Class: slam_zero_to_hero/Unified\2#' \
      ${WS}/src/fast_livo/rviz_cfg/fast_livo2.rviz ${WS}/src/fast_livo/rviz_cfg/fast_livo2_amc.rviz \
      ${WS}/src/fast_livo/rviz_cfg/M300.rviz ${WS}/src/fast_livo/rviz_cfg/ntu_viral.rviz \
    && ! grep -qE '^ +Class: rviz_default_plugins/(Orbit|ThirdPersonFollower|TopDownOrtho)$' \
      ${WS}/install/fast_livo/share/fast_livo/rviz_cfg/*.rviz

RUN printf '%s\n' '#!/usr/bin/env bash' 'set -e' \
      'source /opt/ros/humble/setup.bash' 'source ${WS}/install/setup.bash' 'exec "$@"' \
      > /entrypoint.sh && chmod +x /entrypoint.sh
WORKDIR /uamc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
