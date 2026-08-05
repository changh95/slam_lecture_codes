# syntax=docker/dockerfile:1.7
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-perception-jammy

ARG ROS_DISTRO
ARG WS=/fast_livo_ws

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=${ROS_DISTRO} \
    WS=${WS} \
    CPATH=/opt/ros/${ROS_DISTRO}/include

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libapr1-dev \
    libboost-thread-dev \
    libeigen3-dev \
    libfmt-dev \
    libopencv-dev \
    libpcl-dev \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    ros-${ROS_DISTRO}-ament-cmake \
    ros-${ROS_DISTRO}-ament-cmake-auto \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-rcl-interfaces \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-rclcpp-components \
    ros-${ROS_DISTRO}-rclpy \
    ros-${ROS_DISTRO}-rcutils \
    ros-${ROS_DISTRO}-ros2bag \
    ros-${ROS_DISTRO}-rosbag2 \
    ros-${ROS_DISTRO}-rosbag2-storage-default-plugins \
    ros-${ROS_DISTRO}-rosidl-default-generators \
    ros-${ROS_DISTRO}-rosidl-default-runtime \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-sophus \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# livox_ros_driver2 links against Livox-SDK2 from /usr/local.
RUN git clone --depth=1 https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2 \
    && cmake -S /tmp/Livox-SDK2 -B /tmp/Livox-SDK2/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/Livox-SDK2/build --parallel "$(nproc)" \
    && cmake --install /tmp/Livox-SDK2/build \
    && ldconfig \
    && rm -rf /tmp/Livox-SDK2

WORKDIR ${WS}

RUN mkdir -p ${WS}/src \
    && git clone --depth=1 https://github.com/Livox-SDK/livox_ros_driver2.git ${WS}/src/livox_ros_driver2

RUN git clone --depth=1 https://github.com/U-AMC/rpg_vikit_rational_polynomial.git ${WS}/src/rpg_vikit

# vikit_common is a plain CMake package; install it globally and let colcon build vikit_ros.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && cmake -S ${WS}/src/rpg_vikit/vikit_common -B /tmp/vikit_common_build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /tmp/vikit_common_build --parallel "$(nproc)" \
    && cmake --install /tmp/vikit_common_build \
    && mkdir -p /usr/local/lib/cmake/vikit_common \
    && printf '%s\n' \
      'set(vikit_common_SOURCE_DIR "/usr/local")' \
      'set(vikit_common_DIR "/usr/local/lib/cmake/vikit_common")' \
      'set(vikit_common_INCLUDE_DIR "/usr/local/include")' \
      'set(vikit_common_INCLUDE_DIRS "/usr/local/include")' \
      'set(vikit_common_LIBRARIES "/usr/local/lib/libvikit_common.so")' \
      'set(vikit_common_LIBRARY "/usr/local/lib/libvikit_common.so")' \
      'set(vikit_common_LIBRARY_DIR "/usr/local/lib")' \
      'set(vikit_common_LIBRARY_DIRS "/usr/local/lib")' \
      > /usr/local/lib/cmake/vikit_common/vikit_commonConfig.cmake \
    && ldconfig \
    && touch ${WS}/src/rpg_vikit/vikit_common/COLCON_IGNORE \
    && touch ${WS}/src/rpg_vikit/vikit_py/COLCON_IGNORE \
    && rm -rf /tmp/vikit_common_build

# livox_ros_driver2 switches package metadata by ROS version via build.sh; do it explicitly
# so the whole workspace can be built once together.
RUN cp ${WS}/src/livox_ros_driver2/package_ROS2.xml ${WS}/src/livox_ros_driver2/package.xml \
    && cp -r ${WS}/src/livox_ros_driver2/launch_ROS2 ${WS}/src/livox_ros_driver2/launch

COPY . ${WS}/src/fast_livo

RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --symlink-install --continue-on-error \
      --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DROS_EDITION=ROS2 \
        -DDISTRO_ROS=${ROS_DISTRO}

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'source "/opt/ros/${ROS_DISTRO}/setup.bash"' \
    'source "${WS}/install/setup.bash"' \
    'exec "$@"' \
    > /fast_livo_entrypoint.sh \
    && chmod +x /fast_livo_entrypoint.sh

ENTRYPOINT ["/fast_livo_entrypoint.sh"]
CMD ["bash"]
