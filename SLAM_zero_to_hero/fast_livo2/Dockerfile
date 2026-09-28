# FAST-LIVO2: Tightly-coupled LiDAR-Inertial-Visual Odometry (HKU-MARS)
# Repo: https://github.com/hku-mars/FAST-LIVO2
# Sensors: Livox/Velodyne LiDAR + IMU + Camera
#
# Datasets: HKU-MARS distributes the FAST-LIVO2 reference bags (Retail_Street,
# HKUST_Red_Sculpture, ...) via OneDrive. Links rotate; this Dockerfile bakes
# the build environment but mounts the bag at run time.

FROM docker.io/ros:noetic

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates \
      libeigen3-dev libpcl-dev libopencv-dev libgoogle-glog-dev libgflags-dev \
      libboost-all-dev libusb-1.0-0-dev \
      ros-noetic-pcl-ros ros-noetic-eigen-conversions ros-noetic-cv-bridge \
      ros-noetic-image-transport ros-noetic-tf2-eigen ros-noetic-tf2-sensor-msgs \
      python3-catkin-tools && \
    rm -rf /var/lib/apt/lists/*

# Old non-templated Sophus -- FAST-LIVO2 + rpg_vikit_fast_livo2 rely on the
# pre-templated SE3/SO3 API; commit a621ff predates the templating rewrite.
# GCC>=11 rejects `std::complex::real() = x;` (rvalue); patch to the setter form.
RUN git clone https://github.com/strasdat/Sophus.git /tmp/Sophus && \
    cd /tmp/Sophus && git checkout a621ff && \
    sed -i 's/unit_complex_\.real() = 1\.;/unit_complex_.real(1.);/' sophus/so2.cpp && \
    sed -i 's/unit_complex_\.imag() = 0\.;/unit_complex_.imag(0.);/' sophus/so2.cpp && \
    mkdir -p build && cd build && \
    cmake .. -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF \
             -DCMAKE_CXX_FLAGS="-Wno-error=deprecated-declarations" && \
    make -j$(nproc) && make install && ldconfig && \
    mkdir -p /usr/local/share/sophus/cmake && \
    printf '%s\n' \
      'set(Sophus_INCLUDE_DIRS "/usr/local/include")' \
      'set(Sophus_LIBRARIES "/usr/local/lib/libSophus.so")' \
      'set(SOPHUS_INCLUDE_DIRS "/usr/local/include")' \
      'set(SOPHUS_LIBRARIES "/usr/local/lib/libSophus.so")' \
      'set(Sophus_FOUND TRUE)' \
      > /usr/local/share/sophus/cmake/SophusConfig.cmake && \
    rm -rf /tmp/Sophus
ENV Sophus_DIR=/usr/local/share/sophus/cmake

# Livox-SDK v1 (FAST-LIVO2 uses the original SDK + livox_ros_driver, NOT SDK2/driver2)
# Upstream commits pinned (the versions verified on 2026-09-27).
ARG LIVOX_SDK_COMMIT=9306596a2bf15c1343bc023b497465ed0a32909d
ARG LIVOX_DRIVER_COMMIT=3d240d5666129e1a3052e78ee8487a04b08fdda3
ARG VIKIT_COMMIT=6c886c8e5d83997806e00294826d528cea3581dd
ARG FAST_LIVO2_COMMIT=0d2c0346107b75b59934975adec9a6eeeb913c64

RUN git clone https://github.com/Livox-SDK/Livox-SDK.git /tmp/Livox-SDK && \
    cd /tmp/Livox-SDK && git checkout ${LIVOX_SDK_COMMIT} && \
    mkdir -p build && cd build && cmake .. && make -j$(nproc) && make install && ldconfig && \
    rm -rf /tmp/Livox-SDK

# Build catkin workspace: livox_ros_driver + rpg_vikit (FAST-LIVO2 fork) + FAST-LIVO2
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git -C livox_ros_driver checkout ${LIVOX_DRIVER_COMMIT} && \
    git clone https://github.com/xuankuzcr/rpg_vikit.git && \
    git -C rpg_vikit checkout ${VIKIT_COMMIT} && \
    git clone https://github.com/hku-mars/FAST-LIVO2.git && \
    git -C FAST-LIVO2 checkout ${FAST_LIVO2_COMMIT}

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc) -DCMAKE_BUILD_TYPE=Release

# Smoke-test that the entrypoint binary exists (catches link errors that catkin_make may swallow)
RUN . /opt/ros/noetic/setup.sh && . /catkin_ws/devel/setup.sh && \
    rospack find fast_livo && \
    test -x /catkin_ws/devel/lib/fast_livo/fastlivo_mapping

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

# GUI prerequisites, both of which were missing:
#   ros-noetic-rviz -- mapping_avia.launch defaults to rviz:=true and points at the
#     bundled rviz_cfg/fast_livo2.rviz, but rviz was never installed, so the launch
#     file's own default failed.
#   ros-noetic-compressed-image-transport -- the launch file runs an image_transport
#     `republish compressed in:=... raw out:=...` node. Without the plugin only
#     'raw_sub' is registered, so the republish node cannot subscribe and the image
#     view stays empty on bags that ship compressed images.
# Kept as its own layer after catkin_make so the GUI costs one cheap rebuild.
# xvfb + x11-utils (xdpyinfo) + imagemagick (import): headless rviz screenshots
#   via capture_rviz.sh, no host X server involved.
RUN apt-get update -y && apt-get install -y \
      ros-noetic-rviz ros-noetic-compressed-image-transport \
      xvfb x11-utils imagemagick && \
    rm -rf /var/lib/apt/lists/*

# Course-wide RViz mouse scheme: left drag rotates, right/middle drag pans, wheel zooms.
COPY --from=localhost/slam_zero_to_hero:rviz_unified_controls /rviz_unified_controls /opt/rviz_unified_controls
RUN /opt/rviz_unified_controls/install_ros1.sh /opt/rviz_unified_controls
# Switch upstream's rviz configs to it. Only Views > Current is touched (6-space
# indent, no list dash); the Saved views keep rviz/Orbit.
RUN sed -i -E 's#^      Class: rviz/ThirdPersonFollower$#      Class: slam_zero_to_hero/UnifiedThirdPersonFollower#' \
      /catkin_ws/src/FAST-LIVO2/rviz_cfg/*.rviz && \
    test "$(grep -l 'Class: slam_zero_to_hero/UnifiedThirdPersonFollower' /catkin_ws/src/FAST-LIVO2/rviz_cfg/*.rviz | wc -l)" -eq 4

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
