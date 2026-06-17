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
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git /tmp/Livox-SDK && \
    cd /tmp/Livox-SDK && \
    mkdir -p build && cd build && cmake .. && make -j$(nproc) && make install && ldconfig && \
    rm -rf /tmp/Livox-SDK

# Build catkin workspace: livox_ros_driver + rpg_vikit (FAST-LIVO2 fork) + FAST-LIVO2
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/Livox-SDK/livox_ros_driver.git && \
    git clone https://github.com/xuankuzcr/rpg_vikit.git && \
    git clone https://github.com/hku-mars/FAST-LIVO2.git

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin_make -j$(nproc) -DCMAKE_BUILD_TYPE=Release

# Smoke-test that the entrypoint binary exists (catches link errors that catkin_make may swallow)
RUN . /opt/ros/noetic/setup.sh && . /catkin_ws/devel/setup.sh && \
    rospack find fast_livo && \
    test -x /catkin_ws/devel/lib/fast_livo/fastlivo_mapping

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
