# Based on koide3/glim/docker/ubuntu/Dockerfile.gcc
#
# CUDA 12.9.1, deliberately NOT 13.x. sm_120 (RTX 5090) needs CUDA >= 12.8, and
# 12.9.1 is the newest 12.x -- the last CUDA that gtsam_points v1.0.0 can be
# made to compile against. CUDA 13 breaks it three separate ways:
#   1. CCCL 3.x deleted <cub/iterator/transform_input_iterator.cuh> and
#      <thrust/system/cuda/future.h>; v1.0.0 includes both. Fixing that means
#      rewriting the CUB reduction iterators, not adding an include path.
#   2. cudaGraphAddDependencies() gained a cudaGraphEdgeData* parameter.
#   3. The 13.1 image's forward-compat libcuda (590.44.01) is NEWER than this
#      host driver (580.126.18), so nvidia-container-runtime prepends
#      /usr/local/cuda/compat to the loader path. Forward compat is supported
#      only on data-center GPUs, so on a GeForce card every CUDA call fails with
#      cudaErrorCompatNotSupportedOnDevice. 12.9.1 ships compat 575.57.08 < 580,
#      so the runtime leaves the host driver in place and this cannot happen.
FROM docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates ninja-build \
    libeigen3-dev libpcl-dev libopencv-dev \
    libboost-all-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libtbb-dev \
    libspdlog-dev libfmt-dev \
    libglfw3-dev libglm-dev libpng-dev \
    libgtest-dev nlohmann-json3-dev && \
    rm -rf /var/lib/apt/lists/*

# Build GTSAM 4.2.0 (matches gtsam_points/glim v1.0.0)
RUN git clone https://github.com/borglab/gtsam.git -b 4.2.0 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DGTSAM_USE_SYSTEM_EIGEN=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds" && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# Build iridescence (visualization)
RUN git clone https://github.com/koide3/iridescence.git --recursive && \
    cd iridescence && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf iridescence

# Build gtsam_points (pin to v1.0.0 for GTSAM 4.2.0 compat) with GPU support.
# Two source fixes are needed, and neither can be expressed as a cmake -D flag:
#
#  * CMAKE_CUDA_ARCHITECTURES. Upstream does
#    `set(CMAKE_CUDA_ARCHITECTURES "native" ... FORCE)` on CMake >= 3.24, which
#    silently discards -DCMAKE_CUDA_ARCHITECTURES=120. That would be survivable
#    on this host, except `podman build` has no GPU: nvcc then resolves
#    -arch=native to its own default (sm_52) with nothing but a warning, and the
#    image ships cubins that cannot run on a 5090 ("no kernel image is available
#    for execution on the device"). Pin the architecture instead.
#
#  * thrust forward declarations. v1.0.0 hand-writes `namespace thrust {
#    template<...> class pair; }` (and device_vector / device_ptr) to avoid
#    pulling thrust into three public headers. Every CCCL >= 2.4 -- i.e. every
#    CUDA that can target sm_120 -- puts the real types in a *versioned inline*
#    namespace (thrust::THRUST_200802_SM_1200_NS::pair), so those hand-written
#    declarations introduce second, unrelated types and ~400 "thrust::pair is
#    ambiguous" errors follow. Swap them for the real headers; consumers get the
#    include path automatically because gtsam_points_cuda links CUDA::cudart
#    PUBLIC.
RUN git clone https://github.com/koide3/gtsam_points.git -b v1.0.0 --recursive && \
    cd gtsam_points && \
    sed -i 's/set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)/set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)/' CMakeLists.txt && \
    grep -q 'set(CMAKE_CUDA_ARCHITECTURES 120 CACHE' CMakeLists.txt && \
    sed -i '/^namespace thrust {$/,/^}  \/\/ namespace thrust$/c\#include <thrust\/pair.h>' include/gtsam_points/types/gaussian_voxelmap_gpu.hpp && \
    sed -i '/^namespace thrust {$/,/^}  \/\/ namespace thrust$/c\#include <thrust\/device_vector.h>' include/gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp && \
    sed -i '/^namespace thrust {$/,/^}$/c\#include <thrust\/device_ptr.h>' include/gtsam_points/factors/nonlinear_factor_gpu.hpp && \
    ! grep -rq '^namespace thrust' include src && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_WITH_CUDA=ON \
             -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds" && \
    make -j$(nproc) && make install && ldconfig && \
    cuobjdump --list-elf /usr/local/lib/libgtsam_points_cuda.so | grep -q sm_120 && \
    cd / && rm -rf gtsam_points

# Build GLIM v1.0.0. BUILD_WITH_CUDA already defaults to ON here, but the
# installed gtsam_points-config.cmake sets the same variable and is read first,
# so the flag that actually decides is gtsam_points'. State it anyway: this is
# what produces libodometry_estimation_gpu.so.
RUN git clone https://github.com/koide3/glim.git -b v1.0.0 --recursive && \
    cd glim && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=ON && \
    make -j$(nproc) && make install && ldconfig && \
    test -f /usr/local/lib/libodometry_estimation_gpu.so

# A plain (non-ROS) glim build installs NO executables — it has no add_executable
# and no main() anywhere. Upstream's data-ingesting binaries (glim_rosnode,
# glim_rosbag, offline_viewer) live in the separate koide3/glim_ros1 package,
# which needs ROS 1 Noetic and therefore Ubuntu 20.04 — impossible on this
# 24.04 CUDA base. So without the driver below, this image cannot read a dataset
# at all. glim_kitti feeds KITTI velodyne .bin scans straight through the real
# GLIM pipeline (TimeKeeper -> CloudPreprocessor -> Async{Odometry,SubMapping,
# GlobalMapping}), mirroring GlimROS::insert_frame().
COPY kitti /opt/glim_kitti/src
RUN cmake -S /opt/glim_kitti/src -B /opt/glim_kitti/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/glim_kitti/build -j$(nproc) --target install && \
    rm -rf /opt/glim_kitti/build

# glim's non-ROS build does not install its config JSONs (the install(DIRECTORY
# config ...) rule is inside a ROS-only branch), so the only copy is in the
# source clone at /glim/config. Ship the KITTI-tuned variant on a stable path.
COPY config_kitti /usr/local/share/glim_kitti/config
COPY config_kitti_posegraph /usr/local/share/glim_kitti/config_posegraph
COPY config_kitti_gpu /usr/local/share/glim_kitti/config_gpu

# ROS 2 Jazzy + glim_ros2 v1.0.0. Appended last so the CUDA/GTSAM/gtsam_points/
# glim layers above stay cached (~35 min to rebuild).
# v1.0.0 is the pin because koide3 tags glim and glim_ros2 in lockstep and
# glim_ros2 v1.0.0's package.xml declares <depend>glim</depend> UNVERSIONED, so
# it links the glim v1.0.0 already installed here. glim_ros2 v1.2.x pairs with
# glim v1.2.x, whose gtsam_points targets GTSAM 4.2a9/4.3a1 instead of the 4.2.0
# pinned above - that would cascade into a full CUDA rebuild.
# Ubuntu 24.04 => the ROS 2 distro is Jazzy.
ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=jazzy

RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-image-transport \
      ros-${ROS_DISTRO}-ament-cmake-auto && \
    rm -rf /var/lib/apt/lists/*

# Build glim_ros2 against the glim installed above. Plain cmake, not colcon:
# ament_cmake_auto's find_package(glim) resolves the plain-CMake
# /usr/local/lib/cmake/glim/glim-config.cmake fine, so a colcon workspace (and
# python3-colcon-common-extensions) buys nothing here. Each .cpp #defines
# GLIM_ROS2 itself, so no extra -D is needed. ament_auto_add_executable installs
# executables to lib/glim_ros/, libraries to lib/ - hence the paths below.
# The asserts make a silently missing artifact fail the build; the last one
# actually EXECUTES glim_rosbag (it prints usage and exits 0 before touching
# CUDA, so it works in a GPU-less build) which proves every dynamic dependency
# (glim, gtsam, gtsam_points, iridescence, rclcpp, rosbag2, cv_bridge) resolves.
RUN git clone https://github.com/koide3/glim_ros2.git -b v1.0.0 /opt/glim_ros2 && \
    . /opt/ros/${ROS_DISTRO}/setup.sh && \
    cmake -S /opt/glim_ros2 -B /opt/glim_ros2/build \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_WITH_CUDA=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /opt/glim_ros2/build -j$(nproc) && \
    cmake --install /opt/glim_ros2/build && \
    ldconfig && \
    test -x /usr/local/lib/glim_ros/glim_rosbag && \
    test -f /usr/local/lib/librviz_viewer.so && \
    test -f /usr/local/lib/libglim_ros.so && \
    ! ldd /usr/local/lib/glim_ros/glim_rosbag | grep "not found" && \
    /usr/local/lib/glim_ros/glim_rosbag 2>&1 | grep -q "usage: glim_rosbag" && \
    rm -rf /opt/glim_ros2/build

# Register glim in the ament index. Without this, running glim_rosbag WITHOUT an
# absolute -p config_path:= aborts with
#   terminate called ... ament_index_cpp::PackageNotFoundError: package 'glim'
#   not found, searching: [/usr/local, /opt/ros/jazzy]
# because GlimROS treats a relative config_path as relative to glim's share dir.
# An upstream colcon build gets this for free (glim's CMakeLists calls
# ament_package() only when ROS_VERSION is set, which it was not here). This also
# closes the "the non-ROS build never installs its config JSONs" gap.
RUN mkdir -p /usr/local/share/ament_index/resource_index/packages /usr/local/share/glim && \
    touch /usr/local/share/ament_index/resource_index/packages/glim && \
    cp -r /glim/config /usr/local/share/glim/config && \
    test -f /usr/local/share/glim/config/config.json

# Thin wrappers so no one has to remember to source setup.bash. They are not
# optional: rosbag2 finds its sqlite3 storage plugin through pluginlib, i.e.
# through AMENT_PREFIX_PATH, and rmw_implementation dlopens librmw_fastrtps_cpp
# from /opt/ros/jazzy/lib.
RUN printf '#!/bin/bash\n. /opt/ros/%s/setup.bash\nexport AMENT_PREFIX_PATH=/usr/local:$AMENT_PREFIX_PATH\nexec /usr/local/lib/glim_ros/%s "$@"\n' \
      "${ROS_DISTRO}" glim_rosbag > /usr/local/bin/glim_rosbag && \
    printf '#!/bin/bash\n. /opt/ros/%s/setup.bash\nexport AMENT_PREFIX_PATH=/usr/local:$AMENT_PREFIX_PATH\nexec /usr/local/lib/glim_ros/%s "$@"\n' \
      "${ROS_DISTRO}" glim_rosnode > /usr/local/bin/glim_rosnode && \
    printf '#!/bin/bash\n. /opt/ros/%s/setup.bash\nexport AMENT_PREFIX_PATH=/usr/local:$AMENT_PREFIX_PATH\nexec /usr/local/lib/glim_ros/%s "$@"\n' \
      "${ROS_DISTRO}" offline_viewer > /usr/local/bin/glim_offline_viewer && \
    chmod +x /usr/local/bin/glim_rosbag /usr/local/bin/glim_rosnode /usr/local/bin/glim_offline_viewer && \
    glim_rosbag 2>&1 | grep -q "usage: glim_rosbag"

# Korea_drive config set (headless), plus a viewer variant derived from it so
# there is a single source of truth for every other key.
COPY config_korea /usr/local/share/glim_korea/config
RUN cp -r /usr/local/share/glim_korea/config /usr/local/share/glim_korea/config_viewer && \
    sed -i 's|"extension_modules": \[\]|"extension_modules": ["libstandard_viewer.so", "librviz_viewer.so"]|' \
      /usr/local/share/glim_korea/config_viewer/config_ros.json && \
    grep -q 'librviz_viewer.so' /usr/local/share/glim_korea/config_viewer/config_ros.json

WORKDIR /data
