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

WORKDIR /data
