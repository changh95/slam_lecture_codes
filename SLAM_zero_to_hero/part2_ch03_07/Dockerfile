# ---------------------------------------------------------------------------
# CUDA toolkit, pruned
# ---------------------------------------------------------------------------
# slam:base is not an NVIDIA image and is not cheap to reproduce (besides its
# apt packages it source-builds Ceres, GTSAM, g2o, OpenCV, Pangolin, Sophus and
# more into /usr/local, and the sibling chapters share those layers on disk), so
# CUDA is grafted on rather than the base being swapped for nvidia/cuda.
#
# The full 12.8.1 devel toolkit is 6.6 GB. This chapter needs nvcc, cudart, the
# thrust/cub headers and cublas - fast_gicp calls cuda_add_cublas_to_target -
# so the rest is pruned, leaving about 1.6 GB.
#
# CUDA 12.8 is not a free choice in either direction: it is the FIRST toolkit
# that can emit sm_120 (Blackwell) cubins, and 13.x is the first that CANNOT
# build fast_gicp at all - CCCL 3.0 removed thrust::async, which
# find_voxel_correspondences.cu needs on the required path for both NDTCuda and
# FastVGICPCuda.
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 AS cuda

# Two traps here:
#  - libcudart_static.a and libcudadevrt.a must survive. nvcc links cudart
#    statically by default, so deleting every *.a breaks even a hello-world
#    with "cannot find -lcudart_static".
#  - compat/ must go. It carries a forward-compat libcuda; when that is newer
#    than the host driver, nvidia-container-runtime puts it ahead on the loader
#    path, and forward compat is unsupported on GeForce cards, so every CUDA
#    call fails with cudaErrorCompatNotSupportedOnDevice.
# /bin/sh does not brace-expand, so the libraries are listed in full.
RUN cd /usr/local/cuda-12.8 && \
    rm -rf compat doc share/man extras/demo_suite compute-sanitizer src && \
    find targets/x86_64-linux/lib -name '*.a' \
         ! -name 'libcudart_static.a' ! -name 'libcudadevrt.a' -delete && \
    rm -f targets/x86_64-linux/lib/libcufft* \
          targets/x86_64-linux/lib/libcusolver* \
          targets/x86_64-linux/lib/libcusparse* \
          targets/x86_64-linux/lib/libcurand* \
          targets/x86_64-linux/lib/libnpp* \
          targets/x86_64-linux/lib/libnvjpeg*

# ---------------------------------------------------------------------------
FROM slam:base

COPY --from=cuda /usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN ln -s /usr/local/cuda-12.8 /usr/local/cuda && \
    echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# slam:base carries none of the NVIDIA_* hints that nvidia-container-runtime
# keys its injection on. Without these the hook injects nothing even with the
# right run flags, and cudaGetDeviceCount() returns 35 with zero devices.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# RTX 5090 is sm_120. Override for another card:
#   docker build . --build-arg CUDA_ARCH=86 -t ...
# -arch=sm_NN embeds both an sm_NN cubin and compute_NN PTX, so the result also
# JITs forward onto newer GPUs.
ARG CUDA_ARCH=120

# rerun C++ SDK: the demos stream live to a viewer on the host while they run.
# Pinned to the host viewer version (live gRPC streaming requires the SDK and
# viewer versions to match).
RUN wget -q https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip && \
    cmake -E tar xf rerun_cpp_sdk.zip && \
    cmake -S rerun_cpp_sdk -B rerun_cpp_sdk/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build rerun_cpp_sdk/build -j4 --target install && \
    ldconfig && \
    rm -rf rerun_cpp_sdk rerun_cpp_sdk.zip

# TEASER++: the real solver, so teaser_demo runs certifiably-robust global
# registration rather than a stand-in. Tests/bindings are off - only the
# registration library is needed. Pinned to a master commit for reproducibility.
RUN git clone https://github.com/MIT-SPARK/TEASER-plusplus.git && \
    cd TEASER-plusplus && \
    git checkout 52a9c52ee7d4c838c5e8a75458c33178be5bfb70 && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_DOC=OFF \
        -DBUILD_TEASER_FPFH=OFF \
        -DBUILD_MATLAB_BINDINGS=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF && \
    cmake --build build -j8 --target install && \
    ldconfig && \
    cd .. && rm -rf TEASER-plusplus

# small_gicp: the CPU GICP for the GICP experiment. Header-only for the path
# this chapter uses, but installed properly so find_package works. Pinned.
RUN git clone https://github.com/koide3/small_gicp.git && \
    cd small_gicp && \
    git checkout 8a2d3734f699c042db74ae61d295b0b928163526 && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF && \
    cmake --build build -j8 --target install && \
    ldconfig && \
    cd .. && rm -rf small_gicp

# fast_gicp: the CUDA backends. This is the only source of a GPU NDT - NDTCuda
# is the only NDT in the library and is unconditionally CUDA-gated, and
# small_gicp has no NDT at all.
#
# Upstream master does not build against CUDA 12.8 unpatched, and does not build
# for the right GPU even when it does compile. Two independent problems:
#
#  1. Build-breaking. fast_vgicp_cuda.cuh and ndt_cuda.cuh hand-forward-declare
#     thrust::pair, thrust::device_allocator and thrust::device_vector inside
#     `namespace thrust {`. Since CCCL 2.x those names are aliases into
#     cuda::std reached through an inline ABI namespace, so the redeclarations
#     collide and the two .cu files fail with 131 "thrust::pair is ambiguous"
#     errors. The errors point into /usr/local/cuda/include/thrust/pair.h, which
#     makes it read like a broken CUDA install rather than a fast_gicp bug.
#     Fixed below by reopening the namespace the way thrust itself does, so the
#     forward declarations land in the versioned namespace instead of beside it.
#
#  2. Silent, and worse. The CUDA library is built with legacy
#     cuda_add_library() from FindCUDA, which ignores CMAKE_CUDA_ARCHITECTURES,
#     and the project hard-sets CUDA_NVCC_FLAGS so -DCUDA_NVCC_FLAGS is
#     shadowed too. Unpatched the build SUCCEEDS and emits nvcc's default sm_52.
#     That still RUNS on a 5090 because the fatbin carries compute_52 PTX which
#     the driver JITs - it just takes 3300 ms for the first align() instead of
#     12 ms. CUDA_NVCC_FLAGS_RELEASE is the one knob FindCUDA appends rather
#     than replaces, so the arch goes there, and cuobjdump verifies the result
#     rather than trusting it.
#
# Also note fast_gicp's install() rules sit entirely inside
# if(catkin_FOUND)/elseif(ament_cmake_FOUND). There is no ROS here, so
# `cmake --install` would succeed and install nothing - hence the manual copy.
RUN git clone https://github.com/SMRT-AIST/fast_gicp.git && \
    cd fast_gicp && \
    git checkout 0e7ec1441c99f7be453db2ea216d5de029387417 && \
    git submodule update --init --recursive && \
    for f in include/fast_gicp/cuda/fast_vgicp_cuda.cuh \
             include/fast_gicp/cuda/ndt_cuda.cuh; do \
        sed -i 's|^namespace thrust {|#include <thrust/detail/config.h>\nTHRUST_NAMESPACE_BEGIN|' "$f" && \
        sed -i 's|^}  // namespace thrust|THRUST_NAMESPACE_END|' "$f" && \
        grep -q THRUST_NAMESPACE_BEGIN "$f" || (echo "thrust patch failed on $f" && exit 1); \
    done && \
    ! grep -rq '^namespace thrust {' include/fast_gicp/cuda/ && \
    cmake -S . -B build \
        -DBUILD_VGICP_CUDA=ON \
        -DBUILD_apps=OFF \
        -DBUILD_test=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF \
        -DCUDA_NVCC_FLAGS_RELEASE=-arch=sm_${CUDA_ARCH} && \
    cmake --build build -j8 && \
    cuobjdump --list-elf build/libfast_vgicp_cuda.so | grep -q "sm_${CUDA_ARCH}" || \
        (echo "FATAL: libfast_vgicp_cuda.so was not built for sm_${CUDA_ARCH}" && \
         cuobjdump --list-elf build/libfast_vgicp_cuda.so | head -3 && exit 1) && \
    cp build/libfast_gicp.so build/libfast_vgicp_cuda.so /usr/local/lib/ && \
    cp -r include/fast_gicp /usr/local/include/ && \
    ldconfig && \
    cd .. && rm -rf fast_gicp

WORKDIR /workspace

# Copy project files
COPY . /workspace/part2_ch03_07

# Build the project. TEASER++, small_gicp and fast_gicp are all installed above,
# so every target including the CUDA ones is built against the real libraries.
RUN cd /workspace/part2_ch03_07 && \
    mkdir build && cd build && \
    cmake .. -DUSE_TEASER=ON -DUSE_CUDA=ON && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/part2_ch03_07/build
