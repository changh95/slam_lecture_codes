# ---------------------------------------------------------------------------
# CUDA toolkit, pruned
# ---------------------------------------------------------------------------
# slam:base has OpenCV, Eigen and easy_profiler but no CUDA, so the toolkit is
# grafted on (as in part2_ch03_07) rather than the base being swapped for
# nvidia/cuda. vilib needs nvcc, the CUDA runtime and the headers. The libraries
# it does not use (cuBLAS, cuFFT, NPP, ...) are removed. compute-sanitizer stays
# for the README exercise.
#
# CUDA 12.8 is the first toolkit that can build for sm_120 (RTX 50xx).
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 AS cuda

# Two traps here:
#  - libcudart_static.a and libcudadevrt.a must survive. nvcc links cudart
#    statically by default, so deleting every *.a breaks even a hello-world.
#  - compat/ must go. It carries a forward-compat libcuda; when that is newer
#    than the host driver, nvidia-container-runtime puts it ahead on the loader
#    path, and forward compat is unsupported on GeForce cards, so every CUDA
#    call fails with cudaErrorCompatNotSupportedOnDevice.
RUN cd /usr/local/cuda-12.8 && \
    rm -rf compat doc share/man extras/demo_suite src && \
    find targets/x86_64-linux/lib -name '*.a' \
         ! -name 'libcudart_static.a' ! -name 'libcudadevrt.a' -delete && \
    rm -f targets/x86_64-linux/lib/libcublas* \
          targets/x86_64-linux/lib/libcufft* \
          targets/x86_64-linux/lib/libcufile* \
          targets/x86_64-linux/lib/libcusolver* \
          targets/x86_64-linux/lib/libcusparse* \
          targets/x86_64-linux/lib/libcurand* \
          targets/x86_64-linux/lib/libnpp* \
          targets/x86_64-linux/lib/libnvblas* \
          targets/x86_64-linux/lib/libnvjpeg* \
          targets/x86_64-linux/lib/libnvrtc*

# ---------------------------------------------------------------------------
FROM slam:base

COPY --from=cuda /usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN ln -s /usr/local/cuda-12.8 /usr/local/cuda && \
    echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# slam:base carries none of the NVIDIA_* hints that nvidia-container-runtime
# keys its injection on. Without these the GPU is not visible in the container
# even with the right run flags.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# RTX 5090 is sm_120. Override for another card, e.g. an RTX 3090:
#   docker build . --build-arg CUDA_ARCH=86 -t slam_zero_to_hero:part5_ch03_08
# Give a plain number: the cuobjdump check below looks for sm_<CUDA_ARCH>.
ARG CUDA_ARCH=120

# rerun C++ SDK, for frontend_rerun: it streams to a viewer on the host. Pinned to the host
# viewer's version (live gRPC streaming needs the SDK and the viewer to match; `rerun --version`).
# The Python SDK, for blueprint.py, is pinned to the same version: an slam:base image built
# earlier may carry an older one. Installed before vilib, so that editing vilib does not
# rebuild them.
RUN cd /tmp && \
    wget -q https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip && \
    cmake -E tar xf rerun_cpp_sdk.zip && \
    cmake -S rerun_cpp_sdk -B rerun_cpp_sdk/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build rerun_cpp_sdk/build -j8 --target install && \
    ldconfig && \
    rm -rf rerun_cpp_sdk rerun_cpp_sdk.zip && \
    pip3 install --break-system-packages rerun-sdk==0.33.0

# vilib (uzh-rpg, BSD-3-Clause) is vendored in third_party/vilib: a copy of the last upstream
# commit (June 2021) with two build fixes in its CMakeLists.txt and comments for the course (see
# third_party/README.md). It is copied in and built on its own, before the rest of the exercise,
# so that editing the examples does not rebuild it.
#
# Unpatched, vilib builds for sm_52 without a word. It calls check_language(CUDA) before
# project(), when CMake has not picked a build tool yet, so the check fails and vilib falls back to
# the old FindCUDA path, which ignores CMAKE_CUDA_ARCHITECTURES. Passing CMAKE_CUDA_COMPILER skips
# the check. sm_52 code still runs on a newer GPU, because the driver compiles the embedded PTX at
# start-up, but it is not what we want to time. cuobjdump verifies the result.
# CMAKE_CUDA_RUNTIME_LIBRARY=Shared makes libvilib.so use libcudart.so, the same CUDA runtime the
# examples link, instead of carrying its own static copy.
WORKDIR /workspace
COPY third_party/vilib /workspace/part5_ch03_08/third_party/vilib
RUN cmake -S /workspace/part5_ch03_08/third_party/vilib -B /tmp/vilib-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/vilib-build -j8 --target install && \
    (cuobjdump --list-elf /usr/local/lib/libvilib.so | grep -q "sm_${CUDA_ARCH}" || \
        (echo "FATAL: libvilib.so was not built for sm_${CUDA_ARCH}" && \
         cuobjdump --list-elf /usr/local/lib/libvilib.so | head -3 && exit 1)) && \
    ldconfig && \
    rm -rf /tmp/vilib-build

COPY . /workspace/part5_ch03_08

# Build the examples
RUN cd /workspace/part5_ch03_08 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part5_ch03_08/build
