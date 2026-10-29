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

# vilib (uzh-rpg, BSD-3-Clause), pinned to the last upstream commit (June 2021).
# It does not build unpatched with GCC 13, and builds for the wrong GPU unless told:
#  1. -Werror. GCC 13's -Wall includes -Woverloaded-virtual, which flags
#     DetectorBase::detect() being hidden by FASTGPU's overloads, and vilib
#     compiles with -Werror -Wfatal-errors. The warning is harmless; the flags go.
#  2. libvilib.so calls cv::imshow, cv::FastFeatureDetector, cv::circle, ... but
#     links only opencv_core, so every program that links it fails with undefined
#     references. The modules it uses are added to its link line.
#  3. Unpatched, vilib builds for sm_52 without a word. It calls
#     check_language(CUDA) before project(), when CMake has not picked a build tool
#     yet, so the check fails and vilib falls back to the old FindCUDA path, which
#     ignores CMAKE_CUDA_ARCHITECTURES. Passing CMAKE_CUDA_COMPILER skips the check.
#     sm_52 code still runs on a newer GPU, because the driver compiles the
#     embedded PTX at start-up, but it is not what we want to time. cuobjdump
#     verifies the result.
# CMAKE_CUDA_RUNTIME_LIBRARY=Shared makes libvilib.so use libcudart.so, the same
# CUDA runtime the examples link, instead of carrying its own static copy.
RUN git clone https://github.com/uzh-rpg/vilib.git /tmp/vilib && \
    cd /tmp/vilib && \
    git checkout 7a12fe1db78ca755876469979394076f10fe577c && \
    sed -i 's/ -Werror -Wfatal-errors//' CMakeLists.txt && \
    sed -i 's/find_package(OpenCV REQUIRED COMPONENTS core)/find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d highgui)/' CMakeLists.txt && \
    sed -i 's/^  opencv_core Eigen3::Eigen/  opencv_core opencv_imgproc opencv_features2d opencv_highgui Eigen3::Eigen/' CMakeLists.txt && \
    ! grep -q Werror CMakeLists.txt && \
    grep -q 'COMPONENTS core imgproc features2d highgui)' CMakeLists.txt && \
    grep -q 'opencv_highgui Eigen3::Eigen' CMakeLists.txt && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build -j8 --target install && \
    (cuobjdump --list-elf /usr/local/lib/libvilib.so | grep -q "sm_${CUDA_ARCH}" || \
        (echo "FATAL: libvilib.so was not built for sm_${CUDA_ARCH}" && \
         cuobjdump --list-elf /usr/local/lib/libvilib.so | head -3 && exit 1)) && \
    ldconfig && \
    cd / && rm -rf /tmp/vilib

WORKDIR /workspace
COPY . /workspace/part5_ch03_08

# Build the examples
RUN cd /workspace/part5_ch03_08 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part5_ch03_08/build
