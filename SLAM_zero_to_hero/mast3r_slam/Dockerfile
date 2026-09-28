# MASt3R-SLAM (CVPR 2025) on CUDA 12.8 + PyTorch 2.7.1 cu128, so its CUDA
# kernels run on Blackwell (sm_120) as well as Ampere/Ada.
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git wget curl ca-certificates \
        libgl1 libegl1 libgl1-mesa-dri libglx-mesa0 libglu1-mesa libglib2.0-0 libsm6 \
        libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
        xvfb x11-utils imagemagick && \
    rm -rf /var/lib/apt/lists/*

# MASt3R ViT-L metric checkpoint + retrieval head + codebook (~3.0 GB).
# Kept in its own early layer so rebuilds of the code below do not re-download.
RUN mkdir -p /checkpoints && cd /checkpoints && \
    for f in MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
             MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
             MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl; do \
        wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/$f || exit 1; \
    done && ls -l /checkpoints

# Python 3.11 (upstream's version) in a venv managed by uv
RUN curl -LsSf https://astral.sh/uv/0.8.22/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    uv python install 3.11 && uv venv -p 3.11 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH

RUN uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install setuptools==70.0.0 wheel ninja numpy==1.26.4 "Cython<0.30" PyOpenGL click glfw

# Compile every CUDA extension for Ampere, Ada and Blackwell
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0" MAX_JOBS=8

# lietorch, pinned
RUN uv pip install --no-build-isolation \
        "lietorch @ git+https://github.com/princeton-vl/lietorch.git@e7df86554156b36846008d8ddbcc4d8521a16554"

# MASt3R-SLAM, pinned commit
RUN git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git /MASt3R-SLAM && \
    cd /MASt3R-SLAM && git checkout e6f4e3d && git submodule update --init --recursive
WORKDIR /MASt3R-SLAM

# sm_120 patches: upstream hard-codes gencodes up to sm_86 in setup.py, and
# curope compiles for whatever torch reports; make both follow TORCH_CUDA_ARCH_LIST,
# and build the backend even though no GPU is visible during podman build.
# torch 2.7 also dropped AT_DISPATCH on the deprecated Tensor.type(): use scalar_type().
RUN sed -i 's#AT_DISPATCH_FLOATING_TYPES_AND_HALF(\([A-Za-z0-9_]*\)\.type()#AT_DISPATCH_FLOATING_TYPES_AND_HALF(\1.scalar_type()#' \
        mast3r_slam/backend/src/matching_kernels.cu thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu && \
    grep -q "D11.scalar_type()" mast3r_slam/backend/src/matching_kernels.cu && \
    sed -i 's#"-gencode=arch=compute_86,code=sm_86",#"-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_89,code=sm_89", "-gencode=arch=compute_120,code=sm_120",#' setup.py && \
    grep -q compute_120 setup.py && \
    sed -i 's#^all_cuda_archs = .*#all_cuda_archs = []#' thirdparty/mast3r/dust3r/croco/models/curope/setup.py && \
    sed -i 's#lietorch.git"#lietorch.git@e7df86554156b36846008d8ddbcc4d8521a16554"#' pyproject.toml && \
    sed -i 's#^has_cuda = torch.cuda.is_available()#has_cuda = True#' setup.py && \
    uv pip install --no-build-isolation -e thirdparty/mast3r
RUN uv pip install --no-build-isolation -e thirdparty/in3d
# torch 2.7 no longer exposes torch::linalg in extension headers; linalg_norm with
# ord=None, dim=None is the flattened 2-norm, i.e. Tensor::norm().
RUN sed -i 's#torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {})#dx.norm()#' \
        mast3r_slam/backend/src/gn_kernels.cu && \
    ! grep -q "torch::linalg" mast3r_slam/backend/src/gn_kernels.cu && \
    uv pip install --no-build-isolation -e . && \
    ln -s /checkpoints /MASt3R-SLAM/checkpoints && \
    python -c "import torch, mast3r_slam_backends, lietorch, curope, imgui; print('extensions ok')"

# moderngl/glcontext dlopen the unversioned libGL.so / libEGL.so, which only the
# -dev packages provide; pyrealsense2 (imported unconditionally) needs libusb;
# mesa-utils for glxinfo when debugging the viewer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl-dev libegl-dev libusb-1.0-0 mesa-utils && \
    rm -rf /var/lib/apt/lists/* && \
    python -c "import main; print('main.py imports ok')"

# torch >= 2.6 defaults torch.load to weights_only=True, and the official MASt3R
# checkpoints pickle an argparse.Namespace; they are Naver's own files, so allow it.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

COPY scripts/ /opt/mast3r_demo/
RUN chmod +x /opt/mast3r_demo/*.sh

WORKDIR /MASt3R-SLAM
