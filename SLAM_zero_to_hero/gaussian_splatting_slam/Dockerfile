# MonoGS -- Gaussian Splatting SLAM (Matsuki, Murai, Kelly, Davison, CVPR 2024)
#
# CUDA 12.8 + PyTorch cu128 so the rasteriser runs on sm_120 (RTX 50-series).
# The two CUDA extensions are compiled for the archs in CUDA_ARCHS.
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="8.6;8.9;12.0"
ARG MONOGS_COMMIT=6c9254c

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCHS} \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget ca-certificates \
      python3 python3-pip python3-dev \
      libgl1 libglib2.0-0 libegl1 libgomp1 libxrender1 libxext6 libsm6 \
      libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
      libglfw3 libglew2.2 libosmesa6 libgl1-mesa-dri \
      xvfb x11-apps imagemagick && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
      torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

RUN python3 -m pip install --no-cache-dir \
      "numpy<2" plyfile==1.0.3 tqdm munch trimesh "evo==1.11.0" open3d==0.18.0 \
      opencv-python-headless==4.10.0.84 torchmetrics==1.4.0 imgviz PyOpenGL glfw PyGLM \
      wandb lpips rich matplotlib==3.5.3

RUN git clone https://github.com/muskie82/MonoGS.git /MonoGS && \
    cd /MonoGS && git checkout ${MONOGS_COMMIT} && \
    git submodule update --init --recursive

WORKDIR /MonoGS

# simple-knn uses FLT_MAX without <cfloat>; newer toolchains no longer pull it in.
RUN sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu && \
    sed -i '1i #include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h && \
    MAX_JOBS=8 python3 -m pip install --no-cache-dir --no-build-isolation \
      submodules/simple-knn submodules/diff-gaussian-rasterization

# Pre-fetch the AlexNet/LPIPS weights --eval needs, so runs work offline.
RUN python3 -c "import lpips; lpips.LPIPS(net='alex')" && \
    python3 -c "from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as L; L(net_type='alex')"

# evo defaults to the TkAgg backend and the image has no tkinter; MonoGS saves
# its ATE plot through evo, which would crash the frontend on the first keyframe.
RUN evo_config set plot_backend Agg
ENV MPLBACKEND=Agg

# Without --eval, slam.py joins the backend without draining the queue the backend
# still writes to, so the process never exits after tracking. patches/ fixes that.
COPY patches /opt/patches
RUN git apply /opt/patches/*.patch

COPY scripts /opt/scripts
RUN chmod +x /opt/scripts/*.sh && mkdir -p /MonoGS/datasets

CMD ["bash"]
