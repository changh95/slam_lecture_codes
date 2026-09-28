# ConceptFusion (RSS 2023) — open-set 3D mapping with SAM + OpenCLIP features fused by gradslam.
# CUDA 12.8 + PyTorch cu128 so the kernels run on Blackwell (RTX 50xx, sm_120).
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        python3 python3-pip python3-venv python3-dev \
        libgl1 libegl1 libglib2.0-0 libgomp1 libxrender1 libxext6 libsm6 \
        xvfb x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Everything Python lives in one venv (Ubuntu 24.04 blocks system-wide pip).
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
        numpy==2.1.3 matplotlib==3.8.4 open3d==0.19.0 opencv-python-headless==4.10.0.84 \
        kornia==0.8.1 tyro==0.9.32 h5py imageio natsort pyyaml plotly tqdm \
        open_clip_torch==2.32.0 rerun-sdk==0.33.0

# segment-anything (pinned commit)
RUN pip install --no-cache-dir \
        git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf

# gradslam, `conceptfusion` branch (the main branch has no feature fusion)
RUN git clone -b conceptfusion https://github.com/gradslam/gradslam.git /opt/gradslam && \
    cd /opt/gradslam && git checkout 59ca872 && pip install --no-cache-dir --no-deps -e .

# ConceptFusion itself (pinned commit)
RUN git clone https://github.com/concept-fusion/concept-fusion.git /opt/concept-fusion && \
    cd /opt/concept-fusion && git checkout 4457c1f

COPY scripts/ /opt/cf_tools/
RUN chmod +x /opt/cf_tools/*.sh

# Weights live on the host (see scripts/download_weights.sh), mounted at /weights.
ENV HF_HOME=/weights/hf \
    SAM_CKPT=/weights/sam_vit_h_4b8939.pth \
    PYTHONUNBUFFERED=1

WORKDIR /opt/concept-fusion/examples
CMD ["/opt/cf_tools/run_demo.sh"]
