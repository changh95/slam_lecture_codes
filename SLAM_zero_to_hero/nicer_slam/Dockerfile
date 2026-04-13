FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,utility,graphics

# System deps (NICER-SLAM needs these for the hash encoder build + 3D libs)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libosmesa6-dev libgtk-3-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (architecture-aware: x86_64 or aarch64)
RUN ARCH=$(uname -m) && \
    curl -sSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh" -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone NICER-SLAM (cvg ETH Zurich - monocular RGB neural implicit SLAM)
RUN git clone --recursive https://github.com/cvg/nicer-slam.git /nicer-slam

WORKDIR /nicer-slam

# Create the main conda environment from env_yamls/nicer-slam.yaml
# This installs Python 3.8, PyTorch 1.11 (cu113), Open3D, Trimesh, PyMCubes,
# kornia, LPIPS, OpenCV, Embree, and all the other deps.
RUN conda env create -f env_yamls/nicer-slam.yaml && \
    conda clean -a -y

# Also create the Omnidata and GMFlow preprocessing environments
RUN conda env create -f env_yamls/omnidata.yaml && \
    conda env create -f env_yamls/gmflow.yaml && \
    conda clean -a -y

# Activate nicer-slam env by default in interactive shells
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate nicer-slam" \
    >> /root/.bashrc

# Download Omnidata pretrained weights (for mono depth / normal preprocessing)
# These are ~200 MB. The repo script expects them in preprocess/omnidata/pretrained_models/
RUN mkdir -p /nicer-slam/preprocess/omnidata/pretrained_models && \
    cd /nicer-slam/preprocess/omnidata/pretrained_models && \
    wget -q https://zenodo.org/record/10447888/files/omnidata_dpt_depth_v2.ckpt || true && \
    wget -q https://zenodo.org/record/10447888/files/omnidata_dpt_normal_v2.ckpt || true

VOLUME /data
VOLUME /output
WORKDIR /nicer-slam

# Default entry: bash with nicer-slam env activated
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate nicer-slam && exec \"$@\"", "bash"]
CMD ["bash"]
