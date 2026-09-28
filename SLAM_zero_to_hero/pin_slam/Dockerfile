FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Python 3.10 + the X/GL libraries Open3D's GUI (GLFW + Filament) needs.
# xvfb / x11-apps / imagemagick are only for headless screenshots.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates python3 python3-pip python3-dev \
        libgl1 libegl1 libgomp1 libglib2.0-0 libx11-6 libxext6 libxrender1 \
        libxi6 libxrandr2 libxinerama1 libxcursor1 libxxf86vm1 libxkbcommon0 \
        libc++1 libc++abi1 libusb-1.0-0 libtbb12 \
        xvfb x11-apps x11-utils imagemagick && \
    rm -rf /var/lib/apt/lists/*

# PIN-SLAM v1.1.1 (commit 5976f3f)
RUN git clone https://github.com/PRBonn/PIN_SLAM.git /PIN_SLAM && \
    cd /PIN_SLAM && git checkout 5976f3f
WORKDIR /PIN_SLAM

# RTX 50xx (sm_120) needs CUDA 12.8+ kernels: torch cu128 wheels.
# PIN-SLAM is pure PyTorch (no custom CUDA extension), so the wheel's bundled
# CUDA runtime is all it needs.
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir -r requirements.txt

# evo (trajectory plots at the end of a run) defaults to matplotlib's TkAgg
# backend, which crashes without tkinter or without an X display.
RUN evo_config set plot_backend Agg

COPY scripts/ /PIN_SLAM/demo_scripts/

WORKDIR /PIN_SLAM
