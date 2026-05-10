# Visual Place Recognition Tutorial — uv-based build
# Replaces the Anaconda+TensorRT image with a slim Python 3.11 base + uv.
# Skips TensorFlow on purpose (the upstream HDC-DELF + tensorflow_hub combo
# has a long-standing tf-keras vs tensorflow API mismatch). Six of the seven
# descriptors are torch-based and work without TF: AlexNet, NetVLAD,
# PatchNetVLAD, CosPlace, EigenPlaces, SAD.
FROM docker.io/python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

# System deps:
#   git           - to clone VPR_Tutorial
#   libgl1, libglib2.0-0 - OpenCV runtime
#   libgomp1      - OpenMP runtime that scikit-image / faiss link against
#   curl, ca-certificates - downloading uv + model weights
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary).
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

# Clone the VPR Tutorial.
RUN git clone --depth 1 https://github.com/stschubert/VPR_Tutorial.git /VPR_Tutorial
WORKDIR /VPR_Tutorial

# Create a project venv with uv and install the requirements.
# `--system-site-packages` keeps it small; we install everything via uv pip.
RUN uv venv /opt/venv --python 3.11
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Core deps for VPR_Tutorial (torch path only — no TF/tensorflow_hub).
# - opencv-python-headless avoids X11/GTK pulls; the demo only matplotlib-renders
# - faiss-cpu for nearest-neighbour search (replaces patchnetvlad's pinned
#   faiss==1.5.3 which has no Python 3.11 wheels)
# - patchnetvlad pinned to faiss==1.5.3 in its metadata; install with --no-deps
#   and provide faiss-cpu separately. The runtime code just `import faiss`.
# Install CUDA-enabled PyTorch wheels with Blackwell (sm_120) support.
# Torch < 2.7 is built for sm_50-sm_90 only; RTX 5090 is sm_120, so we need
# torch >= 2.7 with cu128. The runtime CUDA libs ship with the wheels;
# the host nvidia driver's libcuda.so.1 is mounted at `podman run` time.
RUN uv pip install --no-cache \
    --index-strategy=unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7" "torchvision>=0.22"

RUN uv pip install --no-cache \
    "numpy<2" \
    "opencv-python-headless<5" \
    pillow \
    matplotlib \
    natsort \
    scikit-image \
    scikit-learn \
    tqdm \
    "faiss-cpu>=1.7,<2" \
    "setuptools<81"

# Install patchnetvlad without its dependency resolver (avoids faiss==1.5.3 pin).
RUN uv pip install --no-cache --no-deps patchnetvlad

# Patch CosPlace + EigenPlaces feature extractors: torch >= 2.7 makes
# `torch.hub.load(..., trust_repo=...)` mandatory. The default is "ask" which
# raises EOFError when stdin is not a TTY (i.e. inside a non-interactive
# `podman run`). Append `trust_repo=True` to both call sites.
RUN sed -i 's|torch.hub.load("gmberton/cosplace", "get_trained_model",|torch.hub.load("gmberton/cosplace", "get_trained_model", trust_repo=True,|' \
        /VPR_Tutorial/feature_extraction/feature_extractor_cosplace.py && \
    sed -i 's|torch.hub.load("gmberton/eigenplaces", "get_trained_model",|torch.hub.load("gmberton/eigenplaces", "get_trained_model", trust_repo=True,|' \
        /VPR_Tutorial/feature_extraction/feature_extractor_eigenplaces.py

# Pre-populate torch.hub trusted_list. The EigenPlaces hubconf.py internally
# calls torch.hub.load("gmberton/cosplace", ...) without trust_repo, so we
# need both repos on the trusted list (else stdin trust prompt -> EOFError).
RUN mkdir -p /root/.cache/torch/hub && \
    printf 'gmberton_cosplace\ngmberton_eigenplaces\npytorch_vision\n' \
        > /root/.cache/torch/hub/trusted_list

# Install VPR_Tutorial as an editable package (no deps -- we just installed them above).
RUN uv pip install --no-cache --no-deps -e .

# Default workdir: VPR_Tutorial; default cmd shows what's available.
CMD ["bash", "-lc", "echo 'VPR Tutorial container. Try:' && echo '  python3 demo.py --descriptor CosPlace --dataset GardensPoint' && exec bash"]
