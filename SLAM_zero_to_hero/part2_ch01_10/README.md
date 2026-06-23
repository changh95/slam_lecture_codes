# Deep Learning-based Visual Place Recognition

Python exercise using the [VPR_Tutorial](https://github.com/stschubert/VPR_Tutorial) framework with switchable descriptors and datasets.

---

## Project Structure

```
part2_ch01_10/
├── README.md
├── Dockerfile                  # uv-based Python 3.11 + CUDA torch image
├── compare_descriptors.py      # Multi-descriptor comparison script
├── force_gpu.py                # Build-time patch: pin learned descriptors to GPU
└── visualizations/             # Reference output images
```

---

## Build

```bash
docker build . -t slam:vpr
```

Dependencies are installed inside the image:
- **Python 3.11** (slim base) with **uv** package manager
- **PyTorch >= 2.7** with CUDA 12.8 wheels (supports sm_120 / RTX 5090)
- **VPR_Tutorial** (cloned from GitHub): AlexNet, NetVLAD, PatchNetVLAD, CosPlace, EigenPlaces, SAD descriptors
- **faiss-cpu**, **scikit-image**, **scikit-learn**, **opencv-python-headless**, **matplotlib**, **PyQt5**
- TensorFlow is intentionally omitted; all six supported descriptors are torch-based.

---

## Run

> **GPU is required.** The learned descriptors (AlexNet, NetVLAD, PatchNetVLAD,
> CosPlace, EigenPlaces) are pinned to CUDA/MPS at build time — upstream they
> silently fall back to CPU when no GPU is visible, which runs 10–100× slower
> with no warning. If you launch the container without GPU passthrough they now
> abort with a message telling you which flags to add. The non-neural SAD
> baseline still runs on CPU. To intentionally run the models on CPU anyway, set
> `VPR_ALLOW_CPU=1` in the container environment.

### Docker (NVIDIA Container Toolkit)

```bash
xhost +local:docker
docker run -it --rm \
    --gpus all \
    --shm-size=2g \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    slam:vpr
```

### Podman (e.g. RTX 5090 / sm_120 host)

```bash
xhost +local:
podman run -it --rm \
    --runtime=/usr/bin/nvidia-container-runtime \
    --security-opt=label=disable \
    --shm-size=2g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam:vpr
```

`--shm-size=2g` is required for the torch `DataLoader` shared memory.

### First run needs network access

Neither the dataset nor the model weights are baked into the image. On first
invocation `datasets/load_dataset.py` auto-downloads the dataset (e.g.
`GardensPoint_Walking.zip` from tu-chemnitz.de) into `images/`, and `torch.hub`
pulls each descriptor's weights (AlexNet ~233 MB; VGG16 ~528 MB for
NetVLAD/PatchNetVLAD; CosPlace/EigenPlaces ResNet50 checkpoints). These are
cached only inside the container, so a fresh `--rm` run re-downloads them unless
you persist them with mounts, e.g. `-v $PWD/images:/VPR_Tutorial/images` and
`-v vpr-hub:/root/.cache/torch/hub`.

### Inside the container

The working directory is `/VPR_Tutorial` and the venv is pre-activated.

**Single-descriptor demo:**

```bash
python3 demo.py --descriptor CosPlace --dataset GardensPoint
# descriptors: AlexNet, NetVLAD, PatchNetVLAD, CosPlace, EigenPlaces, SAD
# datasets:    GardensPoint, StLucia, SFU
```

Outputs four matplotlib windows: similarity matrix, correct/wrong match examples, matching decisions, and precision/recall curve.

**Multi-descriptor comparison:**

```bash
python3 /workspace/compare_descriptors.py
```

Runs all torch-based descriptors on GardensPoint and renders a combined window with overlaid PR curves, a metrics table (AUC, R@100P, R@K, wall time), similarity matrices side-by-side, and query→prediction example rows.

To save output to disk, mount a volume:

```bash
-v $PWD/visualizations:/out
# saves: visualizations/all_descriptors_comparison.png
```

**Headless (no X server):**

```bash
MPLBACKEND=Agg python3 demo.py --descriptor CosPlace --dataset GardensPoint
```

---

## References

- [VPR_Tutorial (stschubert)](https://github.com/stschubert/VPR_Tutorial)
- [PyTorch](https://pytorch.org/)
- [faiss](https://github.com/facebookresearch/faiss)
- [CosPlace](https://github.com/gmberton/CosPlace)
- [EigenPlaces](https://github.com/gmberton/EigenPlaces)
- [PatchNetVLAD](https://github.com/QVPR/Patch-NetVLAD)
