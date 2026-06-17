# Deep Learning-based Visual Place Recognition

Python exercise using the [VPR_Tutorial](https://github.com/stschubert/VPR_Tutorial) framework with switchable descriptors and datasets.

---

## Project Structure

```
part2_ch01_10/
├── README.md
├── Dockerfile                  # uv-based Python 3.11 + CUDA torch image
├── Dockerfile_pip              # pip-based alternative
├── compare_descriptors.py      # Multi-descriptor comparison script
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
