# SuperPoint and SuperGlue using C++ and TensorRT

C++ inference demo for SuperPoint feature detection and SuperGlue feature matching,
accelerated with NVIDIA TensorRT.

---

## Project Structure

```
part2_ch01_04/
├── README.md
├── CMakeLists.txt
├── Dockerfile                        # TensorRT 8.4.1 (Pascal/Turing/Ampere/Ada GPUs)
├── Dockerfile.blackwell              # TensorRT 10.x (Blackwell GPUs)
├── config/
│   └── config.yaml                  # Model paths, image size, keypoint thresholds
├── include/
│   ├── super_point.h                # SuperPoint class declaration
│   ├── super_glue.h                 # SuperGlue class declaration
│   ├── read_config.h                # Configuration reader
│   └── utils.h                      # Utility functions
├── src/
│   ├── super_point.cpp              # SuperPoint TensorRT inference
│   └── super_glue.cpp               # SuperGlue TensorRT inference
├── examples/
│   ├── inference_image.cpp          # Match features on a single image pair
│   ├── inference_sequence.cpp       # Match consecutive frames in a monocular sequence
│   └── inference_stereo_sequence.cpp # Match cam0[i] <-> cam1[i] in a stereo sequence
├── convert2onnx/
│   ├── superpoint.py
│   ├── superglue.py
│   ├── convert_superpoint_to_onnx.py
│   └── convert_superglue_to_onnx.py
├── 3rdparty/
│   └── tensorrtbuffer/              # TensorRT buffer management utilities
└── weights/
    └── .gitkeep                     # Place ONNX/engine files here
```

---

## Build

### Dependencies (all required)

- **NVIDIA GPU** with CUDA support
- **CUDA 11.x+** (12.x for newer GPUs)
- **TensorRT 8.x** (Pascal/Turing/Ampere/Ada) **or TensorRT 10.x** (Blackwell)
- **OpenCV 4.2+**
- **Eigen3**
- **yaml-cpp**

### GPU / Dockerfile selection

| GPU generation | Examples | Dockerfile | TensorRT base |
|---|---|---|---|
| Pascal / Turing / Ampere / Ada | GTX 10xx, RTX 20xx, 30xx, 40xx | `Dockerfile` | `nvcr.io/nvidia/tensorrt:22.07-py3` (TRT 8.4.1) |
| Blackwell | RTX 50xx | `Dockerfile.blackwell` | `nvcr.io/nvidia/tensorrt:25.04-py3` (TRT 10.9) |

### Docker

```bash
# Pre-Blackwell GPUs (RTX 2080 Ti, RTX 30xx, RTX 40xx, …)
docker build . -t slam_zero_to_hero:part2_ch01_04

# Blackwell GPUs (RTX 5090, …)
docker build -f Dockerfile.blackwell . -t slam_zero_to_hero:part2_ch01_04_blackwell
```

### Local

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## Run

### 1. Prepare ONNX models

Convert PyTorch checkpoints to ONNX (requires the original SuperGlue weights):

```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
cd convert2onnx
python convert_superpoint_to_onnx.py
python convert_superglue_to_onnx.py
```

Place the resulting `.onnx` files in `weights/`. Paths are configured in `config/config.yaml`.

### 2. Start Docker container

```bash
xhost +local:root

# Pre-Blackwell GPUs
docker run -it --rm \
    --gpus all \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume $(pwd)/weights:/workspace/superpointglue/weights \
    --volume $(pwd)/data:/workspace/superpointglue/data \
    --volume $(pwd)/config:/workspace/superpointglue/config \
    --volume ~/data:/datasets \
    slam_zero_to_hero:part2_ch01_04 /bin/bash

# Blackwell GPUs: swap tag for :part2_ch01_04_blackwell (same flags)
```

Run from the `part2_ch01_04/` directory so bind mounts resolve correctly. The
`weights/` mount caches built `.engine` files on the host — only the first run
pays the ~10–20 min TensorRT engine build cost.

**Podman on Ubuntu 22.04** (no CDI in podman 3.x):

```bash
podman run -it --rm \
    --runtime=/usr/bin/nvidia-container-runtime \
    --security-opt=label=disable \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/weights:/workspace/superpointglue/weights \
    -v $(pwd)/data:/workspace/superpointglue/data \
    -v $(pwd)/config:/workspace/superpointglue/config \
    -v ~/data:/datasets \
    slam_zero_to_hero:part2_ch01_04_blackwell /bin/bash
```

### 3. Run inference

```bash
# Single image pair
./build/superpointglue_image config/config.yaml image1.png image2.png

# Monocular sequence (directory of images)
./build/superpointglue_sequence config/config.yaml /path/to/images/

# Stereo sequence (cam0 and cam1 directories, matched by index)
./build/superpointglue_stereo_sequence config/config.yaml /path/to/cam0/ /path/to/cam1/
```

### 4. EuRoC MAV dataset

EuRoC stores images under `mav0/cam0/data/` and `mav0/cam1/data/`. With the
`-v ~/data:/datasets` mount above:

```bash
# Monocular
./build/superpointglue_sequence \
    config/config.yaml \
    /datasets/euroc_mav/MH_01_easy/mav0/cam0/data/

# Stereo
./build/superpointglue_stereo_sequence \
    config/config.yaml \
    /datasets/euroc_mav/MH_01_easy/mav0/cam0/data/ \
    /datasets/euroc_mav/MH_01_easy/mav0/cam1/data/
```

Swap `MH_01_easy` for any other sequence (`MH_02_easy`, `V1_01_easy`, etc.).
Images are resized to `image_width`/`image_height` from `config/config.yaml` (default 640×480).

---

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) — official PyTorch weights
- [SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT) — original TensorRT implementation
