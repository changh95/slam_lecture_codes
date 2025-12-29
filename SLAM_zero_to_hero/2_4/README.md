# SuperPoint and SuperGlue using C++ and TensorRT

This tutorial demonstrates how to deploy deep learning-based feature detection and matching (SuperPoint + SuperGlue) in C++ using NVIDIA TensorRT for high-performance inference. This is essential for deploying learned features in real-time SLAM systems.

## Overview

### Deep Learning-based Features in SLAM

Traditional feature detectors (SIFT, ORB, SURF) rely on hand-crafted algorithms to detect keypoints and compute descriptors. While effective, they have limitations in challenging conditions such as:
- Large viewpoint changes
- Significant illumination variations
- Repetitive textures
- Motion blur

**Deep learning-based features** learn to detect and describe keypoints from data, often outperforming traditional methods in challenging scenarios.

---

## SuperPoint: Self-Supervised Interest Point Detection and Description

### What is SuperPoint?

SuperPoint is a self-supervised deep learning model that jointly detects interest points and computes descriptors in a single forward pass.

**Architecture:**
```
Input Image (H x W x 1)
       |
   [Shared Encoder CNN]
       |
   +---+---+
   |       |
[Detector] [Descriptor]
   |           |
Keypoints   256-D Descriptors
(H/8 x W/8)  (H/8 x W/8 x 256)
```

**Key Features:**
- **Single-network architecture**: Encoder-decoder with two heads (detection + description)
- **Self-supervised training**: Uses homographic adaptation without manual labeling
- **Real-time performance**: ~15-30ms on modern GPUs
- **Dense descriptors**: Produces semi-dense descriptor maps, sampled at keypoint locations

**Detection Head:**
- Outputs a score map at 1/8 resolution
- Each cell predicts the probability of a keypoint
- Non-maximum suppression (NMS) extracts final keypoints

**Descriptor Head:**
- Outputs a 256-dimensional descriptor for each 8x8 patch
- Descriptors are L2-normalized
- Bilinear interpolation samples descriptors at keypoint locations

---

## SuperGlue: Learning Feature Matching with Graph Neural Networks

### What is SuperGlue?

SuperGlue is a learned feature matching network that uses graph neural networks (GNN) and attention mechanisms to match keypoints between image pairs.

**Architecture:**
```
Image 0 Features          Image 1 Features
(keypoints, descriptors)  (keypoints, descriptors)
        |                         |
    [Keypoint Encoder]      [Keypoint Encoder]
        |                         |
        +-----> [Attentional GNN] <-----+
               (self + cross attention)
                       |
              [Optimal Transport]
              (Sinkhorn Algorithm)
                       |
                Match Assignments
```

**Key Components:**

1. **Keypoint Encoder:**
   - Encodes keypoint position using sinusoidal positional encoding
   - Combines with descriptor to create initial node features

2. **Attentional Graph Neural Network:**
   - Self-attention: Aggregates information within each image
   - Cross-attention: Exchanges information between images
   - Multiple alternating layers of self and cross attention

3. **Optimal Transport Layer:**
   - Uses Sinkhorn algorithm for differentiable matching
   - Handles partial visibility (dustbin for unmatched points)
   - Outputs soft assignment matrix converted to hard matches

**Why SuperGlue is Better Than Traditional Matching:**
- Learns to reject outliers during matching (not post-hoc)
- Considers global context, not just local descriptor similarity
- Handles occlusions and partial visibility explicitly

---

## TensorRT Deployment Workflow

### Why TensorRT?

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime. It provides:
- **Layer fusion**: Combines multiple operations for fewer kernel launches
- **Precision calibration**: FP16/INT8 quantization with minimal accuracy loss
- **Kernel auto-tuning**: Selects optimal kernels for target GPU
- **Memory optimization**: Efficient memory allocation and reuse

### ONNX to TensorRT Pipeline

```
PyTorch Model (.pth)
       |
   [torch.onnx.export()]
       |
ONNX Model (.onnx)
       |
   [TensorRT Parser]
   [Optimization]
   [Kernel Selection]
       |
TensorRT Engine (.engine)
       |
   [Runtime Inference]
```

### Step 1: Export PyTorch to ONNX

The ONNX (Open Neural Network Exchange) format provides an intermediate representation that TensorRT can parse.

**SuperPoint ONNX Export (see convert2onnx/convert_superpoint_to_onnx.py):**
```python
import torch
import torch.onnx

# Load pretrained SuperPoint model
model = SuperPoint()
model.load_state_dict(torch.load('superpoint_v1.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 1, 480, 640)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "superpoint.onnx",
    input_names=['input'],
    output_names=['scores', 'descriptors'],
    dynamic_axes={
        'input': {2: 'height', 3: 'width'},
        'scores': {1: 'height', 2: 'width'},
        'descriptors': {2: 'height', 3: 'width'}
    },
    opset_version=11
)
```

### Step 2: Build TensorRT Engine

TensorRT parses the ONNX model and builds an optimized engine:

```cpp
// Create builder and network
auto builder = nvinfer1::createInferBuilder(logger);
auto network = builder->createNetworkV2(
    1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

// Parse ONNX file
auto parser = nvonnxparser::createParser(*network, logger);
parser->parseFromFile("superpoint.onnx", static_cast<int>(ILogger::Severity::kWARNING));

// Configure builder
auto config = builder->createBuilderConfig();
config->setMaxWorkspaceSize(512 * (1 << 20));  // 512 MB
config->setFlag(BuilderFlag::kFP16);  // Enable FP16

// Build engine (this takes time - serialize to file for reuse)
auto engine = builder->buildSerializedNetwork(*network, *config);
```

### Step 3: Runtime Inference

```cpp
// Deserialize engine
auto runtime = nvinfer1::createInferRuntime(logger);
auto engine = runtime->deserializeCudaEngine(engine_data, engine_size);
auto context = engine->createExecutionContext();

// Allocate buffers
void* buffers[3];  // input, scores, descriptors
cudaMalloc(&buffers[0], input_size);
cudaMalloc(&buffers[1], scores_size);
cudaMalloc(&buffers[2], descriptors_size);

// Copy input to GPU
cudaMemcpy(buffers[0], input_data, input_size, cudaMemcpyHostToDevice);

// Run inference
context->executeV2(buffers);

// Copy output to CPU
cudaMemcpy(output_scores, buffers[1], scores_size, cudaMemcpyDeviceToHost);
cudaMemcpy(output_descriptors, buffers[2], descriptors_size, cudaMemcpyDeviceToHost);
```

---

## Project Structure

```
2_4/
├── README.md                    # This file
├── CMakeLists.txt               # Build configuration
├── Dockerfile                   # Docker build environment
├── config/
│   └── config.yaml              # Model and inference configuration
├── include/
│   ├── super_point.h            # SuperPoint class declaration
│   ├── super_glue.h             # SuperGlue class declaration
│   ├── read_config.h            # Configuration reader
│   └── utils.h                  # Utility functions
├── src/
│   ├── super_point.cpp          # SuperPoint implementation
│   └── super_glue.cpp           # SuperGlue implementation
├── examples/
│   ├── inference_image.cpp      # Single image pair matching
│   └── inference_sequence.cpp   # Video/sequence matching
├── convert2onnx/
│   ├── superpoint.py            # SuperPoint PyTorch model
│   ├── superglue.py             # SuperGlue PyTorch model
│   ├── convert_superpoint_to_onnx.py
│   └── convert_superglue_to_onnx.py
├── 3rdparty/
│   └── tensorrtbuffer/          # TensorRT buffer management utilities
└── weights/
    └── .gitkeep                 # ONNX/engine files go here
```

---

## How to Build

### Prerequisites

- NVIDIA GPU with CUDA support
- TensorRT 8.x+
- CUDA 11.x+
- OpenCV 4.x
- Eigen3
- yaml-cpp

### Docker Build (Recommended)

```bash
# Build the Docker image
docker build . -t slam_zero_to_hero:2_4
```

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## How to Run

### 1. Start Docker Container

```bash
docker run -it --rm \
    --gpus all \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume ~/datasets:/datasets \
    slam_zero_to_hero:2_4 /bin/bash
```

### 2. Download/Convert Models

Before running inference, you need ONNX models. You can either:

**Option A: Use pre-converted models**
```bash
# Download pre-converted ONNX models (if available)
wget -O weights/superpoint_v1.onnx <model_url>
wget -O weights/superglue_indoor.onnx <model_url>
```

**Option B: Convert from PyTorch checkpoints**
```bash
# Clone original SuperGlue repository for weights
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git

# Run conversion scripts
cd convert2onnx
python convert_superpoint_to_onnx.py
python convert_superglue_to_onnx.py
```

### 3. Run Inference

```bash
# On image pair
./build/superpointglue_image config/config.yaml image1.png image2.png

# On image sequence
./build/superpointglue_sequence config/config.yaml /datasets/sequence/images/
```

**Note:** First run will take ~10-20 minutes to build TensorRT engines. Subsequent runs use cached engines.

---

## Integration with SLAM

### Replacing Traditional Features

To use SuperPoint+SuperGlue in a SLAM system:

```cpp
class LearnedFeatureFrontend {
public:
    LearnedFeatureFrontend(const std::string& config_path) {
        // Initialize SuperPoint and SuperGlue
        superpoint_ = std::make_shared<SuperPoint>(config.superpoint_config);
        superglue_ = std::make_shared<SuperGlue>(config.superglue_config);

        superpoint_->build();  // Load/build TensorRT engine
        superglue_->build();
    }

    void extractFeatures(const cv::Mat& image,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors) {
        Eigen::Matrix<double, 259, Eigen::Dynamic> features;
        superpoint_->infer(image, features);

        // Convert to OpenCV format for SLAM integration
        for (int i = 0; i < features.cols(); ++i) {
            keypoints.emplace_back(features(1, i), features(2, i), 8, -1, features(0, i));
        }
        // ... convert descriptors
    }

    void matchFeatures(const Frame& frame0, const Frame& frame1,
                      std::vector<cv::DMatch>& matches) {
        superglue_->matching_points(frame0.features, frame1.features, matches);
    }

private:
    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<SuperGlue> superglue_;
};
```

### Performance Considerations

| Component | Time (RTX 3080) | Notes |
|-----------|-----------------|-------|
| SuperPoint | ~5-8 ms | 640x480 image |
| SuperGlue | ~8-15 ms | 500 keypoints each |
| Total | ~15-25 ms | Real-time capable |

### Comparison with Traditional Features

| Aspect | ORB | SIFT | SuperPoint+SuperGlue |
|--------|-----|------|---------------------|
| Speed | ~10 ms | ~50 ms | ~20 ms |
| Repeatability | Good | Excellent | Excellent |
| Viewpoint invariance | Limited | Moderate | Excellent |
| Matching accuracy | ~60% | ~75% | ~90%+ |
| GPU required | No | No | Yes |

---

## Key Concepts

### Dynamic Input Shapes

Both SuperPoint and SuperGlue support dynamic input shapes (variable image sizes, variable number of keypoints). TensorRT handles this with optimization profiles:

```cpp
auto profile = builder->createOptimizationProfile();
profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 1, 100, 100));
profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(1, 1, 480, 640));
profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(1, 1, 1080, 1920));
config->addOptimizationProfile(profile);
```

### FP16 Precision

TensorRT can use FP16 precision for faster inference with minimal accuracy loss:

```cpp
config->setFlag(BuilderFlag::kFP16);
```

### Engine Serialization

TensorRT engines are GPU-specific. Serialize to avoid rebuilding:

```cpp
// Save engine
IHostMemory* serialized = engine->serialize();
std::ofstream file("superpoint.engine", std::ios::binary);
file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());

// Load engine
std::ifstream file("superpoint.engine", std::ios::binary);
// ... deserialize
```

---

## References

- [SuperPoint Paper](https://arxiv.org/abs/1712.07629) - DeTone et al., CVPR 2018
- [SuperGlue Paper](https://arxiv.org/abs/1911.11763) - Sarlin et al., CVPR 2020
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Original SuperGlue Repository](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT) - Original TensorRT implementation by yuefanhao

---

## Troubleshooting

### TensorRT Engine Build Fails
- Ensure CUDA and TensorRT versions are compatible
- Check GPU memory (engine building requires significant memory)
- Verify ONNX model is correctly exported

### Poor Matching Results
- Ensure images are grayscale
- Check image preprocessing (normalization to 0-1 range)
- Verify keypoint threshold in config

### Slow First Run
- First run builds TensorRT engines (~10-20 minutes)
- Subsequent runs load cached `.engine` files
- Delete `.engine` files to force rebuild if issues occur
