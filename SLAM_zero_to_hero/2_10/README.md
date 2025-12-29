# Learning-based Global Feature Extraction using PyTorch and TensorFlow

This tutorial covers deep learning-based global feature extraction for place recognition and image retrieval in Visual SLAM. We demonstrate how to use learned global descriptors like NetVLAD, GeM, and CosPlace.

---

## Overview

Traditional global features (like Bag of Visual Words) rely on hand-crafted local descriptors. **Learning-based global features** use deep neural networks to directly learn compact, discriminative image representations for place recognition.

### Why Deep Global Features?

| Aspect | Traditional (BoVW) | Learning-based |
|--------|-------------------|----------------|
| Feature design | Hand-crafted | Learned from data |
| Invariance | Limited | Viewpoint, lighting, seasonal changes |
| Descriptor size | Variable | Fixed (typically 256-4096 dims) |
| Computation | CPU-friendly | GPU-accelerated |

---

## Key Architectures

### 1. NetVLAD (2016)

NetVLAD is a differentiable version of VLAD (Vector of Locally Aggregated Descriptors) that can be trained end-to-end.

**Architecture:**
```
Input Image
     |
  [CNN Backbone (VGG-16)]
     |
  [Feature Maps (H x W x D)]
     |
  [NetVLAD Layer]
  - Soft assignment to K clusters
  - Aggregation of residuals
     |
  [Global Descriptor (K x D)]
     |
  [L2 Normalization]
     |
  [PCA Whitening (optional)]
     |
  Output: 4096-D or 256-D vector
```

**Key Innovation:**
- Soft cluster assignment: `a_k(x) = softmax(-alpha * ||x - c_k||^2)`
- Differentiable aggregation: `V_k = sum_i a_k(x_i) * (x_i - c_k)`

### 2. GeM Pooling (Generalized Mean)

GeM is a simple but effective pooling layer that generalizes average and max pooling:

```
GeM(X) = (1/|X| * sum_i x_i^p)^(1/p)

p = 1: Average pooling
p -> inf: Max pooling
p = 3: Good balance (learnable)
```

**Architecture:**
```
Input Image -> CNN Backbone -> Feature Maps -> GeM Pooling -> FC Layer -> L2 Norm
```

### 3. CosPlace (2022)

A recent approach optimized for visual place recognition:

**Key Features:**
- Group-based classification training
- Optimized for geographic localization
- Efficient inference

---

## Project Structure

```
2_10/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── models/
│   └── .gitkeep                    # Pre-trained model weights
├── examples/
│   ├── netvlad_inference.py        # NetVLAD feature extraction
│   ├── gem_inference.py            # GeM pooling inference
│   ├── place_recognition.py        # End-to-end place recognition
│   └── export_to_onnx.py           # Export models for C++ deployment
└── cpp/
    ├── inference_onnx.cpp          # ONNX Runtime inference in C++
    └── CMakeLists.txt
```

---

## How to Build

### Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision
pip install tensorflow  # Optional, for TF models
pip install opencv-python numpy
pip install faiss-gpu   # For efficient similarity search
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:2_10
```

### C++ Build (ONNX Runtime)

```bash
mkdir build && cd build
cmake ..
make -j4
```

---

## How to Run

### Python Examples

```bash
# NetVLAD feature extraction
python examples/netvlad_inference.py --image test.jpg --output features.npy

# GeM pooling inference
python examples/gem_inference.py --image test.jpg

# Place recognition demo
python examples/place_recognition.py \
    --database /path/to/database/images \
    --query /path/to/query/images \
    --output results.txt

# Export to ONNX
python examples/export_to_onnx.py --model netvlad --output netvlad.onnx
```

### Docker

```bash
docker run -it --rm --gpus all \
    -v /path/to/data:/data \
    slam_zero_to_hero:2_10
```

---

## Code Examples

### 1. NetVLAD Feature Extraction (PyTorch)

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=512, alpha=100.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha

        # Cluster centers
        self.centroids = nn.Parameter(torch.randn(num_clusters, dim))
        # Soft assignment weights
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True)

    def forward(self, x):
        N, C, H, W = x.shape

        # Soft assignment
        soft_assign = self.conv(x)  # N x K x H x W
        soft_assign = F.softmax(soft_assign * self.alpha, dim=1)

        # Flatten spatial dimensions
        x_flatten = x.view(N, C, -1)  # N x C x (H*W)
        soft_assign = soft_assign.view(N, self.num_clusters, -1)  # N x K x (H*W)

        # VLAD aggregation
        residual = x_flatten.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(-1)
        vlad = (soft_assign.unsqueeze(2) * residual).sum(dim=-1)  # N x K x C

        # Normalize
        vlad = F.normalize(vlad, p=2, dim=2)  # Intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class NetVLADModel(nn.Module):
    def __init__(self, num_clusters=64):
        super().__init__()
        # VGG-16 backbone (up to conv5)
        vgg = models.vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:-2])
        self.pool = NetVLAD(num_clusters=num_clusters, dim=512)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x

# Usage
model = NetVLADModel()
model.eval()

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('test.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    descriptor = model(image_tensor)  # [1, 32768] for K=64, D=512
```

### 2. GeM Pooling (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

class GeMModel(nn.Module):
    def __init__(self, backbone='resnet50', output_dim=2048):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gem = GeM()
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gem(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Usage
model = GeMModel(output_dim=512)
descriptor = model(image_tensor)  # [1, 512]
```

### 3. Place Recognition Pipeline

```python
import numpy as np
import faiss

class PlaceRecognition:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.database = None
        self.index = None

    def extract_features(self, images):
        """Extract global features from a batch of images."""
        features = []
        with torch.no_grad():
            for img in images:
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                feat = self.model(img_tensor).cpu().numpy()
                features.append(feat)
        return np.vstack(features)

    def build_database(self, database_images):
        """Build FAISS index from database images."""
        self.database = self.extract_features(database_images)

        # Build FAISS index for fast similarity search
        dim = self.database.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.database.astype('float32'))

    def query(self, query_image, k=5):
        """Find k most similar images in database."""
        query_feat = self.extract_features([query_image])
        distances, indices = self.index.search(query_feat.astype('float32'), k)
        return indices[0], distances[0]

# Usage
pr = PlaceRecognition(model)
pr.build_database(database_images)
matches, scores = pr.query(query_image, k=5)
```

### 4. TensorFlow Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

class NetVLADLayer(layers.Layer):
    def __init__(self, num_clusters=64, **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters

    def build(self, input_shape):
        self.D = input_shape[-1]
        self.centroids = self.add_weight(
            shape=(self.num_clusters, self.D),
            initializer='glorot_uniform',
            trainable=True,
            name='centroids'
        )
        self.conv = layers.Conv2D(self.num_clusters, 1, use_bias=True)

    def call(self, x):
        N = tf.shape(x)[0]
        H, W = x.shape[1], x.shape[2]

        # Soft assignment
        soft_assign = self.conv(x)
        soft_assign = tf.nn.softmax(soft_assign * 100.0, axis=-1)
        soft_assign = tf.reshape(soft_assign, [N, -1, self.num_clusters])

        # VLAD
        x_flat = tf.reshape(x, [N, -1, self.D])
        residual = tf.expand_dims(x_flat, 2) - self.centroids
        weighted = tf.expand_dims(soft_assign, -1) * residual
        vlad = tf.reduce_sum(weighted, axis=1)

        # Normalize
        vlad = tf.nn.l2_normalize(vlad, axis=-1)
        vlad = tf.reshape(vlad, [N, -1])
        vlad = tf.nn.l2_normalize(vlad, axis=-1)

        return vlad

def create_netvlad_model(num_clusters=64):
    base = VGG16(weights='imagenet', include_top=False)
    x = base.output
    x = NetVLADLayer(num_clusters)(x)
    return Model(inputs=base.input, outputs=x)
```

---

## ONNX Export for C++ Deployment

```python
import torch

def export_to_onnx(model, output_path, input_size=(1, 3, 480, 640)):
    model.eval()
    dummy_input = torch.randn(*input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['descriptor'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'descriptor': {0: 'batch'}
        },
        opset_version=11
    )
    print(f"Model exported to {output_path}")

# Export
export_to_onnx(model, "netvlad.onnx")
```

---

## C++ Inference with ONNX Runtime

```cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class GlobalFeatureExtractor {
public:
    GlobalFeatureExtractor(const std::string& model_path) {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "GlobalFeatures");
        session_options_.SetIntraOpNumThreads(4);
        session_ = Ort::Session(env_, model_path.c_str(), session_options_);
    }

    std::vector<float> extract(const cv::Mat& image) {
        // Preprocess
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1.0/255.0,
                               cv::Size(640, 480),
                               cv::Scalar(0.485, 0.456, 0.406), true);

        // Create input tensor
        std::vector<int64_t> input_shape = {1, 3, 480, 640};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, (float*)blob.data, blob.total(),
            input_shape.data(), input_shape.size());

        // Run inference
        const char* input_names[] = {"input"};
        const char* output_names[] = {"descriptor"};
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        // Get output
        float* output_data = outputs[0].GetTensorMutableData<float>();
        size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

        return std::vector<float>(output_data, output_data + output_size);
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
};
```

---

## Pre-trained Models

| Model | Training Data | Dim | Download |
|-------|--------------|-----|----------|
| NetVLAD (VGG16) | Pittsburgh | 4096 | [Link](https://github.com/Relja/netvlad) |
| GeM (ResNet-101) | Google Landmarks | 2048 | [Link](https://github.com/filipradenovic/cnnimageretrieval-pytorch) |
| CosPlace | SF-XL | 512 | [Link](https://github.com/gmberton/CosPlace) |

---

## Performance Comparison

| Method | Recall@1 | Recall@5 | Inference Time |
|--------|----------|----------|----------------|
| BoVW (DBoW2) | 65% | 78% | 5 ms (CPU) |
| NetVLAD | 85% | 93% | 15 ms (GPU) |
| GeM | 82% | 91% | 8 ms (GPU) |
| CosPlace | 88% | 95% | 10 ms (GPU) |

*Benchmarks on Pittsburgh250k dataset*

---

## Best Practices

1. **Model Selection**: Use CosPlace or NetVLAD for best accuracy
2. **PCA Whitening**: Reduces dimensionality and improves performance
3. **Multi-scale Features**: Extract at multiple resolutions for robustness
4. **GPU Batching**: Process multiple images together for efficiency
5. **FAISS Indexing**: Use for efficient similarity search on large databases

---

## SLAM Integration

```cpp
class DeepPlaceRecognition {
public:
    bool detectLoopClosure(const cv::Mat& current_frame) {
        // Extract global feature
        auto descriptor = extractor_.extract(current_frame);

        // Query database
        auto [indices, distances] = database_.search(descriptor, 5);

        // Threshold check
        if (distances[0] < threshold_ &&
            abs(indices[0] - current_id_) > temporal_gap_) {
            // Geometric verification
            return verifyGeometrically(current_frame, indices[0]);
        }
        return false;
    }
};
```

---

## References

- [NetVLAD Paper](https://arxiv.org/abs/1511.07247): "NetVLAD: CNN architecture for weakly supervised place recognition"
- [GeM Paper](https://arxiv.org/abs/1711.02512): "Fine-tuning CNN Image Retrieval with No Human Annotation"
- [CosPlace Paper](https://arxiv.org/abs/2204.02287): "Rethinking Visual Geo-localization for Large-Scale Applications"
- [FAISS](https://github.com/facebookresearch/faiss): Facebook AI Similarity Search
