#!/usr/bin/env python3
"""
NetVLAD Implementation for Visual Place Recognition

This script demonstrates:
1. NetVLAD layer implementation (soft assignment to visual words)
2. Feature extraction with pre-trained backbone (VGG-16)
3. Place recognition using global descriptors
4. Similarity search for image retrieval

NetVLAD is a CNN architecture for place recognition that aggregates
local features into a global descriptor using a differentiable VLAD layer.

Reference: Arandjelovic et al., "NetVLAD: CNN architecture for weakly
supervised place recognition", CVPR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Optional


class NetVLADLayer(nn.Module):
    """
    NetVLAD layer that aggregates local features into a global descriptor.

    Instead of hard assignment to clusters (like traditional VLAD),
    NetVLAD uses soft assignment, making it differentiable and trainable.

    Args:
        num_clusters: Number of visual words (K)
        dim: Dimension of local features (D)
        alpha: Softmax temperature parameter
        normalize_input: Whether to L2-normalize input features
    """

    def __init__(self, num_clusters: int = 64, dim: int = 512,
                 alpha: float = 100.0, normalize_input: bool = True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input

        # Cluster centers (visual words)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

        # Soft assignment weights
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True)

        # Initialize conv weights from centroids
        self._init_params()

    def _init_params(self):
        """Initialize parameters for soft assignment."""
        # Initialize conv weights to produce soft assignment similar to
        # distance-based assignment to centroids
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            -self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: aggregate local features into VLAD descriptor.

        Args:
            x: Input features [B, D, H, W]

        Returns:
            VLAD descriptor [B, K*D]
        """
        B, D, H, W = x.shape
        N = H * W  # Number of local features

        # L2 normalize input features
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # Soft assignment: compute assignment weights
        # soft_assign: [B, K, H, W]
        soft_assign = self.conv(x)
        soft_assign = F.softmax(soft_assign, dim=1)

        # Reshape for VLAD computation
        # x_flatten: [B, D, N]
        x_flatten = x.view(B, D, -1)

        # soft_assign: [B, K, N]
        soft_assign = soft_assign.view(B, self.num_clusters, -1)

        # Compute VLAD: residuals weighted by soft assignment
        # For each cluster k:
        #   vlad_k = sum_i (a_ik * (x_i - c_k))
        # where a_ik is soft assignment weight

        # Expand dimensions for broadcasting
        # x_flatten: [B, 1, D, N]
        # centroids: [1, K, D, 1]
        x_expand = x_flatten.unsqueeze(1)
        centroids_expand = self.centroids.unsqueeze(0).unsqueeze(-1)

        # Residuals: [B, K, D, N]
        residuals = x_expand - centroids_expand

        # Weighted sum over spatial locations
        # soft_assign: [B, K, 1, N]
        soft_assign_expand = soft_assign.unsqueeze(2)

        # VLAD: [B, K, D]
        vlad = (residuals * soft_assign_expand).sum(dim=-1)

        # Intra-normalization (normalize each cluster's descriptor)
        vlad = F.normalize(vlad, p=2, dim=2)

        # Flatten to [B, K*D]
        vlad = vlad.view(B, -1)

        # L2 normalize final descriptor
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


class NetVLAD(nn.Module):
    """
    Complete NetVLAD model with VGG backbone and VLAD aggregation.

    Architecture:
    1. VGG-16 backbone (conv layers only)
    2. NetVLAD aggregation layer
    3. Whitening/PCA (optional)
    """

    def __init__(self, num_clusters: int = 64, output_dim: int = 4096,
                 pretrained_backbone: bool = True):
        super().__init__()

        # VGG-16 backbone (conv layers only, up to conv5)
        vgg = models.vgg16(pretrained=pretrained_backbone)
        layers = list(vgg.features.children())[:-2]  # Remove last pooling
        self.backbone = nn.Sequential(*layers)

        # Feature dimension from VGG conv5
        self.feature_dim = 512

        # NetVLAD layer
        self.vlad = NetVLADLayer(
            num_clusters=num_clusters,
            dim=self.feature_dim
        )

        # Output dimension
        vlad_dim = num_clusters * self.feature_dim
        self.output_dim = output_dim

        # Optional dimensionality reduction
        if output_dim != vlad_dim:
            self.fc = nn.Linear(vlad_dim, output_dim)
        else:
            self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global descriptor from image.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Global descriptor [B, output_dim]
        """
        # Extract local features
        features = self.backbone(x)  # [B, 512, H', W']

        # Aggregate with NetVLAD
        vlad = self.vlad(features)  # [B, K*512]

        # Optional dimensionality reduction
        if self.fc is not None:
            vlad = self.fc(vlad)

        # L2 normalize
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


def extract_descriptor(model: nn.Module, image_path: str,
                       transform: transforms.Compose,
                       device: torch.device) -> np.ndarray:
    """Extract global descriptor from an image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract descriptor
    with torch.no_grad():
        descriptor = model(image_tensor)

    return descriptor.cpu().numpy().flatten()


def compute_similarity(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """Compute cosine similarity between descriptors."""
    return np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))


def find_similar_images(query_desc: np.ndarray,
                        database_descs: List[np.ndarray],
                        top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Find most similar images in database.

    Returns:
        List of (index, similarity_score) tuples, sorted by similarity
    """
    similarities = []
    for i, db_desc in enumerate(database_descs):
        sim = compute_similarity(query_desc, db_desc)
        similarities.append((i, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def main():
    print("=== NetVLAD for Visual Place Recognition ===\n")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    print("\nCreating NetVLAD model...")
    model = NetVLAD(
        num_clusters=64,
        output_dim=4096,
        pretrained_backbone=True
    ).to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Test with random input
    print("\nTesting with random input...")
    dummy_input = torch.randn(1, 3, 480, 640).to(device)
    with torch.no_grad():
        descriptor = model(dummy_input)

    print(f"Input shape:      {dummy_input.shape}")
    print(f"Descriptor shape: {descriptor.shape}")
    print(f"Descriptor norm:  {descriptor.norm().item():.4f}")

    # Demonstrate place recognition workflow
    print("\n=== Place Recognition Workflow ===\n")

    # Simulate database of descriptors
    print("Building database of synthetic descriptors...")
    num_db_images = 100
    database_descs = []

    for i in range(num_db_images):
        # Generate random but structured descriptors
        # (in practice, these would come from real images)
        fake_input = torch.randn(1, 3, 480, 640).to(device)
        with torch.no_grad():
            desc = model(fake_input).cpu().numpy().flatten()
        database_descs.append(desc)

    print(f"Database size: {len(database_descs)} images")

    # Query with a "similar" descriptor
    print("\nQuerying database...")
    query_input = torch.randn(1, 3, 480, 640).to(device)
    with torch.no_grad():
        query_desc = model(query_input).cpu().numpy().flatten()

    # Find similar images
    results = find_similar_images(query_desc, database_descs, top_k=5)

    print("\nTop-5 similar images:")
    for rank, (idx, sim) in enumerate(results, 1):
        print(f"  {rank}. Image {idx}: similarity = {sim:.4f}")

    # NetVLAD internals demonstration
    print("\n=== NetVLAD Internals ===\n")

    # Show cluster assignments
    features = model.backbone(dummy_input)
    print(f"Local features shape: {features.shape}")
    print(f"  (B={features.shape[0]}, D={features.shape[1]}, "
          f"H={features.shape[2]}, W={features.shape[3]})")

    # Soft assignment visualization
    with torch.no_grad():
        soft_assign = model.vlad.conv(features)
        soft_assign = F.softmax(soft_assign, dim=1)

    print(f"\nSoft assignment shape: {soft_assign.shape}")
    print(f"  (B={soft_assign.shape[0]}, K={soft_assign.shape[1]}, "
          f"H={soft_assign.shape[2]}, W={soft_assign.shape[3]})")

    # Show assignment statistics
    assign_max = soft_assign.max(dim=1)[0]  # Max assignment per location
    assign_entropy = -(soft_assign * torch.log(soft_assign + 1e-10)).sum(dim=1)

    print(f"\nAssignment statistics:")
    print(f"  Max assignment (mean): {assign_max.mean().item():.4f}")
    print(f"  Entropy (mean):        {assign_entropy.mean().item():.4f}")

    print("\n=== NetVLAD Demo Complete ===")


if __name__ == '__main__':
    main()
