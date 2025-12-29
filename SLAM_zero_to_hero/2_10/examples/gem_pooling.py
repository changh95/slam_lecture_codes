#!/usr/bin/env python3
"""
Generalized Mean (GeM) Pooling for Image Retrieval

This script demonstrates:
1. GeM pooling layer implementation
2. Comparison with Max and Average pooling
3. Feature extraction for image retrieval
4. Learnable pooling parameter

GeM pooling generalizes both max and average pooling through a
learnable parameter p:
- p -> inf: approaches max pooling
- p = 1: equals average pooling
- p > 1: emphasizes larger activations (typically p ≈ 3 works well)

Reference: Radenović et al., "Fine-tuning CNN Image Retrieval with
No Human Annotation", TPAMI 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from typing import Optional


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.

    GeM(x) = (1/|X| * sum_i x_i^p)^(1/p)

    Args:
        p: Initial pooling parameter (default: 3.0)
        eps: Small value for numerical stability
        learnable: Whether p is learnable
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6,
                 learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Pooled features [B, C]
        """
        # Clamp to avoid numerical issues
        x = x.clamp(min=self.eps)

        # Apply power, mean, then root
        # Use adaptive pooling for flexibility
        x_pow = x.pow(self.p)
        pooled = F.adaptive_avg_pool2d(x_pow, 1)
        return pooled.pow(1.0 / self.p).squeeze(-1).squeeze(-1)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p.item():.2f})'


class GeMResNet(nn.Module):
    """
    ResNet with GeM pooling for image retrieval.

    Architecture:
    1. ResNet backbone (conv layers)
    2. GeM pooling
    3. L2 normalization
    4. Optional whitening layer
    """

    def __init__(self, backbone: str = 'resnet50',
                 gem_p: float = 3.0,
                 output_dim: Optional[int] = None,
                 pretrained: bool = True):
        super().__init__()

        # Load backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove avgpool and fc layers
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # GeM pooling
        self.gem = GeM(p=gem_p, learnable=True)

        # Optional whitening/dimensionality reduction
        self.output_dim = output_dim or self.feature_dim
        if output_dim and output_dim != self.feature_dim:
            self.whiten = nn.Linear(self.feature_dim, output_dim)
        else:
            self.whiten = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global descriptor.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Global descriptor [B, output_dim]
        """
        # Extract features
        features = self.backbone(x)  # [B, C, H', W']

        # GeM pooling
        pooled = self.gem(features)  # [B, C]

        # Optional whitening
        if self.whiten is not None:
            pooled = self.whiten(pooled)

        # L2 normalize
        pooled = F.normalize(pooled, p=2, dim=1)

        return pooled

    def get_gem_p(self) -> float:
        """Get current GeM parameter value."""
        return self.gem.p.item()


def compare_pooling_methods(features: torch.Tensor) -> dict:
    """
    Compare different pooling methods.

    Args:
        features: Input features [B, C, H, W]

    Returns:
        Dictionary of pooled outputs
    """
    results = {}

    # Average pooling (GeM with p=1)
    avg_pool = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
    results['avg'] = avg_pool

    # Max pooling (approximation of GeM with p->inf)
    max_pool = F.adaptive_max_pool2d(features, 1).squeeze(-1).squeeze(-1)
    results['max'] = max_pool

    # GeM with different p values
    for p in [1.5, 2.0, 3.0, 5.0, 10.0]:
        gem = GeM(p=p, learnable=False)
        results[f'gem_p{p}'] = gem(features)

    return results


def demonstrate_gem_behavior():
    """Show how GeM behaves with different p values."""
    print("=== GeM Pooling Behavior ===\n")

    # Create synthetic feature map
    B, C, H, W = 1, 4, 4, 4

    # Feature map with one strong activation and weak background
    features = torch.ones(B, C, H, W) * 0.1
    features[0, 0, 1, 1] = 1.0  # Strong activation
    features[0, 1, 2, 2] = 0.8
    features[0, 2, 0, 3] = 0.6
    features[0, 3, 3, 0] = 0.4

    print("Feature map statistics:")
    print(f"  Shape: {features.shape}")
    print(f"  Min: {features.min().item():.2f}")
    print(f"  Max: {features.max().item():.2f}")
    print(f"  Mean: {features.mean().item():.2f}")

    print("\nPooling results (per channel):")

    # Compare pooling methods
    results = compare_pooling_methods(features)

    print(f"\n  {'Method':<12} {'Ch0':>8} {'Ch1':>8} {'Ch2':>8} {'Ch3':>8}")
    print("  " + "-" * 44)

    for name, pooled in results.items():
        values = [f"{pooled[0, i].item():.4f}" for i in range(C)]
        print(f"  {name:<12} {values[0]:>8} {values[1]:>8} "
              f"{values[2]:>8} {values[3]:>8}")

    print("\nObservations:")
    print("  - Average pooling: influenced by background")
    print("  - Max pooling: captures only the strongest activation")
    print("  - GeM (p=3): balanced emphasis on strong activations")
    print("  - Higher p -> closer to max pooling")


def main():
    print("=== GeM Pooling for Image Retrieval ===\n")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Demonstrate GeM behavior
    demonstrate_gem_behavior()

    # Create model
    print("\n=== Model Creation ===\n")

    model = GeMResNet(
        backbone='resnet50',
        gem_p=3.0,
        output_dim=2048,
        pretrained=True
    ).to(device)
    model.eval()

    print(f"Backbone: ResNet-50")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Output dimension: {model.output_dim}")
    print(f"Initial GeM p: {model.get_gem_p():.2f}")

    # Test inference
    print("\n=== Inference Test ===\n")

    dummy_input = torch.randn(1, 3, 480, 640).to(device)

    with torch.no_grad():
        descriptor = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {descriptor.shape}")
    print(f"Output norm: {descriptor.norm().item():.4f}")

    # Compare descriptors from different pooling
    print("\n=== Pooling Comparison on Real Features ===\n")

    with torch.no_grad():
        features = model.backbone(dummy_input)

    print(f"Feature map shape: {features.shape}")

    results = compare_pooling_methods(features)

    # Compute similarities between pooling methods
    print("\nCosine similarity between pooling methods:")

    methods = list(results.keys())
    for i, m1 in enumerate(methods[:4]):  # Compare first 4
        similarities = []
        for m2 in methods[:4]:
            sim = F.cosine_similarity(
                results[m1].flatten().unsqueeze(0),
                results[m2].flatten().unsqueeze(0)
            ).item()
            similarities.append(f"{sim:.3f}")
        print(f"  {m1:<12}: {' '.join(similarities)}")

    # GeM parameter learning simulation
    print("\n=== GeM Parameter Learning ===\n")

    # Create model with learnable p
    model_learnable = GeMResNet(
        backbone='resnet18',  # Smaller for speed
        gem_p=3.0,
        pretrained=True
    ).to(device)

    optimizer = torch.optim.Adam([model_learnable.gem.p], lr=0.1)

    print("Simulating gradient updates on GeM parameter...")
    print(f"Initial p: {model_learnable.get_gem_p():.4f}")

    # Simulate a few optimization steps
    for i in range(5):
        dummy_input = torch.randn(2, 3, 224, 224).to(device)

        # Forward pass
        desc = model_learnable(dummy_input)

        # Dummy loss (maximize descriptor norm)
        loss = -desc.norm(dim=1).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp p to reasonable range
        model_learnable.gem.p.data.clamp_(1.0, 10.0)

        print(f"  Step {i+1}: p = {model_learnable.get_gem_p():.4f}, "
              f"loss = {loss.item():.4f}")

    print("\n=== GeM Pooling Demo Complete ===")


if __name__ == '__main__':
    main()
