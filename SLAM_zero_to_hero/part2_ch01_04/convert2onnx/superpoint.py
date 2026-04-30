"""
SuperPoint PyTorch Model Definition

This is a simplified version of the SuperPoint architecture for ONNX export.
The full model includes the encoder and two heads (detection + description).

Reference:
    DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection
    and Description", CVPR 2018

Network Architecture:
    Input: Grayscale image (1 x 1 x H x W)

    Encoder (VGG-style):
        Conv(1->64) -> ReLU -> Conv(64->64) -> ReLU -> MaxPool
        Conv(64->64) -> ReLU -> Conv(64->64) -> ReLU -> MaxPool
        Conv(64->128) -> ReLU -> Conv(128->128) -> ReLU -> MaxPool
        Conv(128->128) -> ReLU -> Conv(128->128) -> ReLU

    Detector Head:
        Conv(128->256) -> ReLU -> Conv(256->65)
        Output: 65 channels (64 for 8x8 cell + 1 dustbin), then softmax + reshape
        Final output: Score map (1 x H x W)

    Descriptor Head:
        Conv(128->256) -> ReLU -> Conv(256->256)
        Output: 256-D descriptor at 1/8 resolution
        Final output: Descriptors (1 x 256 x H/8 x W/8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPointEncoder(nn.Module):
    """VGG-style encoder backbone."""

    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Block 2
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Block 3
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Block 4
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)

        # Block 2
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)

        # Block 3
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)

        # Block 4
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        return x


class SuperPointDetector(nn.Module):
    """Keypoint detection head."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # 65 = 64 (8x8 cell positions) + 1 (dustbin/no keypoint)
        self.conv_out = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.conv_out(x)

        # Reshape and softmax
        # x: (B, 65, H/8, W/8) -> (B, 65, H/8 * W/8) -> softmax -> (B, 64, H/8, W/8)
        B, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, 65)
        x = F.softmax(x, dim=-1)
        x = x[:, :, :-1]  # Remove dustbin
        x = x.reshape(B, H, W, 64).permute(0, 3, 1, 2)

        # Reshape to full resolution: (B, 64, H/8, W/8) -> (B, 1, H, W)
        x = F.pixel_shuffle(x, 8)
        x = x.squeeze(1)  # (B, H, W)

        return x


class SuperPointDescriptor(nn.Module):
    """Descriptor extraction head."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.conv_out(x)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)

        return x


class SuperPoint(nn.Module):
    """
    Complete SuperPoint model.

    Input:
        Grayscale image tensor of shape (B, 1, H, W), values in [0, 1]
        H and W should be divisible by 8

    Output:
        scores: Keypoint score map (B, H, W)
        descriptors: Dense descriptor map (B, 256, H/8, W/8)
    """

    def __init__(self):
        super().__init__()
        self.encoder = SuperPointEncoder()
        self.detector = SuperPointDetector()
        self.descriptor = SuperPointDescriptor()

    def forward(self, x):
        """
        Args:
            x: Input image (B, 1, H, W), normalized to [0, 1]

        Returns:
            scores: Keypoint scores (B, H, W)
            descriptors: Descriptor vectors (B, 256, H/8, W/8)
        """
        # Shared encoder
        features = self.encoder(x)

        # Detection head
        scores = self.detector(features)

        # Description head
        descriptors = self.descriptor(features)

        return scores, descriptors


def load_pretrained_weights(model, weights_path):
    """
    Load pretrained SuperPoint weights.

    The original SuperPoint weights are available from:
    https://github.com/magicleap/SuperGluePretrainedNetwork

    Note: Weight loading may require renaming keys depending on the source.
    """
    state_dict = torch.load(weights_path, map_location='cpu')

    # Handle different weight formats
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # Load with strict=False to handle missing/extra keys
    model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':
    # Test model
    model = SuperPoint()
    model.eval()

    # Create dummy input
    x = torch.randn(1, 1, 480, 640)

    # Forward pass
    with torch.no_grad():
        scores, descriptors = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Descriptors shape: {descriptors.shape}")
