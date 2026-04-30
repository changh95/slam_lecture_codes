"""
SuperGlue PyTorch Model Definition (Simplified)

This is a simplified version of the SuperGlue architecture for understanding
the model structure. The full implementation includes more sophisticated
attention mechanisms.

Reference:
    Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural
    Networks", CVPR 2020

Network Architecture:
    Inputs:
        - keypoints_0, keypoints_1: (B, N, 2) normalized coordinates
        - scores_0, scores_1: (B, N) keypoint confidence scores
        - descriptors_0, descriptors_1: (B, 256, N) descriptor vectors

    Keypoint Encoder:
        - MLP to encode keypoint positions
        - Add positional encoding to descriptors

    Attentional Graph Neural Network:
        - Multiple layers of alternating self-attention and cross-attention
        - Self-attention: aggregate info within each image
        - Cross-attention: exchange info between images

    Optimal Transport:
        - Compute score matrix between all keypoint pairs
        - Apply Sinkhorn algorithm for differentiable matching
        - Output soft assignment matrix

    Output:
        - Assignment matrix (B, N0+1, N1+1) with dustbin for unmatched
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointEncoder(nn.Module):
    """
    Encode keypoint positions using MLP with sinusoidal positional encoding.

    The encoder combines:
    1. Keypoint score (confidence)
    2. Keypoint position (x, y) with positional encoding
    """

    def __init__(self, feature_dim: int = 256, layers: list = [32, 64, 128, 256]):
        super().__init__()

        # Input: (score, x, y) = 3 dimensions
        self.encoder = self._make_mlp(3, layers + [feature_dim])

    def _make_mlp(self, in_dim: int, layer_dims: list) -> nn.Sequential:
        layers = []
        for out_dim in layer_dims[:-1]:
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        layers.append(nn.Conv1d(in_dim, layer_dims[-1], kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, keypoints: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints: (B, N, 2) normalized keypoint coordinates
            scores: (B, N) keypoint confidence scores

        Returns:
            encoded: (B, feature_dim, N) encoded keypoint features
        """
        # Combine position and score: (B, N, 3)
        inputs = torch.cat([scores.unsqueeze(-1), keypoints], dim=-1)

        # Transpose for Conv1d: (B, 3, N)
        inputs = inputs.transpose(1, 2)

        return self.encoder(inputs)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, feature_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.query = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.key = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.value = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, C, N) query features
            source: (B, C, M) source features to attend to

        Returns:
            output: (B, C, N) attended features
        """
        B = query.size(0)

        # Project to Q, K, V
        Q = self.query(query).view(B, self.num_heads, self.head_dim, -1)
        K = self.key(source).view(B, self.num_heads, self.head_dim, -1)
        V = self.value(source).view(B, self.num_heads, self.head_dim, -1)

        # Attention: softmax(Q^T K / sqrt(d)) V
        scores = torch.einsum('bhdn,bhdm->bhnm', Q, K) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        output = torch.einsum('bhnm,bhdm->bhdn', attention, V)

        # Merge heads
        output = output.contiguous().view(B, -1, query.size(-1))
        return self.merge(output)


class AttentionalGNN(nn.Module):
    """
    Attentional Graph Neural Network.

    Alternates between self-attention (within image) and
    cross-attention (between images).
    """

    def __init__(self, feature_dim: int = 256, num_layers: int = 9, num_heads: int = 4):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternate: self, cross, self, cross, ...
            layer_type = 'self' if i % 2 == 0 else 'cross'
            self.layers.append(nn.ModuleDict({
                'attention': MultiHeadAttention(feature_dim, num_heads),
                'mlp': nn.Sequential(
                    nn.Conv1d(2 * feature_dim, 2 * feature_dim, kernel_size=1),
                    nn.BatchNorm1d(2 * feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=1),
                ),
                'type': layer_type,
            }))

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            desc0: (B, C, N) features for image 0
            desc1: (B, C, M) features for image 1

        Returns:
            desc0: (B, C, N) updated features for image 0
            desc1: (B, C, M) updated features for image 1
        """
        for layer in self.layers:
            if layer['type'] == 'self':
                # Self-attention
                delta0 = layer['attention'](desc0, desc0)
                delta1 = layer['attention'](desc1, desc1)
            else:
                # Cross-attention
                delta0 = layer['attention'](desc0, desc1)
                delta1 = layer['attention'](desc1, desc0)

            # Update with residual
            desc0 = desc0 + layer['mlp'](torch.cat([desc0, delta0], dim=1))
            desc1 = desc1 + layer['mlp'](torch.cat([desc1, delta1], dim=1))

        return desc0, desc1


class SuperGlue(nn.Module):
    """
    Complete SuperGlue model.

    Inputs:
        keypoints_0: (B, N, 2) normalized keypoint coordinates for image 0
        scores_0: (B, N) keypoint scores for image 0
        descriptors_0: (B, 256, N) descriptors for image 0
        keypoints_1: (B, M, 2) normalized keypoint coordinates for image 1
        scores_1: (B, M) keypoint scores for image 1
        descriptors_1: (B, 256, M) descriptors for image 1

    Output:
        scores: (B, N+1, M+1) assignment matrix (including dustbin)
    """

    def __init__(self, feature_dim: int = 256, num_layers: int = 9, sinkhorn_iters: int = 100):
        super().__init__()

        self.kp_encoder = KeypointEncoder(feature_dim)
        self.gnn = AttentionalGNN(feature_dim, num_layers)

        self.final_proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

        # Dustbin score (learned)
        self.dustbin = nn.Parameter(torch.tensor(1.0))

        self.sinkhorn_iters = sinkhorn_iters

    def forward(self,
                keypoints_0: torch.Tensor,
                scores_0: torch.Tensor,
                descriptors_0: torch.Tensor,
                keypoints_1: torch.Tensor,
                scores_1: torch.Tensor,
                descriptors_1: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            scores: (B, N+1, M+1) log assignment matrix
        """
        # Encode keypoints and add to descriptors
        kp_enc0 = self.kp_encoder(keypoints_0, scores_0)
        kp_enc1 = self.kp_encoder(keypoints_1, scores_1)

        desc0 = descriptors_0 + kp_enc0
        desc1 = descriptors_1 + kp_enc1

        # Attentional GNN
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final projection
        desc0 = self.final_proj(desc0)
        desc1 = self.final_proj(desc1)

        # Compute score matrix: (B, N, M)
        scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        scores = scores / (desc0.size(1) ** 0.5)

        # Add dustbin
        B, N, M = scores.shape
        dustbin = self.dustbin.expand(B, N, 1)
        scores = torch.cat([scores, dustbin], dim=2)
        dustbin = self.dustbin.expand(B, 1, M + 1)
        scores = torch.cat([scores, dustbin], dim=1)

        # Log-domain Sinkhorn (optimal transport)
        scores = self._log_sinkhorn(scores)

        return scores

    def _log_sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Log-domain Sinkhorn algorithm for optimal transport.

        Args:
            scores: (B, N+1, M+1) raw scores

        Returns:
            log_assignment: (B, N+1, M+1) log assignment probabilities
        """
        B, N, M = scores.shape

        # Initialize marginals
        log_mu = torch.zeros(B, N, device=scores.device)
        log_nu = torch.zeros(B, M, device=scores.device)

        # Set last row/column marginals for dustbin
        log_mu[:, -1] = torch.log(torch.tensor(M - 1.0))
        log_nu[:, -1] = torch.log(torch.tensor(N - 1.0))

        # Sinkhorn iterations
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)

        for _ in range(self.sinkhorn_iters):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)

        return scores + u.unsqueeze(2) + v.unsqueeze(1)


if __name__ == '__main__':
    # Test model
    model = SuperGlue()
    model.eval()

    # Create dummy inputs
    B, N, M = 1, 100, 120
    keypoints_0 = torch.randn(B, N, 2)
    scores_0 = torch.rand(B, N)
    descriptors_0 = F.normalize(torch.randn(B, 256, N), dim=1)
    keypoints_1 = torch.randn(B, M, 2)
    scores_1 = torch.rand(B, M)
    descriptors_1 = F.normalize(torch.randn(B, 256, M), dim=1)

    # Forward pass
    with torch.no_grad():
        assignment = model(keypoints_0, scores_0, descriptors_0,
                          keypoints_1, scores_1, descriptors_1)

    print(f"Keypoints 0 shape: {keypoints_0.shape}")
    print(f"Keypoints 1 shape: {keypoints_1.shape}")
    print(f"Assignment matrix shape: {assignment.shape}")
    print(f"Expected shape: ({B}, {N+1}, {M+1})")
