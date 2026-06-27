"""
Feature Projector P
Maps raw feature patches / concept vectors into a compact,
discriminative space where intra-concept features cluster
and inter-concept features separate.

Used in:
  - Stage 3 training: project v_k → v'_k before contrastive loss
  - Inference: project f(h,w) → f'(h,w) before cosine similarity with prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    """
    Two-layer MLP projector with L2 normalization at output.

    Architecture:
      Linear(in_dim, hidden_dim) → BN → ReLU → Linear(hidden_dim, out_dim) → L2-norm

    The output dimension matches the prototype embedding dimension.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features, any shape (..., in_dim)
               Works on both flat vectors and spatial patches.

        Returns:
            projected and L2-normalized features (..., out_dim)
        """
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1])    # flatten leading dims
        projected = self.net(x_flat)               # (N, out_dim)
        projected = F.normalize(projected, p=2, dim=-1)
        return projected.view(*original_shape[:-1], self.out_dim)

    def project_feature_map(self, f: torch.Tensor) -> torch.Tensor:
        """
        Project a spatial feature map patch-by-patch.

        Args:
            f: feature map (B, C, H, W)

        Returns:
            projected feature map (B, out_dim, H, W)
        """
        B, C, H, W = f.shape
        # Rearrange to (B*H*W, C) for batch projection
        f_patches = f.permute(0, 2, 3, 1).reshape(B * H * W, C)
        projected = self.forward(f_patches)                   # (B*H*W, out_dim)
        return projected.view(B, H, W, self.out_dim).permute(0, 3, 1, 2)  # (B, out_dim, H, W)
