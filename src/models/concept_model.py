"""
Stage 1: Concept Model
Trains a backbone with per-concept CAM heads to generate
Class Activation Maps (cam_k) for each concept.

Output used in Stage 2 to extract local concept vectors v_k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConceptModel(nn.Module):
    """
    Backbone + per-concept 1x1 conv heads.

    Forward pass returns:
      - f:      feature map (B, C, H, W)
      - cam:    concept activation maps (B, K, H, W)
      - logits: concept classification logits (B, K)  -- used for BCE loss
    """

    def __init__(self, num_concepts: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.num_concepts = num_concepts

        # --- Backbone (feature extractor F) ---
        if backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            # Remove avgpool and fc — keep spatial feature maps
            self.feature_extractor = nn.Sequential(*list(base.children())[:-2])
            self.feature_dim = 2048
        elif backbone == "resnet34":
            base = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-2])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # --- Per-concept 1x1 conv CAM head ---
        # Produces K activation maps, one per concept
        self.cam_head = nn.Conv2d(self.feature_dim, num_concepts, kernel_size=1, bias=False)

        # Global average pool for classification logits
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input images (B, 3, H, W)

        Returns:
            f:      feature map (B, C, H, W)
            cam:    normalized activation maps per concept (B, K, H, W)
            logits: per-concept logits for BCE loss (B, K)
        """
        # Feature extraction
        f = self.feature_extractor(x)                        # (B, C, H, W)

        # Concept activation maps
        cam_raw = self.cam_head(f)                           # (B, K, H, W)

        # Normalize cam spatially to [0,1] for soft-selection
        B, K, H, W = cam_raw.shape
        cam_flat = cam_raw.view(B, K, -1)                    # (B, K, H*W)
        cam_min = cam_flat.min(dim=-1, keepdim=True).values
        cam_max = cam_flat.max(dim=-1, keepdim=True).values
        cam_norm = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam_norm.view(B, K, H, W)                      # (B, K, H, W)

        # GAP over cam for classification logits
        logits = self.gap(cam_raw).squeeze(-1).squeeze(-1)   # (B, K)

        return f, cam, logits

    def generate_concept_vectors(self, x: torch.Tensor):
        """
        Stage 2: Generate local concept vectors v_k for each concept.

        v_k = Σ_{h,w} softmax(cam_k(h,w)) * f(h,w)
            = soft-weighted sum of feature patches, weighted by concept activation

        Args:
            x: input images (B, 3, H, W)

        Returns:
            v: local concept vectors (B, K, C)
               v[b, k, :] is the concept-k vector for image b
        """
        with torch.no_grad():
            f, cam, _ = self.forward(x)                      # f: (B,C,H,W), cam: (B,K,H,W)

        B, C, H, W = f.shape
        K = cam.shape[1]

        # Spatial softmax over cam for each concept
        cam_flat = cam.view(B, K, H * W)                     # (B, K, H*W)
        weights = F.softmax(cam_flat, dim=-1)                # (B, K, H*W)

        # Reshape feature map for matmul
        f_flat = f.view(B, C, H * W)                         # (B, C, H*W)

        # Weighted sum: v_k = weights_k @ f^T  → (B, K, C)
        v = torch.bmm(weights, f_flat.permute(0, 2, 1))      # (B, K, C)

        # L2 normalize each concept vector
        v = F.normalize(v, p=2, dim=-1)

        return v
