"""
Improvement B: Uncertainty Head
Estimates per-patch, per-concept uncertainty as the variance of cosine
similarities across all M prototypes of each concept.

Key intuition:
  - If all M prototypes agree on a patch → low variance → model is CONFIDENT
  - If prototypes disagree → high variance → model is UNCERTAIN

Uncertainty maps U_k(h,w) are used to:
  1. Show doctors WHERE the model is uncertain (alongside similarity maps)
  2. Trigger a safety check before applying doctor-drawn bounding boxes

No new training parameters are needed beyond the existing prototypes.
Optionally, a lightweight calibration loss (ECE) can be added in Stage 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    """
    Computes uncertainty maps from the variance of prototype similarities.

    This module is STATELESS — it uses the existing prototypes from
    PrototypeLearner and requires no additional learned parameters
    (unless calibration is enabled).
    """

    def __init__(self, use_calibration: bool = False, num_concepts: int = None):
        """
        Args:
            use_calibration: if True, add a per-concept temperature parameter
                             to calibrate the uncertainty scale
            num_concepts:    required if use_calibration=True
        """
        super().__init__()
        self.use_calibration = use_calibration

        if use_calibration:
            assert num_concepts is not None
            # Learnable temperature per concept (initialized to 1.0 = no effect)
            self.temperature = nn.Parameter(torch.ones(num_concepts))

    def compute_uncertainty_maps(
        self,
        f_prime: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-patch mean similarity and variance (uncertainty) across
        all M prototypes for each concept.

        μ_k(h,w)  = (1/M) Σ_m cos(p_km, f'(h,w))
        U_k(h,w)  = (1/M) Σ_m (cos(p_km, f'(h,w)) - μ_k(h,w))²

        Args:
            f_prime:    projected feature map (B, D, H, W)
            prototypes: L2-normalized prototypes (K, M, D)

        Returns:
            mean_maps: mean similarity maps (B, K, H, W)
            var_maps:  variance (uncertainty) maps (B, K, H, W)
                       high values = uncertain, low values = confident
        """
        B, D, H, W = f_prime.shape
        K, M, _ = prototypes.shape

        # Flatten spatial: (B, H*W, D)
        patches = f_prime.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # All prototype similarities: (B, H*W, K*M)
        protos_flat = prototypes.view(K * M, D)
        sim_all = torch.matmul(patches, protos_flat.T)        # (B, H*W, K*M)

        # Reshape to (B, H*W, K, M)
        sim_per_proto = sim_all.view(B, H * W, K, M)

        # Mean and variance over M prototypes for each concept
        mean_sim = sim_per_proto.mean(dim=-1)                 # (B, H*W, K)
        var_sim = sim_per_proto.var(dim=-1, unbiased=False)   # (B, H*W, K)

        # Reshape back to spatial maps
        mean_maps = mean_sim.view(B, H, W, K).permute(0, 3, 1, 2)  # (B, K, H, W)
        var_maps = var_sim.view(B, H, W, K).permute(0, 3, 1, 2)    # (B, K, H, W)

        # Optional temperature scaling for calibration
        if self.use_calibration:
            temp = self.temperature.view(1, K, 1, 1).abs().clamp(min=1e-3)
            var_maps = var_maps / temp

        return mean_maps, var_maps

    def get_uncertainty_score(
        self,
        f_prime: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a single uncertainty score per concept per image
        (max uncertainty over all spatial positions).

        Used for ranking concepts by uncertainty in the UI.

        Returns:
            uncertainty_scores: (B, K) — higher = more uncertain
        """
        _, var_maps = self.compute_uncertainty_maps(f_prime, prototypes)
        return var_maps.amax(dim=(-2, -1))                   # (B, K)

    def safety_check(
        self,
        box_region: torch.Tensor,
        similarity_maps: torch.Tensor,
        intended_concept: int,
        uncertainty_threshold: float = 0.05,
    ) -> dict:
        """
        Check if a doctor-drawn bounding box region is dominated by a
        concept OTHER than the intended one.

        Args:
            box_region:       boolean mask (H, W) — True inside the drawn box
            similarity_maps:  (K, H, W) — per-concept similarity maps for one image
            intended_concept: index of the concept the doctor is trying to target
            uncertainty_threshold: variance above which we flag uncertainty

        Returns:
            dict with keys:
              'safe':            bool — True if no conflict detected
              'dominant_concept': int — concept with highest mean sim in box
              'dominant_score':  float
              'intended_score':  float
              'warning_message': str (empty if safe)
        """
        K, H, W = similarity_maps.shape
        box_mask = box_region.bool()                         # (H, W)

        # Mean similarity per concept inside the box
        box_sims = similarity_maps[:, box_mask]              # (K, N_pixels)
        mean_box_sims = box_sims.mean(dim=-1)                # (K,)

        dominant_concept = mean_box_sims.argmax().item()
        dominant_score = mean_box_sims[dominant_concept].item()
        intended_score = mean_box_sims[intended_concept].item()

        safe = (dominant_concept == intended_concept)
        warning = ""

        if not safe and (dominant_score - intended_score) > uncertainty_threshold:
            warning = (
                f"Warning: This region is primarily attributed to concept "
                f"#{dominant_concept} (score={dominant_score:.2f}), not concept "
                f"#{intended_concept} (score={intended_score:.2f}). "
                f"Consider also rejecting concept #{dominant_concept}."
            )

        return {
            "safe": safe,
            "dominant_concept": dominant_concept,
            "dominant_score": dominant_score,
            "intended_score": intended_score,
            "warning_message": warning,
        }
