"""
All loss functions for CSR++ training.

Stage 1:  ConceptBCELoss       — multi-label concept classification
Stage 3:  PrototypeLoss        — multi-prototype contrastive loss + VLM alignment
Stage 4:  ClassificationLoss   — cross-entropy for final diagnosis

Combined: TotalLoss            — weighted sum for joint training (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptBCELoss(nn.Module):
    """
    Stage 1: Binary cross-entropy loss for multi-label concept prediction.
    Used to train the backbone + CAM head to activate on the correct concepts.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        """
        Args:
            pos_weight: class weights for imbalanced datasets (num_concepts,)
                        e.g. if concept k is rare, upweight its positive samples
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: concept logits (B, K)
            labels: binary concept labels (B, K)

        Returns:
            scalar loss
        """
        return self.bce(logits, labels.float())


class PrototypeLoss(nn.Module):
    """
    Stage 3: Combined prototype learning loss.

    L_total = L_con-m  +  λ_align * L_align

    L_con-m: multi-prototype contrastive loss (from PrototypeLearner)
    L_align: VLM alignment loss (from VLMAligner) — only if use_vlm=True
    """

    def __init__(self, lambda_align: float = 0.1, use_vlm: bool = True):
        super().__init__()
        self.lambda_align = lambda_align
        self.use_vlm = use_vlm

    def forward(
        self,
        contrastive_loss: torch.Tensor,
        alignment_loss: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            contrastive_loss: scalar from PrototypeLearner.contrastive_loss()
            alignment_loss:   scalar from VLMAligner.compute_alignment_loss()
                              (None if use_vlm=False)

        Returns:
            total prototype loss (scalar)
        """
        if self.use_vlm and alignment_loss is not None:
            return contrastive_loss + self.lambda_align * alignment_loss
        return contrastive_loss


class ClassificationLoss(nn.Module):
    """
    Stage 4: Cross-entropy loss for multi-class target prediction.
    Applied to GNN (or linear) task head output.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes)
            targets: (B,) long integer class indices

        Returns:
            scalar cross-entropy loss
        """
        return self.ce(logits, targets)


class CalibrationLoss(nn.Module):
    """
    Optional Stage 4 addition for Improvement B.
    Encourages the variance (uncertainty) estimates to be well-calibrated
    by penalizing cases where high confidence (low variance) is wrong.

    Approximation of Expected Calibration Error (ECE) as a differentiable loss.
    """

    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins

    def forward(
        self,
        confidences: torch.Tensor,
        correct: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            confidences: model confidence scores (B,) in [0,1]
                         computed as 1 - normalized_uncertainty
            correct:     binary correctness (B,) — 1 if prediction was correct

        Returns:
            ECE approximation (scalar)
        """
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=confidences.device)
        ece = torch.tensor(0.0, device=confidences.device)

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = correct[mask].float().mean()
            bin_weight = mask.float().mean()
            ece += bin_weight * torch.abs(bin_conf - bin_acc)

        return ece
