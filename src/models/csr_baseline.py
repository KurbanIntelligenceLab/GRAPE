"""
Full CSR++ Model Assembly
Assembles all components into a single forward-passable module.

Supports three modes:
  'baseline'  — original CSR (linear task head)
  'improved'  — CSR with GNN (A) + Uncertainty (B) + VLM alignment (C)
  'ablation'  — selectively enable/disable A, B, C via flags
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_model import ConceptModel
from .projector import FeatureProjector
from .prototype_learner import PrototypeLearner
from .gnn_task_head import GNNTaskHead
from .uncertainty_head import UncertaintyHead
from .vlm_alignment import VLMAligner


class CSRModel(nn.Module):
    """
    Full CSR++ model with configurable improvements.

    Inference flow:
      x → F → f → P → f' → cosine sim vs {p_km} → s → H → ŷ

    With improvements:
      [A] H is GNNTaskHead instead of linear
      [B] UncertaintyHead adds variance maps alongside similarity maps
      [C] VLMAligner adds alignment loss during Stage 3 training
    """

    def __init__(
        self,
        num_concepts: int,
        num_prototypes: int,
        num_classes: int,
        backbone: str = "resnet50",
        proto_dim: int = 256,
        # Improvement flags
        use_gnn: bool = True,          # [A]
        use_uncertainty: bool = True,  # [B]
        use_vlm: bool = True,          # [C]
        # GNN config
        gnn_hidden_dim: int = 64,
        gnn_num_heads: int = 4,
        # VLM config
        vlm_text_dim: int = 768,
        vlm_lambda: float = 0.1,
        vlm_model_name: str = "microsoft/BiomedVLP-BioViL-T",
        # Other
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.use_gnn = use_gnn
        self.use_uncertainty = use_uncertainty
        self.use_vlm = use_vlm

        # ── Stage 1: Concept model ──────────────────────────────────────────
        self.concept_model = ConceptModel(
            num_concepts=num_concepts,
            backbone=backbone,
            pretrained=pretrained_backbone,
        )
        feature_dim = self.concept_model.feature_dim

        # ── Stage 3: Projector ──────────────────────────────────────────────
        self.projector = FeatureProjector(
            in_dim=feature_dim,
            hidden_dim=feature_dim // 2,
            out_dim=proto_dim,
        )

        # ── Stage 3: Prototype learner ──────────────────────────────────────
        self.prototype_learner = PrototypeLearner(
            num_concepts=num_concepts,
            num_prototypes=num_prototypes,
            proto_dim=proto_dim,
        )

        # ── Stage 4: Task head ──────────────────────────────────────────────
        if use_gnn:
            self.task_head = GNNTaskHead(
                num_concepts=num_concepts,
                num_prototypes=num_prototypes,
                num_classes=num_classes,
                hidden_dim=gnn_hidden_dim,
                num_heads=gnn_num_heads,
            )
        else:
            # Original linear task head
            self.task_head = nn.Linear(num_concepts * num_prototypes, num_classes)

        # ── Improvement B: Uncertainty head ─────────────────────────────────
        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                use_calibration=False,
                num_concepts=num_concepts,
            )

        # ── Improvement C: VLM aligner ──────────────────────────────────────
        if use_vlm:
            self.vlm_aligner = VLMAligner(
                text_dim=vlm_text_dim,
                visual_dim=proto_dim,
                lambda_align=vlm_lambda,
                vlm_model_name=vlm_model_name,
            )

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
        return_uncertainty: bool = False,
    ) -> dict:
        """
        Full inference forward pass.

        Args:
            x:                  input images (B, 3, H, W)
            return_maps:        if True, also return similarity maps (for visualization)
            return_uncertainty: if True, also return uncertainty maps [B only if use_uncertainty]

        Returns:
            dict with keys:
              'logits':       (B, num_classes)
              'sim_scores':   (B, K, M)
              'sim_maps':     (B, K, M, H, W)  [if return_maps]
              'mean_maps':    (B, K, H, W)      [if return_uncertainty and use_uncertainty]
              'var_maps':     (B, K, H, W)      [if return_uncertainty and use_uncertainty]
        """
        # Feature extraction
        f, cam, concept_logits = self.concept_model(x)      # f: (B,C,H,W)

        # Project feature map
        f_prime = self.projector.project_feature_map(f)     # (B, proto_dim, H, W)

        # Similarity scores
        sim_scores = self.prototype_learner.get_similarity_scores(f_prime)  # (B, K, M)

        # Task head prediction
        if self.use_gnn:
            logits = self.task_head(sim_scores)              # (B, num_classes)
        else:
            B = sim_scores.shape[0]
            logits = self.task_head(sim_scores.view(B, -1))  # (B, num_classes)

        out = {
            "logits": logits,
            "sim_scores": sim_scores,
            "concept_logits": concept_logits,
        }

        if return_maps:
            out["sim_maps"] = self.prototype_learner.get_similarity_maps(f_prime)

        if return_uncertainty and self.use_uncertainty:
            protos = self.prototype_learner.normalized_prototypes
            mean_maps, var_maps = self.uncertainty_head.compute_uncertainty_maps(f_prime, protos)
            out["mean_maps"] = mean_maps
            out["var_maps"] = var_maps

        return out

    def set_concept_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """Pass the concept co-occurrence graph to the GNN task head."""
        assert self.use_gnn, "GNN task head is not enabled."
        self.task_head.set_graph(edge_index, edge_weight)

    def get_stage_parameters(self, stage: int) -> list:
        """
        Return parameter groups for each training stage.

        Stage 1: concept model only
        Stage 3: projector + prototype learner (+ VLM aligner projection W)
        Stage 4: task head only
        """
        if stage == 1:
            return list(self.concept_model.parameters())
        elif stage == 3:
            params = (
                list(self.projector.parameters())
                + list(self.prototype_learner.parameters())
            )
            if self.use_vlm:
                params += list(self.vlm_aligner.text_projection.parameters())
            return params
        elif stage == 4:
            return list(self.task_head.parameters())
        else:
            raise ValueError(f"Unknown stage: {stage}")
