"""
Train-time and test-time interaction logic for CSR++.

Train-time interaction (Section 3.2 of paper):
  - Doctor reviews the prototype atlas
  - Discards "shortcut" prototypes (e.g. pacemaker for cardiomegaly)
  - Refined atlas has fewer but more trustworthy prototypes

Test-time interaction (Section 3.3 of paper):
  - Concept-level: doctor rejects a concept (sets its similarity score to 0)
  - Spatial-level:  doctor draws positive/negative bounding boxes
  - [Improvement B] Safety check warns before amplifying potential errors
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """A spatial feedback region drawn by the doctor."""
    x1: int
    y1: int
    x2: int
    y2: int
    positive: bool   # True = "focus here", False = "ignore here"
    concept_idx: int = -1   # -1 means applies to all concepts


@dataclass
class DoctorFeedback:
    """All feedback provided by a doctor for one image."""
    rejected_concepts: list[int] = field(default_factory=list)
    bounding_boxes: list[BoundingBox] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Train-time interaction
# ──────────────────────────────────────────────────────────────────────────────

class AtlasRefiner:
    """
    Manages the train-time prototype atlas curation.

    Doctors inspect each prototype's associated image region and flag
    those that are "shortcuts" (non-pathological correlations).
    Flagged prototypes are masked out and excluded from inference.
    """

    def __init__(self, num_concepts: int, num_prototypes: int):
        self.num_concepts = num_concepts
        self.num_prototypes = num_prototypes
        # True = prototype is active, False = discarded by doctor
        self.active_mask = torch.ones(num_concepts, num_prototypes, dtype=torch.bool)

    def discard_prototype(self, concept_idx: int, proto_idx: int):
        """Mark a prototype as a shortcut — exclude from inference."""
        self.active_mask[concept_idx, proto_idx] = False

    def restore_prototype(self, concept_idx: int, proto_idx: int):
        """Restore a previously discarded prototype."""
        self.active_mask[concept_idx, proto_idx] = True

    def get_active_count(self) -> dict[int, int]:
        """Return number of active prototypes per concept."""
        return {k: self.active_mask[k].sum().item() for k in range(self.num_concepts)}

    def apply_mask_to_scores(self, sim_scores: torch.Tensor) -> torch.Tensor:
        """
        Zero out similarity scores for discarded prototypes.

        Args:
            sim_scores: (B, K, M) similarity scores

        Returns:
            masked similarity scores (B, K, M)
        """
        mask = self.active_mask.to(sim_scores.device)            # (K, M)
        return sim_scores * mask.unsqueeze(0).float()            # (B, K, M)

    def summary(self) -> str:
        lines = ["Atlas refinement summary:"]
        for k in range(self.num_concepts):
            active = self.active_mask[k].sum().item()
            total = self.num_prototypes
            discarded = total - active
            lines.append(f"  Concept {k:2d}: {active}/{total} active, {discarded} discarded")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Test-time interaction
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeInteraction:
    """
    Applies doctor feedback to recalibrate model predictions at test time.

    Two interaction types:
      1. Concept-level: reject a concept entirely (zero out its scores)
      2. Spatial-level: draw +/- boxes to create an importance map

    [Improvement B] Safety check: warn if a positive box targets a region
    already attributed to a different concept.
    """

    def __init__(self, alpha: float = 0.5, uncertainty_threshold: float = 0.05):
        """
        Args:
            alpha: weight for unspecified regions in importance map (0 < α < 1)
                   α=0.5 means unspecified regions contribute at half weight
            uncertainty_threshold: minimum score gap to trigger safety warning
        """
        self.alpha = alpha
        self.uncertainty_threshold = uncertainty_threshold

    def apply_concept_rejection(
        self,
        sim_scores: torch.Tensor,
        rejected_concepts: list[int],
    ) -> torch.Tensor:
        """
        Zero out similarity scores for rejected concepts.

        Args:
            sim_scores:        (B, K, M) or (K, M) for a single image
            rejected_concepts: list of concept indices to reject

        Returns:
            modified similarity scores with rejected concepts zeroed
        """
        scores = sim_scores.clone()
        for k in rejected_concepts:
            scores[..., k, :] = 0.0
        return scores

    def build_importance_map(
        self,
        bounding_boxes: list[BoundingBox],
        height: int,
        width: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Build spatial importance map A ∈ R^(H, W) from doctor's bounding boxes.

        A(h,w) = 1.0  if (h,w) ∈ positive boxes
               = 0.0  if (h,w) ∈ negative boxes
               = α    otherwise

        Args:
            bounding_boxes: list of BoundingBox annotations
            height, width:  spatial dimensions of the feature map

        Returns:
            importance map (H, W) with values in {0, α, 1}
        """
        A = torch.full((height, width), self.alpha, device=device)

        for bb in bounding_boxes:
            # Clamp coordinates to map dimensions
            x1 = max(0, min(bb.x1, width - 1))
            x2 = max(0, min(bb.x2, width))
            y1 = max(0, min(bb.y1, height - 1))
            y2 = max(0, min(bb.y2, height))
            val = 1.0 if bb.positive else 0.0
            A[y1:y2, x1:x2] = val

        return A

    def apply_spatial_interaction(
        self,
        sim_maps: torch.Tensor,
        bounding_boxes: list[BoundingBox],
        concept_idx: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply importance map to similarity maps, then recompute max scores.

        Ŝ_k(h,w) = A(h,w) ⊙ S_k(h,w)
        ŝ_k = max_{h,w} Ŝ_k(h,w)

        Args:
            sim_maps:     (K, M, H, W) similarity maps for one image
            bounding_boxes: doctor-drawn boxes
            concept_idx:  if specified, only apply to this concept's maps

        Returns:
            refined_maps:   (K, M, H, W) after importance map
            refined_scores: (K, M) max scores after spatial reweighting
        """
        K, M, H, W = sim_maps.shape
        device = sim_maps.device

        A = self.build_importance_map(bounding_boxes, H, W, device=device)  # (H, W)

        refined_maps = sim_maps.clone()

        if concept_idx is not None:
            # Apply only to specified concept
            refined_maps[concept_idx] = sim_maps[concept_idx] * A.unsqueeze(0)
        else:
            # Apply to all concepts
            refined_maps = sim_maps * A.unsqueeze(0).unsqueeze(0)

        # Recompute max scores
        refined_scores = refined_maps.amax(dim=(-2, -1))                     # (K, M)

        return refined_maps, refined_scores

    def safety_check(
        self,
        bb: BoundingBox,
        sim_maps: torch.Tensor,
        intended_concept: int,
    ) -> dict:
        """
        [Improvement B] Check if a positive bounding box targets a region
        that is already attributed to a different (possibly wrong) concept.

        Args:
            bb:               the bounding box the doctor is about to draw
            sim_maps:         (K, M, H, W) similarity maps for one image
            intended_concept: the concept the doctor intends to target

        Returns:
            dict with 'safe', 'dominant_concept', 'warning_message' keys
        """
        if not bb.positive:
            return {"safe": True, "dominant_concept": -1, "warning_message": ""}

        K, M, H, W = sim_maps.shape
        x1, x2 = max(0, bb.x1), min(bb.x2, W)
        y1, y2 = max(0, bb.y1), min(bb.y2, H)

        if x1 >= x2 or y1 >= y2:
            return {"safe": True, "dominant_concept": -1, "warning_message": ""}

        # Max similarity per concept in the box region
        box_maps = sim_maps[:, :, y1:y2, x1:x2]                             # (K, M, h, w)
        box_max_per_concept = box_maps.amax(dim=(1, 2, 3))                   # (K,)

        dominant_concept = box_max_per_concept.argmax().item()
        dominant_score = box_max_per_concept[dominant_concept].item()
        intended_score = box_max_per_concept[intended_concept].item()

        safe = (dominant_concept == intended_concept)
        warning = ""

        if not safe and (dominant_score - intended_score) > self.uncertainty_threshold:
            warning = (
                f"Safety check: The drawn region is primarily attributed to "
                f"concept #{dominant_concept} (score={dominant_score:.3f}), "
                f"not the intended concept #{intended_concept} "
                f"(score={intended_score:.3f}). "
                f"Placing a positive box here may amplify an incorrect attribution. "
                f"Consider also rejecting concept #{dominant_concept}."
            )

        return {
            "safe": safe,
            "dominant_concept": dominant_concept,
            "dominant_score": dominant_score,
            "intended_score": intended_score,
            "warning_message": warning,
        }

    def apply_all_feedback(
        self,
        sim_scores: torch.Tensor,
        sim_maps: torch.Tensor,
        feedback: DoctorFeedback,
        run_safety_checks: bool = True,
    ) -> dict:
        """
        Apply all doctor feedback in sequence: concept rejection → spatial interaction.

        Args:
            sim_scores: (K, M) similarity scores for one image
            sim_maps:   (K, M, H, W) similarity maps for one image
            feedback:   DoctorFeedback containing all corrections
            run_safety_checks: if True, collect safety warnings before applying

        Returns:
            dict with:
              'refined_scores': (K, M)
              'refined_maps':   (K, M, H, W)
              'warnings':       list of warning strings (may be empty)
        """
        warnings = []

        # Step 1: Concept-level rejection
        scores = self.apply_concept_rejection(sim_scores, feedback.rejected_concepts)

        # Step 2: Spatial interaction with optional safety checks
        maps = sim_maps.clone()
        for bb in feedback.bounding_boxes:
            if run_safety_checks and bb.positive and bb.concept_idx >= 0:
                check = self.safety_check(bb, maps, bb.concept_idx)
                if check["warning_message"]:
                    warnings.append(check["warning_message"])

            target_concept = bb.concept_idx if bb.concept_idx >= 0 else None
            maps, scores_updated = self.apply_spatial_interaction(maps, [bb], target_concept)

        # Use spatially refined scores
        refined_scores = maps.amax(dim=(-2, -1))                             # (K, M)

        # Override rejected concepts (concept rejection takes precedence)
        for k in feedback.rejected_concepts:
            refined_scores[k] = 0.0

        return {
            "refined_scores": refined_scores,
            "refined_maps": maps,
            "warnings": warnings,
        }
