"""
Evaluation metrics for CSR++.

Primary:
  - macro_f1: Macro-averaged F1 score (main diagnostic metric)

Trustworthiness:
  - pointing_game: Pointing Game hit rate (max activation inside GT bbox)

Uncertainty quality:
  - expected_calibration_error: ECE for uncertainty calibration

Ablation helpers:
  - per_concept_f1: F1 score per concept
"""

import torch
import numpy as np
from sklearn.metrics import f1_score


def macro_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    num_classes: int = None,
) -> float:
    """
    Macro-averaged F1 score for multi-class classification.

    Args:
        predictions: logits or probabilities (N, num_classes) or predicted classes (N,)
        targets:     ground truth class indices (N,)
        threshold:   used only if predictions are probabilities for multi-label
        num_classes: number of classes (inferred if None)

    Returns:
        macro F1 score in [0, 1]
    """
    if predictions.dim() == 2:
        pred_classes = predictions.argmax(dim=-1)
    else:
        pred_classes = predictions

    pred_np = pred_classes.cpu().numpy()
    target_np = targets.cpu().numpy()

    return f1_score(target_np, pred_np, average="macro", zero_division=0)


def pointing_game(
    similarity_maps: torch.Tensor,
    gt_bboxes: list[dict],
    concept_indices: list[int] = None,
    input_size: int = 224,
) -> dict:
    """
    Pointing Game evaluation (from [37] in the paper).

    A "hit" is scored if the point of maximum activation in the similarity map
    lies within the ground-truth bounding box for that concept.

    Args:
        similarity_maps: (B, K, H, W) — max over M prototypes per concept
                         Compute as: sim_maps.amax(dim=2) from (B, K, M, H, W)
        gt_bboxes:       list of B dicts, each mapping concept_idx → (x1,y1,x2,y2)
                         Coordinates are in input_size×input_size image space.
                         Example: [{0: (10, 20, 50, 80), 2: (30, 40, 60, 90)}, ...]
        concept_indices: which concepts to evaluate (all if None)
        input_size:      the image size that bbox coordinates refer to (default 224)

    Returns:
        dict with:
          'hit_rate':      overall hit rate across all evaluated (image, concept) pairs
          'per_concept':   dict mapping concept_idx → hit rate
          'num_evaluated': total number of (image, concept) pairs evaluated
    """
    B, K, H, W = similarity_maps.shape
    if concept_indices is None:
        concept_indices = list(range(K))

    # Scale factor: bbox coords are in input_size space, maps are H×W
    scale_x = W / input_size
    scale_y = H / input_size

    hits = {k: 0 for k in concept_indices}
    totals = {k: 0 for k in concept_indices}

    for b in range(B):
        bbox_dict = gt_bboxes[b] if b < len(gt_bboxes) else {}

        for k in concept_indices:
            if k not in bbox_dict:
                continue

            x1, y1, x2, y2 = bbox_dict[k]

            # Scale bbox from input_size space → feature map (H×W) space
            x1 = max(0, min(int(x1 * scale_x), W - 1))
            x2 = max(1, min(int(x2 * scale_x), W))
            y1 = max(0, min(int(y1 * scale_y), H - 1))
            y2 = max(1, min(int(y2 * scale_y), H))
            # Ensure box has non-zero area after scaling
            if x2 <= x1: x2 = x1 + 1
            if y2 <= y1: y2 = y1 + 1

            # Find maximum activation point
            smap = similarity_maps[b, k]                     # (H, W)
            max_idx = smap.argmax()
            max_h = (max_idx // W).item()
            max_w = (max_idx % W).item()

            # Check if max point is inside the GT bbox
            hit = (x1 <= max_w < x2) and (y1 <= max_h < y2)

            hits[k] += int(hit)
            totals[k] += 1

    per_concept = {
        k: hits[k] / totals[k] if totals[k] > 0 else 0.0
        for k in concept_indices
    }
    total_hits = sum(hits.values())
    total_evals = sum(totals.values())
    overall = total_hits / total_evals if total_evals > 0 else 0.0

    return {
        "hit_rate": overall,
        "per_concept": per_concept,
        "num_evaluated": total_evals,
    }


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well model confidence aligns with actual accuracy.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        confidences: confidence scores (N,) in [0, 1]
        accuracies:  binary correctness (N,) — 1 if prediction was correct
        num_bins:    number of equally-spaced confidence bins

    Returns:
        ECE value in [0, 1] — lower is better
    """
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    N = len(confidences)

    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        bin_weight = mask.sum() / N
        ece += bin_weight * abs(bin_conf - bin_acc)

    return float(ece)


def per_concept_f1(
    concept_predictions: torch.Tensor,
    concept_labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[int, float]:
    """
    Per-concept F1 score for multi-label concept prediction (Stage 1 evaluation).

    Args:
        concept_predictions: sigmoid-activated logits (N, K) in [0,1]
        concept_labels:      binary labels (N, K)
        threshold:           decision boundary

    Returns:
        dict mapping concept_idx → F1 score
    """
    pred_binary = (concept_predictions >= threshold).cpu().numpy()
    labels_np = concept_labels.cpu().numpy()
    K = labels_np.shape[1]

    result = {}
    for k in range(K):
        result[k] = f1_score(labels_np[:, k], pred_binary[:, k], zero_division=0)
    return result
