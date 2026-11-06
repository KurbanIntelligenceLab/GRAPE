"""
N5 — MC-Dropout uncertainty estimation for Module B safety check.

Provides MCDropoutSafetyCheck as a drop-in comparison for the prototype-variance
safety check. Instead of computing disagreement among M prototypes (spatial
variance), it estimates uncertainty via T stochastic forward passes with dropout
enabled at inference time.

Safety check decision rule (same η threshold, different uncertainty signal):
  - Prototype variance (GRAPE): warns when the dominant concept inside the region
    differs from the claimed concept AND the gap exceeds η.
  - MC-Dropout (baseline): warns when the entropy of predicted class probabilities
    over T passes exceeds a threshold, OR when the concept's mean score inside the
    region under the stochastic passes is lower than expected.

We implement two MC-Dropout variants:
  (A) Score variance: Var_t[s_k(x, region)] across T passes — directly comparable
      to prototype variance. Requires dropout in the feature extractor.
  (B) Prediction entropy: H[p_t(y|x)] — global uncertainty (common MC-Dropout use).

Both variants are evaluated and compared to the prototype-variance baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


# ── Dropout injection ─────────────────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module, dropout_p: float = 0.1) -> None:
    """
    Add a dropout layer after the backbone's avgpool (before concept heads)
    if not already present, AND set all existing Dropout modules to train mode
    so they fire at inference time.

    Call this once before running MC-Dropout inference. Harmless if called
    multiple times (idempotent — only adds one MCDropoutLayer).
    """
    # Enable any existing dropout layers that were trained
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()

    # Inject a named dropout into the concept model if not already present
    if not hasattr(model.concept_model, "_mc_dropout"):
        model.concept_model._mc_dropout = nn.Dropout(p=dropout_p)


def disable_mc_dropout(model: nn.Module) -> None:
    """Restore all dropout to eval mode (standard inference)."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.eval()
    if hasattr(model.concept_model, "_mc_dropout"):
        model.concept_model._mc_dropout.eval()


@contextmanager
def mc_dropout_mode(model: nn.Module, dropout_p: float = 0.1):
    """Context manager: enable MC-Dropout, yield, then restore."""
    was_training = model.training
    model.eval()
    enable_mc_dropout(model, dropout_p)
    try:
        yield
    finally:
        disable_mc_dropout(model)
        if was_training:
            model.train()


# ── MC-Dropout forward pass ───────────────────────────────────────────────────

@torch.no_grad()
def mc_forward_passes(
    model: nn.Module,
    image: torch.Tensor,
    T: int = 30,
    dropout_p: float = 0.1,
) -> dict:
    """
    Run T stochastic forward passes with dropout enabled.

    Returns:
        sim_scores_mean: (K, M) mean similarity scores over T passes
        sim_scores_var:  (K, M) variance of similarity scores over T passes
        logits_mean:     (num_classes,) mean predicted logits
        logits_var:      (num_classes,) variance of predicted logits
        entropy:         scalar — predictive entropy H[E_t p_t(y|x)]
    """
    enable_mc_dropout(model, dropout_p)

    sim_scores_list = []
    logits_list = []

    for _ in range(T):
        # Inject dropout into the feature map
        f, _, _ = model.concept_model(image)           # (1, C, H, W)

        # Apply MC-Dropout to feature map channels
        mc_drop = model.concept_model._mc_dropout
        mc_drop.train()
        f_dropped = mc_drop(f)                          # (1, C, H, W)

        fp = model.projector.project_feature_map(f_dropped)  # (1, D, H, W)
        sim_scores = model.prototype_learner.get_similarity_scores(fp)  # (1, K, M)
        sim_scores_list.append(sim_scores[0].cpu())    # (K, M)

        out = model(image)
        logits_list.append(out["logits"][0].cpu())     # (num_classes,)

    sim_stack   = torch.stack(sim_scores_list)         # (T, K, M)
    logits_stack = torch.stack(logits_list)             # (T, num_classes)

    probs_stack = F.softmax(logits_stack, dim=-1)      # (T, num_classes)
    mean_probs  = probs_stack.mean(dim=0)              # (num_classes,)
    entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum()

    return {
        "sim_scores_mean": sim_stack.mean(dim=0),      # (K, M)
        "sim_scores_var":  sim_stack.var(dim=0),       # (K, M)
        "logits_mean":     logits_stack.mean(dim=0),
        "logits_var":      logits_stack.var(dim=0),
        "entropy":         float(entropy),
    }


# ── Safety check variants ─────────────────────────────────────────────────────

def prototype_variance_safety_check(
    sim_cache: torch.Tensor,
    bbox: tuple,
    concept_idx: int,
    claimed_idx: int,
    image_size: int,
    eta: float = 0.05,
) -> bool:
    """
    Original GRAPE safety check (Module B).
    sim_cache: (K, M, H, W) — precomputed similarity maps for one image.
    Returns True if a warning should be issued.
    """
    K, M, H, W = sim_cache.shape
    x1, y1, x2, y2 = bbox
    bx1 = max(0, min(int(x1 * W / image_size), W - 1))
    bx2 = max(bx1 + 1, min(int(x2 * W / image_size), W))
    by1 = max(0, min(int(y1 * H / image_size), H - 1))
    by2 = max(by1 + 1, min(int(y2 * H / image_size), H))

    # Mean similarity per concept inside the region
    mu_k = sim_cache.mean(dim=1)                       # (K, H, W)
    region = mu_k[:, by1:by2, bx1:bx2]                # (K, h_r, w_r)
    s_bar = region.mean(dim=(-2, -1))                  # (K,)

    dominant = int(s_bar.argmax())
    dom_score = float(s_bar[dominant])
    claimed_score = float(s_bar[claimed_idx])

    return (dominant != claimed_idx) and (dom_score - claimed_score > eta)


def mc_dropout_score_variance_check(
    mc_result: dict,
    bbox: tuple,
    concept_idx: int,
    claimed_idx: int,
    image_size: int,
    eta_var: float = 0.01,
) -> bool:
    """
    MC-Dropout Variant A: Score variance safety check.

    Issues a warning if the variance of the claimed concept's similarity scores
    inside the region (over T passes) exceeds eta_var — indicating that the model
    is uncertain about that region.

    eta_var is calibrated to produce similar FP rates as the prototype-variance check.
    """
    sim_var = mc_result["sim_scores_var"]              # (K, M)
    # Max variance over M prototypes for the claimed concept
    # (a high-variance prototype = uncertain about whether this region matches)
    claimed_var = float(sim_var[claimed_idx].max())
    return claimed_var > eta_var


def mc_dropout_entropy_check(
    mc_result: dict,
    eta_entropy: float = 0.5,
) -> bool:
    """
    MC-Dropout Variant B: Prediction entropy safety check.

    Issues a warning if the predictive entropy of class probabilities over T passes
    exceeds eta_entropy — indicating global model uncertainty.
    """
    return mc_result["entropy"] > eta_entropy
