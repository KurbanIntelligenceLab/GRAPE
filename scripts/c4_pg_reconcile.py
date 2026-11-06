"""
C4 — Pointing Game Aggregation Reconciliation

Table 3 (per-concept) weighted average = (120×0.1707 + 87×0.0233)/207 ≈ 0.108
Table 1 (aggregate) reports 0.1401 for the same model.

Root cause: the two numbers use different aggregation rules.
  - Per-pair PG:  each (image, concept, bbox) triple is one trial.
                  A single image with 2 annotated concepts contributes 2 trials.
  - Per-image PG: each image is one trial, HIT if ANY concept's max
                  activation lands in its respective box.

This script computes BOTH metrics from the same checkpoint so we can:
  1. Confirm the discrepancy and its root cause.
  2. Report both in the paper (Table 3 footnote + Table 1 caption update).

Usage:
  python scripts/c4_pg_reconcile.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

DATA_DIR   = "data/tbx11k"
CKPT_PATH  = "checkpoints/tbx11k_no_C_novlm/csr_stage4_best.pt"
RESULTS_PATH = "results/c4_pg_reconcile.json"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3

os.makedirs("results", exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = CSRModel(
    num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
    proto_dim=256, use_gnn=True, use_uncertainty=True,
    use_vlm=False, pretrained_backbone=False,
).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# ── Build and register concept graph ──────────────────────────────────────────
_train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
_loader   = get_dataloader(_train_ds, 128, 4, shuffle=False)
_labels   = torch.cat([b["concept_labels"] for b in _loader]).to(DEVICE)
_ei, _ew  = build_cooccurrence_graph(_labels, threshold=0.10)
_ew_norm  = normalize_edge_weights(_ei, _ew, NUM_CONCEPTS)
model.set_concept_graph(_ei, _ew_norm)
del _train_ds, _loader, _labels

# ── Load bbox_eval split ───────────────────────────────────────────────────────
bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
print(f"bbox_eval images: {len(bbox_ds)}")

CONCEPT_NAMES = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis", "PulmonaryTuberculosis"]

# Per-pair accumulators
per_pair_hits  = {k: 0 for k in range(NUM_CONCEPTS)}
per_pair_total = {k: 0 for k in range(NUM_CONCEPTS)}

# Per-image accumulators  (HIT if any concept hits)
per_image_hits  = 0
per_image_total = 0

with torch.no_grad():
    for i in range(len(bbox_ds)):
        item   = bbox_ds[i]
        bboxes = {int(k): v for k, v in item["bbox"].items()}
        if not bboxes:
            continue

        img = item["image"].unsqueeze(0).to(DEVICE)
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()  # (K,M,H,W)

        image_hit = False
        for c_idx, bbox in bboxes.items():
            raw = sm[c_idx].amax(0).numpy()   # (H, W) — max over prototypes
            H, W = raw.shape
            x1, y1, x2, y2 = bbox
            bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
            bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
            by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
            by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))

            idx      = int(np.argmax(raw))
            mh, mw   = divmod(idx, W)
            hit      = (bx1 <= mw < bx2) and (by1 <= mh < by2)

            per_pair_hits[c_idx]  += int(hit)
            per_pair_total[c_idx] += 1
            if hit:
                image_hit = True

        per_image_hits  += int(image_hit)
        per_image_total += 1

# ── Results ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("C4: Pointing Game Aggregation Reconciliation")
print("="*60)

print("\nPer-PAIR Pointing Game (each bbox triple is one trial):")
total_pairs = 0
total_hits  = 0
for k, name in enumerate(CONCEPT_NAMES):
    n  = per_pair_total[k]
    h  = per_pair_hits[k]
    pg = h / n if n > 0 else float("nan")
    print(f"  {name:<35}: {h:>3}/{n:>3}  PG = {pg:.4f}")
    total_pairs += n
    total_hits  += h
per_pair_pg_overall = total_hits / total_pairs if total_pairs else 0.0
# Weighted average (should match Table 3 implied calc)
per_pair_pg_weighted = sum(
    per_pair_hits[k] / per_pair_total[k] * per_pair_total[k]
    for k in range(NUM_CONCEPTS) if per_pair_total[k] > 0
) / total_pairs if total_pairs > 0 else 0.0

print(f"\n  Total pairs: {total_pairs}  Hits: {total_hits}")
print(f"  Per-pair PG (overall ratio):   {per_pair_pg_overall:.4f}")
print(f"  Per-pair PG (weighted avg):    {per_pair_pg_weighted:.4f}")

print("\nPer-IMAGE Pointing Game (HIT if ANY concept hits on that image):")
per_image_pg = per_image_hits / per_image_total if per_image_total > 0 else 0.0
print(f"  Images: {per_image_total}  Hits: {per_image_hits}  PG = {per_image_pg:.4f}")

print("\n" + "-"*60)
print("RECONCILIATION:")
print(f"  Table 3 implied per-pair PG = {per_pair_pg_overall:.4f}")
print(f"  Table 1 reported aggregate  = 0.1401  (per-image rule?)")
print(f"  Per-image PG this run       = {per_image_pg:.4f}")

results = {
    "per_concept": {
        CONCEPT_NAMES[k]: {
            "hits": per_pair_hits[k],
            "total": per_pair_total[k],
            "pg": per_pair_hits[k] / per_pair_total[k] if per_pair_total[k] > 0 else None,
        }
        for k in range(NUM_CONCEPTS)
    },
    "per_pair_pg_overall":  per_pair_pg_overall,
    "per_pair_pg_weighted": per_pair_pg_weighted,
    "per_image_pg":         per_image_pg,
    "total_pairs":          total_pairs,
    "total_images":         per_image_total,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nC4 done. Results → {RESULTS_PATH}")
