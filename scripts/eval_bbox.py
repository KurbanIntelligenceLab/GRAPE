"""
Re-evaluate all trained models on the bbox test set to get Pointing Game scores.

For each ablation variant:
  1. Load the trained model checkpoint (stage4_best.pt)
  2. Run inference on the nih_bbox TEST split
  3. Compute: Macro F1 + Pointing Game hit rate per concept + overall

Results are printed as a summary table.
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.metrics import macro_f1, pointing_game
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

# ── Config ────────────────────────────────────────────────────────────────────
# Paths are taken from environment variables so the script runs anywhere.
BBOX_DATA_DIR  = os.environ.get("DATA_DIR", "data/nih_bbox")
CKPT_BASE      = os.environ.get("CKPT_BASE", "checkpoints")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 64
NUM_WORKERS    = 4
IMAGE_SIZE     = 224
NUM_CONCEPTS   = len(CONCEPTS["nih"])   # 14
NUM_CLASSES    = len(CLASS_NAMES["nih"]) # 2

FINDING_NAMES = CONCEPTS["nih"]

EXPERIMENTS = {
    "full_ABC":    dict(use_gnn=True,  use_vlm=True,  use_uncertainty=True),
    "no_A_linear": dict(use_gnn=False, use_vlm=True,  use_uncertainty=True),
    "no_C_novlm":  dict(use_gnn=True,  use_vlm=False, use_uncertainty=True),
    "baseline_CSR":dict(use_gnn=False, use_vlm=False, use_uncertainty=False),
}

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading bbox eval set...")
test_ds = CSRDataset(
    root_dir=BBOX_DATA_DIR,
    dataset_name="nih",
    split="bbox_eval",   # dedicated split: all 709 bbox-annotated images
    image_size=IMAGE_SIZE,
)
test_loader = get_dataloader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
print(f"  Test samples: {len(test_ds)}")

# Also load train split to build the concept graph
train_ds = CSRDataset(
    root_dir=BBOX_DATA_DIR,
    dataset_name="nih",
    split="train",
    image_size=IMAGE_SIZE,
)
train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

# Pre-build graph from training labels (used by GNN variants)
print("Building concept graph from training labels...")
all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
edge_index, edge_weight = build_cooccurrence_graph(all_labels, threshold=0.05)
edge_weight = normalize_edge_weights(edge_index, edge_weight, NUM_CONCEPTS)
print(f"  Graph: {NUM_CONCEPTS} nodes, {edge_index.shape[1]} edges")


# ── Evaluation function ───────────────────────────────────────────────────────
def evaluate_model(name: str, flags: dict) -> dict:
    ckpt_path = os.path.join(CKPT_BASE, f"ablation_{name}", "csr_stage4_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [{name}] checkpoint not found: {ckpt_path}")
        return {}

    # Build model with matching flags
    model = CSRModel(
        num_concepts=NUM_CONCEPTS,
        num_prototypes=100,
        num_classes=NUM_CLASSES,
        proto_dim=256,
        use_gnn=flags["use_gnn"],
        use_uncertainty=flags["use_uncertainty"],
        use_vlm=flags["use_vlm"],
        pretrained_backbone=False,  # loading weights, no need for pretrained
    ).to(DEVICE)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # Register graph for GNN variants
    if flags["use_gnn"]:
        model.set_concept_graph(edge_index, edge_weight)

    # Inference
    all_preds, all_targets = [], []
    all_sim_maps = []   # (B, K, H, W) — max over M prototypes
    all_bboxes   = []

    with torch.no_grad():
        for batch in test_loader:
            images  = batch["image"].to(DEVICE)
            targets = batch["class_label"]

            out = model(images, return_maps=True)
            all_preds.append(out["logits"].cpu())
            all_targets.append(targets)

            # sim_maps: (B, K, M, H, W) → max over M → (B, K, H, W)
            if "sim_maps" in out:
                maps = out["sim_maps"].cpu()           # (B, K, M, H, W)
                all_sim_maps.append(maps.amax(dim=2))  # (B, K, H, W)
            all_bboxes.extend(batch["bbox"])

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    f1      = macro_f1(preds, targets)

    # Pointing Game
    pg_result = {"hit_rate": 0.0, "num_evaluated": 0, "per_concept": {}}
    if all_sim_maps:
        sim_maps_tensor = torch.cat(all_sim_maps)      # (N, K, H, W)
        # Convert bbox list-of-dicts to list-of-dicts with int keys
        bboxes_clean = [{int(k): v for k, v in d.items()} for d in all_bboxes]
        pg_result = pointing_game(sim_maps_tensor, bboxes_clean)

    return {"f1": f1, "pg": pg_result}


# ── Run all experiments ───────────────────────────────────────────────────────
results = {}
for name, flags in EXPERIMENTS.items():
    print(f"\nEvaluating: {name} ...")
    r = evaluate_model(name, flags)
    results[name] = r
    if r:
        print(f"  F1:            {r['f1']:.4f}")
        print(f"  Pointing Game: {r['pg']['hit_rate']:.4f}  ({r['pg']['num_evaluated']} evaluated)")


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'RESULTS SUMMARY':^65}")
print("=" * 65)
print(f"{'Variant':<20} {'Test F1':>10} {'PG Overall':>12} {'#Eval':>8}")
print("-" * 65)
for name, r in results.items():
    if not r:
        print(f"{name:<20} {'N/A':>10} {'N/A':>12} {'N/A':>8}")
        continue
    print(f"{name:<20} {r['f1']:>10.4f} {r['pg']['hit_rate']:>12.4f} {r['pg']['num_evaluated']:>8}")

# Per-concept Pointing Game for best model
best_name = max(results, key=lambda n: results[n].get("pg", {}).get("hit_rate", 0))
best_pg   = results[best_name].get("pg", {})
if best_pg.get("per_concept"):
    print(f"\nPer-concept Pointing Game — {best_name} (best PG):")
    print(f"  {'Concept':<22} {'Hit Rate':>10}")
    print(f"  {'-'*34}")
    per = best_pg["per_concept"]
    for k in sorted(per.keys()):
        name_k = FINDING_NAMES[k] if k < len(FINDING_NAMES) else f"concept_{k}"
        print(f"  {name_k:<22} {per[k]:>10.4f}")

print("\nDone.")
