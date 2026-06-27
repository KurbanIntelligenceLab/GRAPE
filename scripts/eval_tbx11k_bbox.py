"""
Evaluate TBX11K ablation checkpoints on bbox test split (Pointing Game).

For each variant:
  1. Load checkpoint from ${CKPT_BASE}/tbx11k_{name}/csr_stage4_best.pt
  2. Run inference on bbox_eval split (200 images with TB bounding boxes)
  3. Report: Macro F1 + Pointing Game per-concept and overall
"""

import os

import torch
from torch.utils.data import DataLoader

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.metrics import macro_f1, pointing_game
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

# ── Config ────────────────────────────────────────────────────────────────────
# Paths are taken from environment variables so the script runs anywhere.
DATA_DIR     = os.environ.get("DATA_DIR", "data/tbx11k")
CKPT_BASE    = os.environ.get("CKPT_BASE", "checkpoints")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64
NUM_WORKERS  = 4
IMAGE_SIZE   = 224
NUM_CONCEPTS = len(CONCEPTS["tbx11k"])    # 3
NUM_CLASSES  = len(CLASS_NAMES["tbx11k"]) # 3

CONCEPT_NAMES = CONCEPTS["tbx11k"]

EXPERIMENTS = {
    "full_ABC":    dict(use_gnn=True,  use_vlm=True,  use_uncertainty=True),
    "no_A_linear": dict(use_gnn=False, use_vlm=True,  use_uncertainty=True),
    "no_C_novlm":  dict(use_gnn=True,  use_vlm=False, use_uncertainty=True),
    "baseline_CSR":dict(use_gnn=False, use_vlm=False, use_uncertainty=False),
}

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading bbox_eval split...")
test_ds = CSRDataset(
    root_dir=DATA_DIR,
    dataset_name="tbx11k",
    split="bbox_eval",
    image_size=IMAGE_SIZE,
)
test_loader = get_dataloader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
print(f"  bbox_eval samples: {len(test_ds)}")

print("Loading train split (for graph construction)...")
train_ds = CSRDataset(root_dir=DATA_DIR, dataset_name="tbx11k", split="train", image_size=IMAGE_SIZE)
train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

print("Building concept co-occurrence graph...")
all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
edge_index, edge_weight = build_cooccurrence_graph(all_labels, threshold=0.1)
edge_weight = normalize_edge_weights(edge_index, edge_weight, NUM_CONCEPTS)
print(f"  Graph: {NUM_CONCEPTS} nodes, {edge_index.shape[1]} edges")


# ── Evaluation function ───────────────────────────────────────────────────────
def evaluate(name: str, flags: dict) -> dict:
    ckpt_path = os.path.join(CKPT_BASE, f"tbx11k_{name}", "csr_stage4_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [{name}] checkpoint not found: {ckpt_path}")
        return {}

    model = CSRModel(
        num_concepts=NUM_CONCEPTS,
        num_prototypes=100,
        num_classes=NUM_CLASSES,
        proto_dim=256,
        use_gnn=flags["use_gnn"],
        use_uncertainty=flags["use_uncertainty"],
        use_vlm=flags["use_vlm"],
        pretrained_backbone=False,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    if flags["use_gnn"]:
        model.set_concept_graph(edge_index, edge_weight)

    all_preds, all_targets = [], []
    all_sim_maps = []
    all_bboxes   = []

    with torch.no_grad():
        for batch in test_loader:
            images  = batch["image"].to(DEVICE)
            targets = batch["class_label"]
            out = model(images, return_maps=True)
            all_preds.append(out["logits"].cpu())
            all_targets.append(targets)
            if "sim_maps" in out:
                maps = out["sim_maps"].cpu()           # (B, K, M, H, W)
                all_sim_maps.append(maps.amax(dim=2))  # (B, K, H, W)
            all_bboxes.extend(batch["bbox"])

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    f1      = macro_f1(preds, targets)

    pg_result = {"hit_rate": 0.0, "num_evaluated": 0, "per_concept": {}}
    if all_sim_maps:
        sim_maps_tensor = torch.cat(all_sim_maps)
        bboxes_clean = [{int(k): v for k, v in d.items()} for d in all_bboxes]
        pg_result = pointing_game(sim_maps_tensor, bboxes_clean)

    return {"f1": f1, "pg": pg_result}


# ── Run ───────────────────────────────────────────────────────────────────────
results = {}
for name, flags in EXPERIMENTS.items():
    print(f"\nEvaluating: {name} ...")
    r = evaluate(name, flags)
    results[name] = r
    if r:
        print(f"  F1:            {r['f1']:.4f}")
        print(f"  Pointing Game: {r['pg']['hit_rate']:.4f}  ({r['pg']['num_evaluated']} evaluated)")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'TBX11K RESULTS SUMMARY':^65}")
print("=" * 65)
print(f"{'Variant':<20} {'Test F1':>10} {'PG Overall':>12} {'#Eval':>8}")
print("-" * 65)
for name, r in results.items():
    if not r:
        print(f"{name:<20} {'N/A':>10} {'N/A':>12} {'N/A':>8}")
        continue
    print(f"{name:<20} {r['f1']:>10.4f} {r['pg']['hit_rate']:>12.4f} {r['pg']['num_evaluated']:>8}")

best_name = max(results, key=lambda n: results[n].get("pg", {}).get("hit_rate", 0))
best_pg   = results[best_name].get("pg", {})
if best_pg.get("per_concept"):
    print(f"\nPer-concept Pointing Game — {best_name}:")
    print(f"  {'Concept':<32} {'Hit Rate':>10}")
    print(f"  {'-'*44}")
    per = best_pg["per_concept"]
    for k in sorted(per.keys()):
        cname = CONCEPT_NAMES[k] if k < len(CONCEPT_NAMES) else f"concept_{k}"
        print(f"  {cname:<32} {per[k]:>10.4f}")

print("\nDone.")
