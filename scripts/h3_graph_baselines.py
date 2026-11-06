"""
H3 — Graph Head Baselines: ML-GCN and ADD-GCN (Table H3 in paper)

Replaces GRAPE's GAT task head with:
  (a) ML-GCN  (Chen et al., CVPR 2019) — symmetric-norm GCN
  (b) ADD-GCN (Ye et al., ECCV 2020)   — static + dynamic adjacency GCN

All other GRAPE components are identical (same backbone, prototypes, graph).
We load the Stage 1–3 checkpoint from the best GRAPE (no VLM) run, freeze
backbone/projector/prototypes, and re-train only Stage 4 (task head) for
20 epochs with each head type.

Results → new table in §4.4 (ablation) comparing GAT vs ML-GCN vs ADD-GCN.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1, pointing_game

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data/tbx11k"
CKPT_STAGE3   = "checkpoints/tbx11k_no_C_novlm/csr_stage3.pt"  # shared Stage 1-3
CKPT_OUT_DIR  = "checkpoints/tbx11k_h3"
RESULTS_PATH  = "results/h3_graph_baselines.json"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BF16 needs tensor-core support (A100+, sm_80); older GPUs (e.g. Pascal/P100) fall
# back to FP16, which needs a real GradScaler rather than the no-op bf16 one.
AMP_DTYPE     = (torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.get_device_capability(DEVICE)[0] >= 8
                  else torch.float16)
IMAGE_SIZE    = 224
NUM_CONCEPTS  = 3
NUM_CLASSES   = 3
BATCH_SIZE    = 128
NUM_WORKERS   = 8
STAGE4_EPOCHS = 20
STAGE4_LR     = 1e-3
GRAPH_TAU     = 0.10

os.makedirs(CKPT_OUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)
torch.manual_seed(42)

# ── Datasets ──────────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train",     image_size=IMAGE_SIZE)
val_ds   = CSRDataset(DATA_DIR, "tbx11k", split="val",       image_size=IMAGE_SIZE)
test_ds  = CSRDataset(DATA_DIR, "tbx11k", split="test",      image_size=IMAGE_SIZE)
bbox_ds  = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

train_loader = get_dataloader(train_ds, BATCH_SIZE, NUM_WORKERS, shuffle=True)
val_loader   = get_dataloader(val_ds,   BATCH_SIZE, NUM_WORKERS, shuffle=False)
test_loader  = get_dataloader(test_ds,  BATCH_SIZE, NUM_WORKERS, shuffle=False)

# ── Pre-build concept graph ───────────────────────────────────────────────────
print("Building concept co-occurrence graph...")
all_labels = torch.cat([b["concept_labels"] for b in
                        get_dataloader(train_ds, BATCH_SIZE, NUM_WORKERS, shuffle=False)])
all_labels = all_labels.to(DEVICE)
edge_index, edge_weight = build_cooccurrence_graph(all_labels, threshold=GRAPH_TAU)
edge_weight_norm = normalize_edge_weights(edge_index, edge_weight, NUM_CONCEPTS)


def load_stage3_checkpoint(head_type: str) -> CSRModel:
    """Build model, load Stage 1-3 weights, freeze everything except task head."""
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
        task_head_type=head_type,
    ).to(DEVICE)

    ckpt = torch.load(CKPT_STAGE3, map_location=DEVICE, weights_only=False)
    # Load backbone + projector + prototypes from Stage 3 checkpoint
    # (task head weights are freshly initialized — intentionally not loaded)
    missing, unexpected = m.load_state_dict(ckpt["model_state"], strict=False)
    task_head_keys = [k for k in missing if "task_head" in k]
    non_task_missing = [k for k in missing if "task_head" not in k]
    if non_task_missing:
        print(f"  Warning: non-task-head keys missing: {non_task_missing[:5]}")

    # Freeze everything except task head
    for name, param in m.named_parameters():
        param.requires_grad = "task_head" in name

    # Register graph
    m.set_concept_graph(edge_index, edge_weight_norm)
    return m


def train_stage4(model: CSRModel, head_type: str) -> CSRModel:
    """Train only the task head for STAGE4_EPOCHS epochs."""
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=STAGE4_LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE4_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_val_f1, best_state = 0.0, None

    for ep in range(1, STAGE4_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    with torch.no_grad():
                        f, _, _ = model.concept_model(imgs)
                        fp = model.projector.project_feature_map(f)
                        sim_scores = model.prototype_learner.get_similarity_scores(fp)
                    logits = model.task_head(sim_scores)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad():
                    f, _, _ = model.concept_model(imgs)
                    fp = model.projector.project_feature_map(f)
                    sim_scores = model.prototype_learner.get_similarity_scores(fp)
                logits = model.task_head(sim_scores)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        scheduler.step()

        if ep % 5 == 0 or ep == STAGE4_EPOCHS:
            val_f1 = _eval_f1(model, val_loader)
            print(f"  [{head_type}] Epoch {ep:>2}/{STAGE4_EPOCHS}  val_f1={val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def _eval_f1(model, loader):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            out = model(imgs)
            preds.append(out["logits"].argmax(-1).cpu())
            gts.append(batch["class_label"])
    return float(macro_f1(torch.cat(preds), torch.cat(gts), num_classes=NUM_CLASSES))


def _eval_pg(model, bbox_ds):
    """Pointing Game on bbox_eval split."""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for i in range(len(bbox_ds)):
            item = bbox_ds[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if not bboxes:
                continue
            img = item["image"].unsqueeze(0).to(DEVICE)
            f, _, _ = model.concept_model(img)
            fp = model.projector.project_feature_map(f)
            sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()  # (K,M,H,W)
            for c_idx, bbox in bboxes.items():
                raw = sm[c_idx].amax(0).numpy()  # (H, W)
                H, W = raw.shape
                x1, y1, x2, y2 = bbox
                bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
                bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
                by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
                by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
                idx = np.argmax(raw)
                mh, mw = divmod(idx, W)
                hit = (bx1 <= mw < bx2) and (by1 <= mh < by2)
                hits  += int(hit)
                total += 1
    return hits / total if total > 0 else 0.0, total


# ── Main loop ─────────────────────────────────────────────────────────────────
HEAD_TYPES = ["gat", "mlgcn", "addgcn"]
results = {}

for head_type in HEAD_TYPES:
    print(f"\n{'='*60}")
    print(f"  Head type: {head_type.upper()}")
    print(f"{'='*60}")

    model = load_stage3_checkpoint(head_type)
    model = train_stage4(model, head_type)

    # Test evaluation
    test_f1 = _eval_f1(model, test_loader)
    pg, n_pairs = _eval_pg(model, bbox_ds)
    print(f"  [{head_type}] Test F1={test_f1:.4f}  PG={pg:.4f}  ({n_pairs} pairs)")

    results[head_type] = {"macro_f1": test_f1, "pg": pg, "n_pairs": n_pairs}

    # Save checkpoint
    torch.save({"model_state": model.state_dict(), "head_type": head_type},
               os.path.join(CKPT_OUT_DIR, f"h3_{head_type}.pt"))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("H3 Graph Head Baseline Results (TBX11K)")
print(f"{'Head':>10}  {'Macro F1':>10}  {'PG':>8}")
print("-" * 32)
for h, r in results.items():
    print(f"{h.upper():>10}  {r['macro_f1']:>10.4f}  {r['pg']:>8.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nH3 done. Results → {RESULTS_PATH}")
