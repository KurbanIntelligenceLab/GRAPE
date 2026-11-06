"""
N3 — Hyperparameter Sensitivity Analysis

Sweeps:
  1. Graph threshold τ  ∈ {0.01, 0.05, 0.10, 0.20, 0.30}
     → rebuild graph, retrain Stage 4 only (prototypes/projector frozen),
       evaluate Macro F1 + PG on val split.

  2. Safety threshold η ∈ {0.01, 0.03, 0.05, 0.10, 0.20}
     → inference only on bbox_eval, report TP/FP warn rates.
     (Requires c7_safety_eval.py results or runs inline.)

Produces results for updated Appendix A.
"""

import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1, pointing_game
from src.training.trainer import CSRTrainer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/tbx11k"
CKPT_BASE    = "checkpoints"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BF16 needs tensor-core support (A100+, sm_80); older GPUs (e.g. Pascal/P100) fall
# back to FP16, which needs a real GradScaler rather than the no-op bf16 one.
AMP_DTYPE    = (torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.get_device_capability(DEVICE)[0] >= 8
                 else torch.float16)
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
BATCH_SIZE   = 128
NUM_WORKERS  = 4

TAU_SWEEP    = [0.01, 0.05, 0.10, 0.20, 0.30]
ETA_SWEEP    = [0.01, 0.03, 0.05, 0.10, 0.20]

os.makedirs("results", exist_ok=True)
torch.manual_seed(42)

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds    = CSRDataset(DATA_DIR, "tbx11k", split="train",     image_size=IMAGE_SIZE)
val_ds      = CSRDataset(DATA_DIR, "tbx11k", split="val",       image_size=IMAGE_SIZE)
bbox_ds     = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
val_loader   = get_dataloader(val_ds,   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
bbox_loader  = get_dataloader(bbox_ds,  batch_size=1,          num_workers=0, shuffle=False)

# Pre-collect training labels for graph construction
print("Collecting training labels for graph construction...")
all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)


def load_grape_base():
    """Load GRAPE (no VLM) with frozen backbone+projector+prototypes, fresh GNN."""
    path = os.path.join(CKPT_BASE, "tbx11k_no_C_novlm", "csr_stage4_best.pt")
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    return m


def retrain_stage4(model, tau, epochs=20, lr=1e-3):
    """Re-initialise and retrain Stage 4 GNN head with a new graph threshold."""
    # Build graph at this tau
    ei, ew = build_cooccurrence_graph(all_labels, threshold=tau)
    ew_norm = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
    model.set_concept_graph(ei, ew_norm)

    # Freeze backbone, projector, prototypes; train only GNN head
    for name, param in model.named_parameters():
        param.requires_grad = "gnn" in name or "task_head" in name

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    # Re-load train_loader with shuffle
    tr_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model.train()
    for ep in range(epochs):
        for batch in tr_loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                out = model(imgs)
                logits = out["logits"] if isinstance(out, dict) else out
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        scheduler.step()

    model.eval()
    return model


def evaluate(model, tau=None):
    """Evaluate Macro F1 on val set and PG on bbox_eval."""
    if tau is not None:
        ei, ew = build_cooccurrence_graph(all_labels, threshold=tau)
        ew_norm = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
        model.set_concept_graph(ei, ew_norm)

    # F1 on val
    all_preds, all_gt = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(DEVICE)
            out = model(imgs)
            logits = out["logits"] if isinstance(out, dict) else out
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_gt.append(batch["class_label"])
    all_preds = torch.cat(all_preds)
    all_gt    = torch.cat(all_gt)
    f1 = macro_f1(all_preds, all_gt, num_classes=NUM_CLASSES)

    # PG on bbox_eval
    pg_hits, pg_total = 0, 0
    with torch.no_grad():
        for item in bbox_ds:
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if not bboxes:
                continue
            img = item["image"].unsqueeze(0).to(DEVICE)
            feat, _, _ = model.concept_model(img)
            fp = model.projector.project_feature_map(feat)
            sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()  # (K,M,h,w)
            for c_idx, bbox in bboxes.items():
                raw = sm[c_idx].amax(0).numpy()  # (h,w)
                H, W = raw.shape
                x1, y1, x2, y2 = bbox
                bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
                bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
                by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
                by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
                idx = np.argmax(raw)
                mh, mw = divmod(idx, W)
                hit = (bx1 <= mw < bx2) and (by1 <= mh < by2)
                pg_hits  += int(hit)
                pg_total += 1

    pg = pg_hits / pg_total if pg_total > 0 else 0.0
    return float(f1), float(pg)


# ── 1. τ sweep ────────────────────────────────────────────────────────────────
print("\n=== τ (graph threshold) sweep ===")
tau_results = {}
for tau in TAU_SWEEP:
    print(f"  τ = {tau:.2f} ...", end=" ", flush=True)
    model = load_grape_base()
    model = retrain_stage4(model, tau=tau)
    f1, pg = evaluate(model)
    tau_results[tau] = {"macro_f1": f1, "pg": pg}
    print(f"F1={f1:.4f}  PG={pg:.4f}")

print("\nτ sweep results:")
print(f"{'τ':>6}  {'Macro F1':>10}  {'PG':>8}")
print("-" * 28)
for tau, r in tau_results.items():
    print(f"{tau:>6.2f}  {r['macro_f1']:>10.4f}  {r['pg']:>8.4f}")

with open("results/n3_tau_results.json", "w") as f:
    json.dump({str(k): v for k, v in tau_results.items()}, f, indent=2)


# ── 2. η sweep (safety threshold) ─────────────────────────────────────────────
print("\n=== η (safety threshold) sweep ===")
# Re-use the trained GRAPE model (τ=0.10, the default)
model = load_grape_base()
ei, ew = build_cooccurrence_graph(all_labels, threshold=0.10)
model.set_concept_graph(ei, normalize_edge_weights(ei, ew, NUM_CONCEPTS))
model.eval()

import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
N_TRIALS = 5
all_concepts = list(range(NUM_CONCEPTS))
MISC_RATE_FOR_ETA = 0.5   # use p=0.5 to get TP, and p=0 for FP

# Pre-compute sim maps
samples = []
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    bboxes = {int(k): v for k, v in item["bbox"].items()}
    for c_idx, bbox in bboxes.items():
        samples.append((i, c_idx, bbox))

print(f"  Pre-computing sim maps for {len(set(s[0] for s in samples))} images...")
sim_cache = {}
for img_i in set(s[0] for s in samples):
    item = bbox_ds[img_i]
    img = item["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()
    sim_cache[img_i] = sm


def run_check_at_eta(eta, p):
    """Run safety check at given eta and miscorrection rate p."""
    tp_rates, fp_rates = [], []
    for trial in range(N_TRIALS):
        rng = random.Random(RANDOM_SEED + trial)
        tp_w, tp_t, fp_w, fp_t = 0, 0, 0, 0
        for img_i, concept_idx, bbox in samples:
            sm = sim_cache[img_i]
            K, M, H, W = sm.shape
            x1, y1, x2, y2 = bbox
            bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
            bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
            by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
            by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
            mu_k = sm.mean(dim=1)  # (K,H,W)
            region = mu_k[:, by1:by2, bx1:bx2]
            s_bar = region.mean(dim=(-2, -1))
            dominant = int(s_bar.argmax())
            dom_score = float(s_bar[dominant])

            is_misdraw = rng.random() < p
            if is_misdraw:
                wrong_concepts = [c for c in all_concepts if c != concept_idx]
                claimed = rng.choice(wrong_concepts)
                int_score = float(s_bar[claimed])
                warned = (dominant != claimed) and (dom_score - int_score > eta)
                tp_t += 1
                if warned: tp_w += 1
            else:
                int_score = float(s_bar[concept_idx])
                warned = (dominant != concept_idx) and (dom_score - int_score > eta)
                fp_t += 1
                if warned: fp_w += 1

        if tp_t > 0: tp_rates.append(tp_w / tp_t)
        if fp_t > 0: fp_rates.append(fp_w / fp_t)

    return (np.mean(tp_rates) if tp_rates else None,
            np.mean(fp_rates) if fp_rates else None)


eta_results = {}
for eta in ETA_SWEEP:
    tp, fp = run_check_at_eta(eta, p=MISC_RATE_FOR_ETA)
    _, fp0 = run_check_at_eta(eta, p=0.0)
    eta_results[eta] = {"tp_rate": float(tp) if tp is not None else None,
                        "fp_rate": float(fp0) if fp0 is not None else None}
    print(f"  η={eta:.2f}  TP={tp:.3f}  FP={fp0:.3f}")

print(f"\nη sweep (p={MISC_RATE_FOR_ETA} for TP, p=0 for FP):")
print(f"{'η':>6}  {'TP rate':>10}  {'FP rate':>10}")
print("-" * 30)
for eta, r in eta_results.items():
    tp_s = f"{r['tp_rate']:.3f}" if r['tp_rate'] is not None else "—"
    fp_s = f"{r['fp_rate']:.3f}" if r['fp_rate'] is not None else "—"
    print(f"{eta:>6.2f}  {tp_s:>10}  {fp_s:>10}")

with open("results/n3_eta_results.json", "w") as f:
    json.dump({str(k): v for k, v in eta_results.items()}, f, indent=2)

print("\nN3 done. Results in results/")
