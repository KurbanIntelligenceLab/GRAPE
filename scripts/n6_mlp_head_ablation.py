"""
N6 — GAT vs. Parameter-Matched MLP Task Head Ablation (TBX11K)

Is the GAT gain from graph structure, or from added
nonlinearity over a flat linear head?

We train a 2-layer MLP on the flattened K×M = 300 score vector that has
~the same parameter count as the GAT head (~45K params each), with identical
Stage 1-3 weights and Stage 4 training protocol. If the MLP matches the GAT,
the gain is from nonlinearity; if the GAT wins, the graph structure matters.

Results → results/n6_mlp_ablation.json
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1, pointing_game

DATA_DIR     = "data/tbx11k"
CKPT_STAGE3  = "checkpoints/tbx11k_no_C_novlm/csr_stage3.pt"
CKPT_GAT     = "checkpoints/tbx11k_no_C_novlm/csr_stage4_best.pt"
OUT_DIR      = "checkpoints/tbx11k_mlp_head"
RESULTS_PATH = "results/n6_mlp_ablation.json"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
NUM_PROTO    = 100
BATCH_SIZE   = 128
NUM_WORKERS  = 8
EPOCHS       = 20
LR           = 1e-3
WD           = 1e-4
GRAPH_TAU    = 0.10

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)
torch.manual_seed(42)


# ── Parameter-matched MLP task head ──────────────────────────────────────────
class MLPTaskHead(nn.Module):
    """
    2-layer MLP over flattened K×M prototype scores.
    Designed to match GAT param count (~45K) to isolate graph structure value.

    Input: (B, K, M) → flatten → (B, K*M=300)
    300 → 128 → 64 → num_classes
    Params: 300*128 + 128 + 128*64 + 64 + 64*C = 38,400+8,256+192 ≈ 46,848
    GAT params (K=3, M=100, h=64, heads=4): ≈ 45,600
    """
    def __init__(self, num_concepts, num_prototypes, num_classes,
                 hidden1=128, hidden2=64):
        super().__init__()
        in_dim = num_concepts * num_prototypes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, sim_scores, edge_index=None, edge_weight=None):
        # sim_scores: (B, K, M)
        x = sim_scores.flatten(1)   # (B, K*M)
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train",     image_size=IMAGE_SIZE)
val_ds   = CSRDataset(DATA_DIR, "tbx11k", split="val",       image_size=IMAGE_SIZE)
test_ds  = CSRDataset(DATA_DIR, "tbx11k", split="test",      image_size=IMAGE_SIZE)
bbox_ds  = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

train_loader = get_dataloader(train_ds, BATCH_SIZE, NUM_WORKERS, shuffle=True)
val_loader   = get_dataloader(val_ds,   BATCH_SIZE, NUM_WORKERS, shuffle=False)
test_loader  = get_dataloader(test_ds,  BATCH_SIZE, NUM_WORKERS, shuffle=False)

# Build graph (needed only for the GAT eval, but we keep it for consistency)
print("Building co-occurrence graph...")
all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
ei, ew     = build_cooccurrence_graph(all_labels, threshold=GRAPH_TAU)
ew_norm    = normalize_edge_weights(ei, ew, NUM_CONCEPTS)


# ── Load Stage 1-3 base ───────────────────────────────────────────────────────
def load_stage3_base():
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=NUM_PROTO, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=False, use_uncertainty=False, use_vlm=False,
        pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(CKPT_STAGE3, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    return m


# ── Evaluation helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels_list = [], []
    for batch in loader:
        imgs   = batch["image"].to(DEVICE)
        labels = batch["class_label"].to(DEVICE)
        out    = model(imgs)
        all_logits.append(out["logits"].cpu())
        all_labels_list.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels_list)
    return macro_f1(logits, labels, num_classes=NUM_CLASSES)


@torch.no_grad()
def eval_pg(model, bbox_ds_ref):
    """Pointing Game on concept 0 (Active TB) using the same inline logic as c4."""
    model.eval()
    hits, total = 0, 0
    for i in range(len(bbox_ds_ref)):
        item   = bbox_ds_ref[i]
        bboxes = {int(k): v for k, v in item["bbox"].items()}
        if 0 not in bboxes:
            continue
        img = item["image"].unsqueeze(0).to(DEVICE)
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()  # (K,M,H,W)
        raw = sm[0].amax(0).numpy()  # max over M prototypes, concept 0
        H, W = raw.shape
        x1, y1, x2, y2 = bboxes[0]
        bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
        bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
        by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
        by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
        idx  = int(np.argmax(raw))
        mh, mw = divmod(idx, W)
        hits  += int((bx1 <= mw < bx2) and (by1 <= mh < by2))
        total += 1
    return hits / total if total > 0 else 0.0


# ── Stage 4 training loop ─────────────────────────────────────────────────────
def train_stage4(model, head_name):
    # Freeze everything except task_head
    for name, p in model.named_parameters():
        p.requires_grad = "task_head" in name

    opt      = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WD,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val_f1, best_state = 0.0, None

    print(f"\n── {head_name} Stage 4 training ──")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable:,}")

    for ep in range(EPOCHS):
        model.train()
        for batch in train_loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = criterion(out["logits"], labels)
            loss.backward()
            opt.step()

        val_f1 = evaluate(model, val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"   Ep {ep+1:02d}/{EPOCHS}  val_F1={val_f1:.4f}  best={best_val_f1:.4f}")

    model.load_state_dict(best_state)
    return model


# ════════════════════════════════════════════════════════════════════════════
# Experiment 1: MLP head
# ════════════════════════════════════════════════════════════════════════════
print("\n=== MLP Task Head ===")
model_mlp = load_stage3_base()
model_mlp.task_head = MLPTaskHead(NUM_CONCEPTS, NUM_PROTO, NUM_CLASSES).to(DEVICE)
print(f"MLP params: {model_mlp.task_head.count_params():,}")
model_mlp = train_stage4(model_mlp, "MLP")
test_f1_mlp = evaluate(model_mlp, test_loader)
pg_mlp      = eval_pg(model_mlp, bbox_ds)
print(f"MLP  → test F1={test_f1_mlp:.4f}  PG={pg_mlp:.4f}")
torch.save({"model_state": model_mlp.state_dict()},
           os.path.join(OUT_DIR, "mlp_stage4_best.pt"))


# ════════════════════════════════════════════════════════════════════════════
# Experiment 2: Linear head (original CSR baseline)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Linear Task Head (CSR) ===")
model_lin = load_stage3_base()
# task_head is already nn.Linear(K*M, C) from use_gnn=False
lin_params = sum(p.numel() for p in model_lin.task_head.parameters())
print(f"Linear params: {lin_params:,}")
model_lin = train_stage4(model_lin, "Linear")
test_f1_lin = evaluate(model_lin, test_loader)
pg_lin      = eval_pg(model_lin, bbox_ds)
print(f"Linear → test F1={test_f1_lin:.4f}  PG={pg_lin:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 3: GAT head (load existing best checkpoint)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== GAT Task Head (GRAPE) ===")
model_gat = CSRModel(
    num_concepts=NUM_CONCEPTS, num_prototypes=NUM_PROTO, num_classes=NUM_CLASSES,
    proto_dim=256, use_gnn=True, use_uncertainty=False, use_vlm=False,
    pretrained_backbone=False,
).to(DEVICE)
ckpt = torch.load(CKPT_GAT, map_location=DEVICE, weights_only=False)
model_gat.load_state_dict(ckpt["model_state"], strict=False)
model_gat.set_concept_graph(ei, ew_norm)
gat_params = sum(p.numel() for p in model_gat.task_head.parameters())
print(f"GAT params: {gat_params:,}")
test_f1_gat = evaluate(model_gat, test_loader)
pg_gat      = eval_pg(model_gat, bbox_ds)
print(f"GAT  → test F1={test_f1_gat:.4f}  PG={pg_gat:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
results = {
    "Linear (CSR)":   {"f1": float(test_f1_lin), "pg": float(pg_lin),
                       "params": lin_params},
    "MLP (matched)":  {"f1": float(test_f1_mlp), "pg": float(pg_mlp),
                       "params": model_mlp.task_head.count_params()},
    "GAT (GRAPE)":    {"f1": float(test_f1_gat), "pg": float(pg_gat),
                       "params": gat_params},
}

print("\n=== N6 Summary ===")
print(f"{'Head':<18} {'F1':>8} {'PG':>8} {'Params':>10}")
print("-" * 48)
for name, r in results.items():
    print(f"{name:<18} {r['f1']:>8.4f} {r['pg']:>8.4f} {r['params']:>10,}")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {RESULTS_PATH}")
