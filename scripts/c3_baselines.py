"""
C3 — CBM and PIP-Net Baseline Comparison on TBX11K

Implements two concept-based baselines from the literature and evaluates
them on TBX11K (3-class TB detection) for 3 seeds each.

Baselines:
  1. CBM  — Concept Bottleneck Model (Koh et al., 2020 NeurIPS)
             ResNet-50 → concept logits → class head
             PG: per-concept Grad-CAM on concept score w.r.t. feature map
  2. PIPNet — PIP-Net (Nauta et al., 2023 CVPR) style
             ResNet-50 → patch prototypes → prototype presence → class head
             PG: per-concept similarity map (max over M prototypes)

Both use the same ResNet-50 backbone (ImageNet pretrained) and TBX11K dataset.
3 seeds (42, 0, 1) each → 6 training runs total.

Results → results/c3_baselines.json
Update: Tables 1, 2, 10.
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights

from src.data.datasets import CSRDataset, get_dataloader
from src.utils.metrics import macro_f1

DATA_DIR    = "data/tbx11k"
CKPT_BASE   = "checkpoints"
RESULTS     = "results/c3_baselines.json"
LOG_BASE    = "logs"

os.makedirs(CKPT_BASE, exist_ok=True)
os.makedirs(LOG_BASE, exist_ok=True)
os.makedirs("results", exist_ok=True)

SEEDS        = [42, 0, 1]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
BATCH_SIZE   = 128
NUM_WORKERS  = 8
CBM_EPOCHS   = 60
PIP_EPOCHS   = 70   # 35 prototype + 35 head fine-tune
FEATURE_DIM  = 2048
PROTO_DIM    = 128
NUM_PROTOS   = 10   # M prototypes per concept for PIP-Net


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Concept Bottleneck Model (CBM)
# ─────────────────────────────────────────────────────────────────────────────

class CBM(nn.Module):
    """
    Joint CBM: backbone → concept logits → sigmoid → classifier.
    Concept annotations are used as supervision.
    """

    def __init__(self, num_concepts: int, num_classes: int,
                 feature_dim: int = FEATURE_DIM):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Keep feature extractor (up to avgpool), remove final fc
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # (N, 2048, 7, 7)
        self.pool      = nn.AdaptiveAvgPool2d(1)                         # (N, 2048, 1, 1)
        self.concept_head = nn.Linear(feature_dim, num_concepts)
        self.class_head   = nn.Linear(num_concepts, num_classes)

    def forward(self, x):
        feat_map = self.features(x)        # (N, 2048, H, W)
        pooled   = self.pool(feat_map).flatten(1)   # (N, 2048)
        c_logits = self.concept_head(pooled)         # (N, K)
        y_logits = self.class_head(torch.sigmoid(c_logits))  # (N, C)
        return {"logits": y_logits, "concept_logits": c_logits, "feat_map": feat_map}

    @torch.no_grad()
    def get_concept_cam(self, feat_map, concept_idx):
        """
        Class Activation Map for concept k using CAM (Zhou et al., 2016).
        For a GAP → linear head: cam_k = feat_map ⊙ w_k (weight of concept_k).
        Returns (H, W) float tensor on CPU.
        """
        w = self.concept_head.weight[concept_idx]   # (2048,)
        # feat_map: (1, 2048, H, W)
        cam = (feat_map[0] * w.view(-1, 1, 1)).sum(0)  # (H, W)
        cam = cam.clamp(min=0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu()


def get_class_weights(dataset, num_classes, device):
    """Inverse-frequency class weights to handle imbalance."""
    counts = [0] * num_classes
    for s in dataset.samples:
        counts[int(s["class_label"])] += 1
    total = sum(counts)
    weights = torch.tensor([total / (num_classes * c) if c > 0 else 1.0
                             for c in counts], dtype=torch.float32)
    return weights.to(device)


def train_cbm(seed):
    set_seed(seed)
    ckpt_dir = os.path.join(CKPT_BASE, f"tbx11k_cbm_seed{seed}")
    ckpt_path = os.path.join(ckpt_dir, "cbm_best.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.exists(ckpt_path):
        print(f"  CBM seed {seed}: checkpoint exists — skipping training.")
        return ckpt_path

    print(f"\n  Training CBM seed {seed} ...")
    model = CBM(NUM_CONCEPTS, NUM_CLASSES).to(DEVICE)

    train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
    val_ds   = CSRDataset(DATA_DIR, "tbx11k", split="val",   image_size=IMAGE_SIZE)
    train_ldr = get_dataloader(train_ds, BATCH_SIZE, NUM_WORKERS, shuffle=True)
    val_ldr   = get_dataloader(val_ds,   BATCH_SIZE, NUM_WORKERS, shuffle=False)

    # Separate LRs: higher for randomly-initialized heads, lower for pretrained backbone
    opt = Adam([
        {"params": model.features.parameters(),    "lr": 1e-5},
        {"params": model.pool.parameters(),        "lr": 1e-4},
        {"params": model.concept_head.parameters(), "lr": 1e-3},
        {"params": model.class_head.parameters(),   "lr": 1e-3},
    ], weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=CBM_EPOCHS)

    # Class-weighted CE to handle 4600:4600:1200 imbalance
    cw = get_class_weights(train_ds, NUM_CLASSES, DEVICE)
    cls_crit = nn.CrossEntropyLoss(weight=cw)
    con_crit = nn.BCEWithLogitsLoss()
    LAMBDA   = 0.1    # concept loss weight (low: let classification drive early training)

    best_f1, best_state = 0.0, None
    for epoch in range(1, CBM_EPOCHS + 1):
        model.train()
        for batch in train_ldr:
            imgs   = batch["image"].to(DEVICE)
            y_true = batch["class_label"].to(DEVICE)
            c_true = batch["concept_labels"].to(DEVICE)

            out = model(imgs)
            loss = cls_crit(out["logits"], y_true) + \
                   LAMBDA * con_crit(out["concept_logits"], c_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_ldr:
                imgs = batch["image"].to(DEVICE)
                out  = model(imgs)
                preds.append(out["logits"].argmax(-1).cpu())
                labels.append(batch["class_label"])
        f1 = macro_f1(torch.cat(preds), torch.cat(labels), num_classes=NUM_CLASSES)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{CBM_EPOCHS}  val_F1={f1:.4f}  best={best_f1:.4f}")

    torch.save({"model_state": best_state, "best_val_f1": best_f1}, ckpt_path)
    print(f"  CBM seed {seed} done. Best val F1={best_f1:.4f}")
    return ckpt_path


def eval_cbm(ckpt_path, seed):
    set_seed(seed)
    model = CBM(NUM_CONCEPTS, NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Test F1 ──────────────────────────────────────────────────────────────
    test_ds  = CSRDataset(DATA_DIR, "tbx11k", split="test", image_size=IMAGE_SIZE)
    test_ldr = get_dataloader(test_ds, BATCH_SIZE, NUM_WORKERS, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_ldr:
            imgs = batch["image"].to(DEVICE)
            out  = model(imgs)
            preds.append(out["logits"].argmax(-1).cpu())
            labels.append(batch["class_label"])
    f1 = macro_f1(torch.cat(preds), torch.cat(labels), num_classes=NUM_CLASSES)

    # ── Pointing Game (CAM-based) ─────────────────────────────────────────────
    bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
    hits, total = 0, 0
    for i in range(len(bbox_ds)):
        item   = bbox_ds[i]
        bboxes = {int(k): v for k, v in item["bbox"].items()}
        if not bboxes:
            continue
        img = item["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img)
            feat_map = out["feat_map"]   # (1, 2048, H, W)

        image_hit = False
        H = feat_map.shape[2]; W = feat_map.shape[3]
        for c_idx, bbox in bboxes.items():
            cam = model.get_concept_cam(feat_map, c_idx)   # (H, W)
            cam_np = cam.numpy()
            x1, y1, x2, y2 = bbox
            bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
            bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
            by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
            by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
            idx  = int(np.argmax(cam_np))
            mh, mw = divmod(idx, W)
            if (bx1 <= mw < bx2) and (by1 <= mh < by2):
                image_hit = True
        hits  += int(image_hit)
        total += 1

    pg = hits / total if total > 0 else 0.0
    print(f"  CBM seed {seed}: F1={f1:.4f}  PG={pg:.4f}  (hits {hits}/{total})")
    return float(f1), float(pg)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIP-Net (Nauta et al. 2023 CVPR)
# ─────────────────────────────────────────────────────────────────────────────

class PIPNet(nn.Module):
    """
    Simplified PIP-Net:
      - ResNet-50 patch features (H×W×D after conv)
      - K×M prototype vectors (concept-aligned: K concepts, M protos each)
      - Similarity: scaled cosine → sigmoid → max-pool over spatial → presence score
      - Classifier: K×M → C with non-negative weights (exp-parameterized)
    """

    def __init__(self, num_concepts: int, num_protos: int, num_classes: int,
                 proto_dim: int = PROTO_DIM, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.K = num_concepts
        self.M = num_protos
        self.num_classes = num_classes

        # Backbone (strip avgpool + fc)
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # (N, 2048, 7, 7)
        self.add_on   = nn.Sequential(
            nn.Conv2d(feature_dim, proto_dim, 1), nn.Sigmoid()
        )

        # Prototype parameters: (K, M, proto_dim)
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(num_concepts, num_protos, proto_dim), dim=-1)
        )

        # Classifier: log-weight to ensure non-negative after exp, shape (C, K*M)
        # Bias disabled for interpretability
        self.log_class_weights = nn.Parameter(torch.zeros(num_classes, num_concepts * num_protos))
        self.class_bias         = nn.Parameter(torch.zeros(num_classes))

    def _get_similarity_maps(self, patch_feats):
        """
        patch_feats: (N, proto_dim, H, W)
        Returns: (N, K, M, H, W) cosine similarity maps
        """
        N, D, H, W = patch_feats.shape
        # Normalise patch features spatially
        pf_norm = F.normalize(patch_feats, dim=1)           # (N, D, H, W)
        # (N, D, H*W) for matmul
        pf_flat = pf_norm.view(N, D, -1)                    # (N, D, H*W)
        # Protos: (K*M, D)
        proto_flat = F.normalize(self.prototypes, dim=-1).view(-1, D)  # (K*M, D)
        # Cosine sim: (N, K*M, H*W)
        sim = torch.einsum("ndi,md->nmi", pf_flat, proto_flat)        # (N, K*M, H*W)
        sim = sim.view(N, self.K, self.M, H, W)
        return sim

    def forward(self, x):
        feat_map  = self.features(x)                            # (N, 2048, H, W)
        patch_f   = self.add_on(feat_map)                       # (N, proto_dim, H, W)
        sim_maps  = self._get_similarity_maps(patch_f)          # (N, K, M, H, W)
        # Presence score: max over spatial → sigmoid-activated similarity
        presence  = sim_maps.amax(dim=(-2, -1))                 # (N, K, M)
        pres_flat = presence.view(presence.size(0), -1)         # (N, K*M)
        # Non-negative weights via exp
        w = self.log_class_weights.exp()                        # (C, K*M)
        logits = F.linear(pres_flat, w, self.class_bias)        # (N, C)
        return {"logits": logits, "sim_maps": sim_maps, "presence": presence}


def pipnet_push_phase(model, train_ldr, device):
    """
    Push phase: update prototypes to nearest patch feature in the training set.
    Runs one pass over the dataset.
    """
    model.eval()
    K, M, D = model.K, model.M, model.prototypes.shape[-1]
    # For each prototype slot, track the best (highest similarity) patch
    best_sim  = torch.full((K, M), -1e9)
    best_feat = torch.zeros(K, M, D)

    with torch.no_grad():
        for batch in train_ldr:
            imgs     = batch["image"].to(device)
            feat_map = model.features(imgs)
            patch_f  = model.add_on(feat_map)              # (N, D, H, W)
            N, D_, H, W = patch_f.shape
            # Normalise
            pf_norm  = F.normalize(patch_f, dim=1)         # (N, D, H, W)
            pf_flat  = pf_norm.permute(0, 2, 3, 1).reshape(-1, D_)  # (N*H*W, D)
            proto_n  = F.normalize(model.prototypes, dim=-1).view(-1, D_)  # (K*M, D)
            sims     = (pf_flat @ proto_n.T)               # (N*H*W, K*M)
            sims     = sims.view(N, H * W, K, M)
            max_sim, max_idx = sims.max(dim=1)             # (N, K, M)

            for n in range(N):
                for k in range(K):
                    for m in range(M):
                        s = max_sim[n, k, m].item()
                        if s > best_sim[k, m]:
                            best_sim[k, m] = s
                            # Get patch feature
                            hw = max_idx[n, k, m].item()
                            h, w = divmod(hw, W)
                            best_feat[k, m] = pf_flat.view(N, H * W, D_)[n, hw].cpu()

    with torch.no_grad():
        model.prototypes.data = F.normalize(best_feat.to(device), dim=-1)


def train_pipnet(seed):
    set_seed(seed)
    ckpt_dir  = os.path.join(CKPT_BASE, f"tbx11k_pipnet_seed{seed}")
    ckpt_path = os.path.join(ckpt_dir, "pipnet_best.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.exists(ckpt_path):
        print(f"  PIPNet seed {seed}: checkpoint exists — skipping training.")
        return ckpt_path

    print(f"\n  Training PIPNet seed {seed} ...")
    model = PIPNet(NUM_CONCEPTS, NUM_PROTOS, NUM_CLASSES).to(DEVICE)

    train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
    val_ds   = CSRDataset(DATA_DIR, "tbx11k", split="val",   image_size=IMAGE_SIZE)
    train_ldr = get_dataloader(train_ds, BATCH_SIZE, NUM_WORKERS, shuffle=True)
    val_ldr   = get_dataloader(val_ds,   BATCH_SIZE, NUM_WORKERS, shuffle=False)

    # Orthogonality loss between prototypes of different concepts (encourage diversity)
    PUSH_EVERY = 10   # push every N epochs

    # Class-weighted CE for PIPNet too
    train_ds_for_wts = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
    cw = get_class_weights(train_ds_for_wts, NUM_CLASSES, DEVICE)
    cls_crit = nn.CrossEntropyLoss(weight=cw)

    # Phase 1: train all params (proto learning)
    opt1  = Adam([
        {"params": model.features.parameters(),  "lr": 1e-5},
        {"params": model.add_on.parameters(),    "lr": 1e-4},
        {"params": model.prototypes,             "lr": 1e-3},
        {"params": model.log_class_weights,      "lr": 1e-3},
        {"params": model.class_bias,             "lr": 1e-3},
    ], weight_decay=1e-4)
    sched1 = CosineAnnealingLR(opt1, T_max=PIP_EPOCHS // 2)

    # Phase 2: freeze backbone + addon, fine-tune head only
    opt2  = Adam([
        {"params": model.log_class_weights, "lr": 1e-3},
        {"params": model.class_bias,        "lr": 1e-3},
    ], weight_decay=1e-4)
    sched2 = CosineAnnealingLR(opt2, T_max=PIP_EPOCHS // 2)

    best_f1, best_state = 0.0, None
    PHASE2_START = PIP_EPOCHS // 2 + 1

    for epoch in range(1, PIP_EPOCHS + 1):
        opt = opt1 if epoch < PHASE2_START else opt2

        # Freeze/unfreeze
        freeze_backbone = (epoch >= PHASE2_START)
        for p in model.features.parameters():
            p.requires_grad = not freeze_backbone
        for p in model.add_on.parameters():
            p.requires_grad = not freeze_backbone
        model.prototypes.requires_grad = not freeze_backbone

        model.train()
        for batch in train_ldr:
            imgs   = batch["image"].to(DEVICE)
            y_true = batch["class_label"].to(DEVICE)

            out  = model(imgs)
            loss = cls_crit(out["logits"], y_true)  # cls_crit defined above

            # Orthogonality regularizer between proto vectors
            p_norm = F.normalize(model.prototypes.view(-1, PROTO_DIM), dim=-1)  # (K*M, D)
            gram   = p_norm @ p_norm.T                                           # (K*M, K*M)
            I      = torch.eye(gram.shape[0], device=DEVICE)
            orth_loss = (gram - I).pow(2).mean()
            loss = loss + 0.01 * orth_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch < PHASE2_START:
            sched1.step()
            # Prototype push
            if epoch % PUSH_EVERY == 0:
                pipnet_push_phase(model, train_ldr, DEVICE)
        else:
            sched2.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_ldr:
                imgs = batch["image"].to(DEVICE)
                out  = model(imgs)
                preds.append(out["logits"].argmax(-1).cpu())
                labels.append(batch["class_label"])
        f1 = macro_f1(torch.cat(preds), torch.cat(labels), num_classes=NUM_CLASSES)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{PIP_EPOCHS}  val_F1={f1:.4f}  best={best_f1:.4f}")

    torch.save({"model_state": best_state, "best_val_f1": best_f1}, ckpt_path)
    print(f"  PIPNet seed {seed} done. Best val F1={best_f1:.4f}")
    return ckpt_path


def eval_pipnet(ckpt_path, seed):
    set_seed(seed)
    model = PIPNet(NUM_CONCEPTS, NUM_PROTOS, NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Test F1 ──────────────────────────────────────────────────────────────
    test_ds  = CSRDataset(DATA_DIR, "tbx11k", split="test", image_size=IMAGE_SIZE)
    test_ldr = get_dataloader(test_ds, BATCH_SIZE, NUM_WORKERS, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_ldr:
            imgs = batch["image"].to(DEVICE)
            out  = model(imgs)
            preds.append(out["logits"].argmax(-1).cpu())
            labels.append(batch["class_label"])
    f1 = macro_f1(torch.cat(preds), torch.cat(labels), num_classes=NUM_CLASSES)

    # ── Pointing Game (concept similarity map max over M protos) ─────────────
    bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
    hits, total = 0, 0
    for i in range(len(bbox_ds)):
        item   = bbox_ds[i]
        bboxes = {int(k): v for k, v in item["bbox"].items()}
        if not bboxes:
            continue
        img = item["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out      = model(img)
            sim_maps = out["sim_maps"]   # (1, K, M, H, W)

        image_hit = False
        H = sim_maps.shape[-2]; W = sim_maps.shape[-1]
        for c_idx, bbox in bboxes.items():
            # Max over M prototypes for concept c_idx
            sm = sim_maps[0, c_idx].amax(0).cpu().numpy()  # (H, W)
            x1, y1, x2, y2 = bbox
            bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
            bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
            by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
            by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
            idx  = int(np.argmax(sm))
            mh, mw = divmod(idx, W)
            if (bx1 <= mw < bx2) and (by1 <= mh < by2):
                image_hit = True
        hits  += int(image_hit)
        total += 1

    pg = hits / total if total > 0 else 0.0
    print(f"  PIPNet seed {seed}: F1={f1:.4f}  PG={pg:.4f}  (hits {hits}/{total})")
    return float(f1), float(pg)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("C3 — CBM and PIP-Net baselines on TBX11K")
print("=" * 60)
print(f"Device: {DEVICE}")

all_results = {"cbm": {}, "pipnet": {}}

# ── Train all ─────────────────────────────────────────────────────────────────
print("\n--- CBM Training ---")
cbm_ckpts = {}
for seed in SEEDS:
    cbm_ckpts[seed] = train_cbm(seed)

print("\n--- PIPNet Training ---")
pip_ckpts = {}
for seed in SEEDS:
    pip_ckpts[seed] = train_pipnet(seed)

# ── Evaluate all ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating CBM")
print("=" * 60)
for seed in SEEDS:
    if not os.path.exists(cbm_ckpts[seed]):
        print(f"  CBM seed {seed}: checkpoint missing")
        all_results["cbm"][seed] = {"f1": None, "pg": None}
        continue
    f1, pg = eval_cbm(cbm_ckpts[seed], seed)
    all_results["cbm"][seed] = {"f1": f1, "pg": pg}

print("\n" + "=" * 60)
print("Evaluating PIPNet")
print("=" * 60)
for seed in SEEDS:
    if not os.path.exists(pip_ckpts[seed]):
        print(f"  PIPNet seed {seed}: checkpoint missing")
        all_results["pipnet"][seed] = {"f1": None, "pg": None}
        continue
    f1, pg = eval_pipnet(pip_ckpts[seed], seed)
    all_results["pipnet"][seed] = {"f1": f1, "pg": pg}

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("C3 Summary")
print("=" * 60)

for model_name in ["cbm", "pipnet"]:
    f1s = [all_results[model_name][s]["f1"] for s in SEEDS
           if all_results[model_name][s].get("f1") is not None]
    pgs = [all_results[model_name][s]["pg"] for s in SEEDS
           if all_results[model_name][s].get("pg") is not None]
    if f1s:
        mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))
        mean_pg, std_pg = float(np.mean(pgs)), float(np.std(pgs))
        all_results[model_name]["summary"] = {
            "f1_mean": mean_f1, "f1_std": std_f1,
            "pg_mean": mean_pg, "pg_std": std_pg,
        }
        print(f"  {model_name.upper():7s}  F1={mean_f1:.4f}±{std_f1:.4f}  "
              f"PG={mean_pg:.4f}±{std_pg:.4f}  seeds={f1s}")

with open(RESULTS, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nC3 done. Results → {RESULTS}")
print("Update Tables 1, 2, 10 with CBM and PIPNet rows.")
