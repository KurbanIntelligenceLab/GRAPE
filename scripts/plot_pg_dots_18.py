"""
Pointing Game overlay — all GRAPE-HIT images, hollow dots per method, zero gap.
Saves figures/pg_dots_18.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

sys.path.insert(0, "scripts")
from c3_baselines import CBM, PIPNet

DATA_DIR     = "data/tbx11k"
CKPT_BASE    = "checkpoints"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
CONCEPT_IDX  = 0
NCOLS        = 5     # rows determined by number of GRAPE-HIT images

METHODS = [
    {"key": "csr",    "color": "#E53935", "marker": "s",  "label": "CSR Baseline",        "ms": 90},
    {"key": "cbm",    "color": "#8E24AA", "marker": "^",  "label": "CBM  (Koh 2020)",     "ms": 95},
    {"key": "pipnet", "color": "#0097A7", "marker": "D",  "label": "PIP-Net  (Nauta 2023)","ms": 85},
    {"key": "grape",  "color": "#F9A825", "marker": "o",  "label": "Proposed GRAPE",       "ms": 110},
]

# ── loaders ───────────────────────────────────────────────────────
def load_csr(ckpt_name, use_gnn):
    path = os.path.join(CKPT_BASE, ckpt_name, "csr_stage4_best.pt")
    m = CSRModel(num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
                 proto_dim=256, use_gnn=use_gnn, use_uncertainty=use_gnn,
                 use_vlm=False, pretrained_backbone=False).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False); m.eval()
    if use_gnn:
        train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
        loader   = get_dataloader(train_ds, 128, 4, shuffle=False)
        labels   = torch.cat([b["concept_labels"] for b in loader]).to(DEVICE)
        ei, ew   = build_cooccurrence_graph(labels, threshold=0.10)
        ew       = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
        m.set_concept_graph(ei, ew)
    return m

def load_cbm():
    path = os.path.join(CKPT_BASE, "tbx11k_cbm_seed42", "cbm_best.pt")
    m = CBM(num_concepts=NUM_CONCEPTS, num_classes=NUM_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["model_state"])
    return m.eval()

def load_pipnet():
    path = os.path.join(CKPT_BASE, "tbx11k_pipnet_seed42", "pipnet_best.pt")
    m = PIPNet(num_concepts=NUM_CONCEPTS, num_protos=10, num_classes=NUM_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["model_state"])
    return m.eval()

# ── extractors ────────────────────────────────────────────────────
def maxpt(sm):
    r, c = np.unravel_index(np.argmax(sm), sm.shape)
    return int(c), int(r)

def sm_csr(model, img_t):
    with torch.no_grad():
        f, _, _ = model.concept_model(img_t.unsqueeze(0).to(DEVICE))
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
    return cv2.resize(sm[0, CONCEPT_IDX].mean(0).cpu().numpy(),
                      (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

def sm_cbm(model, img_t):
    with torch.no_grad():
        out = model(img_t.unsqueeze(0).to(DEVICE))
        cam = model.get_concept_cam(out["feat_map"], CONCEPT_IDX)
    return cv2.resize(cam.numpy(), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

def sm_pipnet(model, img_t):
    with torch.no_grad():
        out = model(img_t.unsqueeze(0).to(DEVICE))
        raw = out["sim_maps"][0, CONCEPT_IDX].amax(0).cpu().numpy()
    return cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

def inside(x, y, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return (x1 <= x < x2) and (y1 <= y < y2)

# ── load ──────────────────────────────────────────────────────────
print("Loading models...")
m_csr    = load_csr("tbx11k_baseline_csr", use_gnn=False)
m_grape  = load_csr("tbx11k_no_C_novlm",   use_gnn=True)
m_cbm    = load_cbm()
m_pipnet = load_pipnet()

print("Loading data...")
bbox_ds  = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
mean_img = np.array([0.485, 0.456, 0.406])
std_img  = np.array([0.229, 0.224, 0.225])

samples = [(i, {int(k): v for k, v in bbox_ds[i]["bbox"].items()}[CONCEPT_IDX])
           for i in range(len(bbox_ds))
           if CONCEPT_IDX in {int(k): v for k, v in bbox_ds[i]["bbox"].items()}]

# ── compute all, keep only GRAPE-HIT ─────────────────────────────
print("Computing points...")
chosen = []
for n, (idx, bbox) in enumerate(samples):
    if n % 20 == 0: print(f"  {n}/{len(samples)}", flush=True)
    img_t  = bbox_ds[idx]["image"]
    s_grp  = sm_csr(m_grape, img_t)
    px, py = maxpt(s_grp)
    if not inside(px, py, bbox):
        continue   # skip non-GRAPE-HIT images

    s_csr_ = sm_csr(m_csr,    img_t)
    s_cbm_ = sm_cbm(m_cbm,    img_t)
    s_pip_ = sm_pipnet(m_pipnet, img_t)

    img_np = (img_t.permute(1,2,0).numpy() * std_img + mean_img)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    pts = {"csr": maxpt(s_csr_), "cbm": maxpt(s_cbm_),
           "pipnet": maxpt(s_pip_), "grape": (px, py)}
    chosen.append({"img_np": img_np, "bbox": bbox, "pts": pts})

print(f"GRAPE-HIT images: {len(chosen)}")

# sort: GRAPE-only hits first (other methods miss) = most informative
def other_hits(r):
    b = r["bbox"]
    return sum(inside(*r["pts"][k], b) for k in ["csr","cbm","pipnet"])
chosen.sort(key=other_hits)

# layout: fill NCOLS×NROWS, pad with extras if needed
NROWS = (len(chosen) + NCOLS - 1) // NCOLS
N     = NROWS * NCOLS

# ── plot ──────────────────────────────────────────────────────────
CELL = 2.35   # inches per cell
fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(NCOLS * CELL, NROWS * CELL))
fig.patch.set_facecolor("white")
# force zero gap
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                    hspace=0, wspace=0)

for i in range(N):
    row, col = divmod(i, NCOLS)
    ax = axes[row, col] if NROWS > 1 else axes[col]
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    if i >= len(chosen):
        ax.set_visible(False)
        continue

    r = chosen[i]
    ax.imshow(r["img_np"])

    x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
    ax.add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=1.8, edgecolor="#00BFFF", facecolor="none", zorder=3))

    for m in METHODS:
        px, py = r["pts"][m["key"]]
        ax.scatter(px, py, s=m["ms"], marker=m["marker"],
                   facecolors="none", edgecolors=m["color"],
                   linewidths=2.2, zorder=5)

os.makedirs("figures", exist_ok=True)
out_path = "figures/pg_dots_18.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05,
            facecolor="white", edgecolor="none")
print(f"Saved → {out_path}")
