"""
Similarity map comparison: CSR | CBM | PIPNet | GRAPE
Selects images where GRAPE's heatmap energy inside the GT box is most superior.
Saves figures/simmaps_comparison.png
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
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
CONCEPT_IDX  = 0   # Active TB
N_PICK       = 3   # images to show

# warm heatmap, transparent at low values
CMAP = LinearSegmentedColormap.from_list(
    "warm", [(0,"#00000000"), (0.3,"#FF6600AA"), (0.7,"#FF0000CC"), (1,"#8B0000EE")], N=256
)

# ── loaders ────────────────────────────────────────────────────────
def load_csr(ckpt, use_gnn):
    path = os.path.join(CKPT_BASE, ckpt, "csr_stage4_best.pt")
    m = CSRModel(num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
                 proto_dim=256, use_gnn=use_gnn, use_uncertainty=use_gnn,
                 use_vlm=False, pretrained_backbone=False).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["model_state"], strict=False)
    m.eval()
    if use_gnn:
        train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
        ldr      = get_dataloader(train_ds, 128, 4, shuffle=False)
        labels   = torch.cat([b["concept_labels"] for b in ldr]).to(DEVICE)
        ei, ew   = build_cooccurrence_graph(labels, threshold=0.10)
        m.set_concept_graph(ei, normalize_edge_weights(ei, ew, NUM_CONCEPTS))
    return m

def load_cbm():
    m = CBM(num_concepts=NUM_CONCEPTS, num_classes=NUM_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(f"{CKPT_BASE}/tbx11k_cbm_seed42/cbm_best.pt",
                                 map_location=DEVICE, weights_only=False)["model_state"])
    return m.eval()

def load_pipnet():
    m = PIPNet(num_concepts=NUM_CONCEPTS, num_protos=10, num_classes=NUM_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(f"{CKPT_BASE}/tbx11k_pipnet_seed42/pipnet_best.pt",
                                 map_location=DEVICE, weights_only=False)["model_state"])
    return m.eval()

# ── map extractors → normalised (H,W) numpy ────────────────────────
def _resize(raw):
    return cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

def _norm(m):
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo + 1e-8)

def get_sm_csr(model, img_t):
    with torch.no_grad():
        f, _, _ = model.concept_model(img_t.unsqueeze(0).to(DEVICE))
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
    return _norm(_resize(sm[0, CONCEPT_IDX].mean(0).cpu().numpy()))

def get_sm_cbm(model, img_t):
    with torch.no_grad():
        out = model(img_t.unsqueeze(0).to(DEVICE))
        cam = model.get_concept_cam(out["feat_map"], CONCEPT_IDX)
    return _norm(_resize(cam.numpy()))

def get_sm_pipnet(model, img_t):
    with torch.no_grad():
        out = model(img_t.unsqueeze(0).to(DEVICE))
        raw = out["sim_maps"][0, CONCEPT_IDX].amax(0).cpu().numpy()
    return _norm(_resize(raw))

def box_energy(sm, bbox):
    """Fraction of total heatmap energy inside the GT box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    inside = sm[y1:y2, x1:x2].sum()
    total  = sm.sum() + 1e-8
    return float(inside / total)

def overlay(img_np, sm):
    """Blend heatmap over X-ray."""
    rgba  = CMAP(sm)                         # (H,W,4)
    comp  = img_np.astype(float) / 255.0
    alpha = rgba[:, :, 3:4]
    for c in range(3):
        comp[:, :, c] = comp[:, :, c] * (1 - alpha[:, :, 0]) + rgba[:, :, c] * alpha[:, :, 0]
    return np.clip(comp, 0, 1)

# ── load ──────────────────────────────────────────────────────────
print("Loading models...")
m_csr    = load_csr("tbx11k_baseline_csr", use_gnn=False)
m_grape  = load_csr("tbx11k_no_C_novlm",   use_gnn=True)
m_cbm    = load_cbm()
m_pipnet = load_pipnet()

print("Loading data...")
bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
mean_img = np.array([0.485, 0.456, 0.406])
std_img  = np.array([0.229, 0.224, 0.225])

samples = [(i, {int(k): v for k, v in bbox_ds[i]["bbox"].items()}[CONCEPT_IDX])
           for i in range(len(bbox_ds))
           if CONCEPT_IDX in {int(k): v for k, v in bbox_ds[i]["bbox"].items()}]

# ── score all images, pick best GRAPE-advantage cases ─────────────
print("Scoring all images...")
scored = []
for n, (idx, bbox) in enumerate(samples):
    if n % 20 == 0: print(f"  {n}/{len(samples)}", flush=True)
    img_t = bbox_ds[idx]["image"]

    sm_grape = get_sm_csr(m_grape, img_t)
    e_grape  = box_energy(sm_grape, bbox)
    if e_grape < 0.15:       # skip weak GRAPE cases
        continue

    sm_csr_  = get_sm_csr(m_csr,    img_t)
    sm_cbm_  = get_sm_cbm(m_cbm,    img_t)
    sm_pip_  = get_sm_pipnet(m_pipnet, img_t)

    e_csr  = box_energy(sm_csr_,  bbox)
    e_cbm  = box_energy(sm_cbm_,  bbox)
    e_pip  = box_energy(sm_pip_,  bbox)

    # advantage = GRAPE energy − mean of others
    advantage = e_grape - np.mean([e_csr, e_cbm, e_pip])

    img_np = (img_t.permute(1,2,0).numpy() * std_img + mean_img)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    scored.append({
        "idx": idx, "bbox": bbox, "img_np": img_np,
        "sm": {"CSR Baseline": sm_csr_, "CBM": sm_cbm_,
               "PIP-Net": sm_pip_,  "Proposed\nGRAPE": sm_grape},
        "energy": {"CSR Baseline": e_csr, "CBM": e_cbm,
                   "PIP-Net": e_pip, "Proposed\nGRAPE": e_grape},
        "advantage": advantage,
    })

scored.sort(key=lambda r: -r["advantage"])
chosen = scored[:N_PICK]
print(f"Top-{N_PICK} selected (advantage = GRAPE box-energy − mean-others):")
for r in chosen:
    e = r["energy"]
    grape_e = e["Proposed\nGRAPE"]
    print(f"  idx={r['idx']}  GRAPE={grape_e:.3f}  "
          f"CSR={e['CSR Baseline']:.3f}  CBM={e['CBM']:.3f}  PIP={e['PIP-Net']:.3f}  "
          f"adv={r['advantage']:.3f}")

# ── plot ───────────────────────────────────────────────────────────
METHODS  = ["CSR Baseline", "CBM", "PIP-Net", "Proposed\nGRAPE"]
MCOLS    = ["#E53935",       "#8E24AA", "#0097A7", "#F6A800"]
NCOLS    = 1 + len(METHODS)   # Original + 4 methods
NROWS    = N_PICK

fig = plt.figure(figsize=(NCOLS * 2.6, NROWS * 2.9), facecolor="white")
gs  = GridSpec(NROWS, NCOLS, figure=fig,
               hspace=0.03, wspace=0.03,
               left=0.01, right=0.99, top=0.99, bottom=0.01)

# column headers (row 0 only)
col_titles = ["Original + GT box"] + METHODS
col_colors = ["#333333"] + MCOLS

for row, r in enumerate(chosen):
    bbox = r["bbox"]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Col 0: original
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.imshow(r["img_np"])
    ax0.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                  lw=2, edgecolor="#00BFFF", facecolor="none", zorder=3))
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values(): sp.set_visible(False)

    # Cols 1-4: heatmap overlays
    for col, (mname, mcol) in enumerate(zip(METHODS, MCOLS), start=1):
        ax = fig.add_subplot(gs[row, col])
        sm = r["sm"][mname]
        ax.imshow(overlay(r["img_np"], sm))

        # GT box
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                     lw=2, edgecolor="#00BFFF", facecolor="none", zorder=4))

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(mcol if mname == "Proposed\nGRAPE" else "#cccccc")
            sp.set_linewidth(2.5 if mname == "Proposed\nGRAPE" else 0.5)
            sp.set_visible(True)

os.makedirs("figures", exist_ok=True)
out = "figures/simmaps_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print(f"\nSaved → {out}")
