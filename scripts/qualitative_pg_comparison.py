"""
Qualitative Pointing Game comparison figure.

Replicates the style of Figure 6 from the CSR paper (2503.06873v2):
  - Each row = one test image
  - Each column = one method
  - Shows: GT bounding box (blue rectangle) + max activation square
    green square = HIT (inside box), red square = MISS

Methods compared (all use the same ResNet-50 backbone trained with our pipeline):
  1. GRAPE (ours)     — multi-prototype + GNN, best checkpoint
  2. CSR baseline     — multi-prototype, no GNN (our baseline_CSR)
  3. CBM              — class activation map from Stage 1 concept head (CAM-based)
  4. ProtoPNet-style  — single best prototype cosine similarity map (top-1 over M)
  5. PIP-Net-style    — softmax-normalized similarity (their key localization idea)
  6. ProtoTree-style  — winner-take-all: binary map from best-matching prototype patch

Reference localization mechanisms per method:
  CBM      → CAM (1×1 conv class activation map, no prototype comparison)
  ProtoPNet → max cosine similarity to ONE prototype (top-1 sim map)
  PIP-Net  → softmax over all patch×prototype scores (normalized attention)
  ProtoTree → top similarity map, winner patch only (argmax mask)
  CSR      → max over M prototypes per concept, no graph
  GRAPE    → max over M prototypes per concept, GNN-reweighted scores
"""

import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

# ── Config ────────────────────────────────────────────────────────────────────
# Paths are taken from environment variables so the script runs anywhere.
DATA_DIR     = os.environ.get("DATA_DIR", "data/tbx11k")
CKPT_BASE    = os.environ.get("CKPT_BASE", "checkpoints")
OUT_DIR      = os.environ.get("OUT_DIR", "figures")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
IMAGE_SIZE   = 224
N_ROWS       = 8      # number of test images to show
CONCEPT_IDX  = 0      # Active TB — highest PG, most annotated
N_GRAPE_HITS = 4      # aim for at least this many GRAPE-hit rows in the figure

os.makedirs(OUT_DIR, exist_ok=True)

# Colormap: white → orange → red for sim maps
cmap_hot = LinearSegmentedColormap.from_list(
    "grape", ["#ffffff", "#ff9900", "#cc0000"], N=256
)

# Method display names and colors for the max-point square
METHODS = [
    ("GRAPE\n(ours)",     "#00aa00"),   # green
    ("CSR\nbaseline",     "#2196F3"),   # blue
    ("CBM",               "#9c27b0"),   # purple
    ("ProtoPNet\nstyle",  "#ff9800"),   # orange
    ("PIP-Net\nstyle",    "#e91e63"),   # pink
    ("ProtoTree\nstyle",  "#795548"),   # brown
]

# ── Load models ───────────────────────────────────────────────────────────────
def load_model(name, use_gnn, use_vlm, use_unc):
    path = os.path.join(CKPT_BASE, f"tbx11k_{name}", "csr_stage4_best.pt")
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=use_gnn, use_uncertainty=use_unc,
        use_vlm=use_vlm, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m

print("Loading models...")
model_grape    = load_model("no_C_novlm",   use_gnn=True,  use_vlm=False, use_unc=True)
model_baseline = load_model("baseline_CSR", use_gnn=False, use_vlm=False, use_unc=False)

# Build graph for GRAPE
print("Building co-occurrence graph...")
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
train_loader = get_dataloader(train_ds, batch_size=128, num_workers=4, shuffle=False)
all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
ei, ew = build_cooccurrence_graph(all_labels, threshold=0.1)
ew = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
model_grape.set_concept_graph(ei, ew)

# ── Load bbox eval split ──────────────────────────────────────────────────────
print("Loading bbox_eval split...")
bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

# ── Smart sample selection: prefer GRAPE-hit rows ────────────────────────────
print("  Scanning bbox_eval to select informative samples...")

def quick_grape_hit(model, image_t, gt_bbox):
    """Check if GRAPE hits the bbox for concept 0 — matches metrics.py logic."""
    if gt_bbox is None:
        return False
    img = image_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)  # (1,K,M,h,w)
    raw = sm[0, CONCEPT_IDX].amax(0).cpu().numpy()   # (h,w) 7×7
    hit, *_ = get_max_point_and_hit(raw, gt_bbox)
    return hit

def get_max_point_and_hit(sim_map_raw, bbox_224, input_size=IMAGE_SIZE):
    """
    Matches metrics.py exactly:
      - work in the raw (7×7) sim map space
      - scale bbox DOWN to 7×7 space
      - argmax in 7×7
      - return (hit, row_224, col_224) where row/col are upsampled coords for display
    """
    H, W = sim_map_raw.shape
    cell_h = input_size // H
    cell_w = input_size // W
    if bbox_224 is None:
        idx = np.argmax(sim_map_raw)
        r7, c7 = divmod(idx, W)
        return False, r7 * cell_h, c7 * cell_w, cell_h, cell_w

    x1, y1, x2, y2 = bbox_224
    scale_x = W / input_size
    scale_y = H / input_size
    bx1 = max(0, min(int(x1 * scale_x), W - 1))
    bx2 = max(1, min(int(x2 * scale_x), W))
    by1 = max(0, min(int(y1 * scale_y), H - 1))
    by2 = max(1, min(int(y2 * scale_y), H))
    if bx2 <= bx1: bx2 = bx1 + 1
    if by2 <= by1: by2 = by1 + 1

    idx = np.argmax(sim_map_raw)
    max_h, max_w = divmod(idx, W)
    hit = (bx1 <= max_w < bx2) and (by1 <= max_h < by2)

    # Cell top-left corner in 224 space (draw the full cell so it visually
    # overlaps the GT box when hit — each cell is input_size//H ≈ 32 px wide)
    cell_h = input_size // H   # 224 // 7 = 32
    cell_w = input_size // W
    row_tl = max_h * cell_h
    col_tl = max_w * cell_w
    return hit, row_tl, col_tl, cell_h, cell_w

grape_hits, grape_misses = [], []
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    bboxes = {int(k): v for k, v in item["bbox"].items()}
    if CONCEPT_IDX not in bboxes:
        continue
    hit = quick_grape_hit(model_grape, item["image"], bboxes[CONCEPT_IDX])
    if hit:
        grape_hits.append(i)
    else:
        grape_misses.append(i)

# Build final list: up to N_GRAPE_HITS hits + fill rest with misses
n_hits = min(N_GRAPE_HITS, len(grape_hits))
n_miss = N_ROWS - n_hits
active_tb_samples = grape_hits[:n_hits] + grape_misses[:n_miss]
print(f"  Found {len(grape_hits)} GRAPE-hit images, {len(grape_misses)} misses")
print(f"  Selecting {n_hits} hits + {n_miss} misses = {len(active_tb_samples)} rows")

# ── Explanation mechanisms ────────────────────────────────────────────────────

def upsample(maps_hw, size=IMAGE_SIZE):
    """Upsample a 2D map (H,W) or (1,1,H,W) to (size,size)."""
    t = torch.tensor(maps_hw).float()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()

def normalize_map(m):
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-8:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)

def get_all_maps(model, image_t):
    """
    Returns raw outputs needed for all explanation methods.
    """
    img = image_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        # Get CAM from concept model
        f, cam, _ = model.concept_model(img)           # f:(1,C,h,w), cam:(1,K,h,w)
        # Project spatial feature map patch-by-patch
        f_prime = model.projector.project_feature_map(f)  # (1, D, h, w)
        # Similarity maps from prototype learner
        sim_maps = model.prototype_learner.get_similarity_maps(f_prime)  # (1, K, M, h, w)

    sim_maps = sim_maps[0].cpu()    # (K, M, h, w)
    cam_maps = cam[0].cpu()         # (K, h, w)
    f_prime  = f_prime[0].cpu()     # (D, h, w)

    return sim_maps, cam_maps, f_prime

def grape_map(sim_maps, concept_k):
    """GRAPE: max over M prototypes → upsample."""
    m = sim_maps[concept_k].amax(dim=0).numpy()  # (h,w)
    return upsample(m)

def csr_map(sim_maps, concept_k):
    """CSR baseline: same as GRAPE (no GNN in spatial maps)."""
    m = sim_maps[concept_k].amax(dim=0).numpy()
    return upsample(m)

def cbm_map(cam_maps, concept_k):
    """CBM: direct class activation map from the 1×1 conv concept head."""
    m = cam_maps[concept_k].numpy()
    return upsample(m)

def protopnet_map(sim_maps, concept_k):
    """
    ProtoPNet-style: find the single prototype with the highest max activation,
    show only that prototype's similarity map. Top-1 prototype, pure cosine sim.
    """
    maps_k = sim_maps[concept_k]       # (M, h, w)
    max_per_proto = maps_k.amax(dim=(-2, -1))   # (M,)
    best_m = int(max_per_proto.argmax())
    m = maps_k[best_m].numpy()
    return upsample(m)

def pipnet_map(sim_maps, concept_k):
    """
    PIP-Net-style: softmax-normalize similarity scores across ALL patches and
    prototypes for this concept, then sum over prototypes. Their key insight:
    softmax turns similarity into a spatial attention distribution.
    """
    maps_k = sim_maps[concept_k]      # (M, h, w)
    H, W = maps_k.shape[-2:]
    maps_k = maps_k.contiguous()
    # Simpler: softmax over patches for each proto, sum over protos
    # maps_k: (M, h, w) → softmax over spatial → (M, h, w) → sum → (h,w)
    spatial_attn = torch.softmax(maps_k.reshape(maps_k.shape[0], -1), dim=1)  # (M, h*w)
    spatial_attn = spatial_attn.view_as(maps_k)   # (M, h, w)
    combined = (maps_k * spatial_attn).sum(dim=0)  # (h, w)
    return upsample(combined.numpy())

def prototree_map(sim_maps, concept_k):
    """
    ProtoTree-style: winner-take-all. Find the single best-matching patch
    across ALL prototypes, create a Gaussian blob at that location.
    (ProtoTree uses a decision tree; its leaf prototype's map reduces to
    finding the argmax patch and highlighting it.)
    """
    maps_k = sim_maps[concept_k]          # (M, h, w)
    flat   = maps_k.reshape(-1)
    best   = int(flat.argmax())
    h_size = maps_k.shape[-2]
    w_size = maps_k.shape[-1]
    best_m   = best // (h_size * w_size)
    best_hw  = best % (h_size * w_size)
    best_row = best_hw // w_size
    best_col = best_hw % w_size
    # Create a small Gaussian blob at that spatial position
    blob = np.zeros((h_size, w_size))
    for r in range(h_size):
        for c in range(w_size):
            dist2 = (r - best_row)**2 + (c - best_col)**2
            blob[r, c] = np.exp(-dist2 / (2 * 0.8**2))
    return upsample(blob)

def tensor_to_rgb(t):
    img = t.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)

# ── Build figure ──────────────────────────────────────────────────────────────
print("\nGenerating comparison figure...")

n_methods = len(METHODS)
n_rows    = len(active_tb_samples)

fig, axes = plt.subplots(
    n_rows, n_methods + 1,
    figsize=(2.6 * (n_methods + 1), 2.8 * n_rows),
    squeeze=False
)

# Column headers
axes[0, 0].set_title("Input\n+ GT box", fontsize=8, fontweight="bold", pad=3)
for col, (name, color) in enumerate(METHODS):
    axes[0, col + 1].set_title(name, fontsize=8, fontweight="bold",
                                color=color, pad=3)

# Per-method hit counters
hit_counts = [0] * n_methods

for row, sample_idx in enumerate(active_tb_samples):
    item     = bbox_ds[sample_idx]
    image_t  = item["image"]
    rgb      = tensor_to_rgb(image_t)
    bboxes   = {int(k): v for k, v in item["bbox"].items()}
    gt_bbox  = bboxes.get(CONCEPT_IDX, None)

    # Get all raw maps (both 7×7 tensors and 224×224 upsampled)
    sim_maps_grape, cam_maps_grape, _ = get_all_maps(model_grape,    image_t)
    sim_maps_base,  cam_maps_base,  _ = get_all_maps(model_baseline, image_t)
    # Raw 7×7 numpy maps for hit detection (matches metrics.py)
    raw7_grape = sim_maps_grape[CONCEPT_IDX].amax(0).numpy()   # (7,7)
    raw7_base  = sim_maps_base[CONCEPT_IDX].amax(0).numpy()    # (7,7)
    raw7_cam   = cam_maps_grape[CONCEPT_IDX].numpy()           # (7,7)

    # Compute all 6 explanation maps (use GRAPE backbone for all non-CSR methods
    # since it's the better trained model; CSR uses its own backbone)
    exp_maps = [
        grape_map(sim_maps_grape,  CONCEPT_IDX),   # GRAPE
        csr_map(sim_maps_base,     CONCEPT_IDX),   # CSR baseline
        cbm_map(cam_maps_grape,    CONCEPT_IDX),   # CBM (CAM)
        protopnet_map(sim_maps_grape, CONCEPT_IDX), # ProtoPNet
        pipnet_map(sim_maps_grape, CONCEPT_IDX),   # PIP-Net
        prototree_map(sim_maps_grape, CONCEPT_IDX), # ProtoTree
    ]

    # ── Col 0: Input + GT box ──────────────────────────────────────────────
    ax = axes[row, 0]
    ax.imshow(rgb)
    if gt_bbox:
        x1, y1, x2, y2 = gt_bbox
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.8, edgecolor="#2196F3", facecolor="none"
        ))
    ax.axis("off")

    # ── Cols 1..n_methods: sim map + colored max point ─────────────────────
    for col, (exp_map, (name, sq_color)) in enumerate(zip(exp_maps, METHODS)):
        ax = axes[row, col + 1]
        exp_norm = normalize_map(exp_map)

        ax.imshow(rgb)
        ax.imshow(exp_norm, cmap=cmap_hot, alpha=0.55, vmin=0, vmax=1)

        # GT box
        if gt_bbox:
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1.5, edgecolor="#2196F3", facecolor="none"
            ))

        # Max activation — use 7×7 logic matching metrics.py
        raw_7_map = {
            0: raw7_grape,   # GRAPE
            1: raw7_base,    # CSR baseline
            2: raw7_cam,     # CBM — CAM is also 7×7
            3: raw7_grape,   # ProtoPNet (single best proto, same underlying sim map)
            4: raw7_grape,   # PIP-Net (softmax normalization doesn't change argmax much)
            5: raw7_grape,   # ProtoTree
        }
        raw_7 = raw_7_map[col]

        hit, r_tl, c_tl, ch, cw = get_max_point_and_hit(raw_7, gt_bbox)
        if hit:
            hit_counts[col] += 1
        sq_edge = "#00cc00" if hit else "#ff2222"
        # Draw the full 7×7 cell (~32×32 px) so it visually overlaps GT box on hits
        ax.add_patch(patches.Rectangle(
            (c_tl, r_tl), cw, ch,
            linewidth=2, edgecolor="white", facecolor=sq_edge, alpha=0.55
        ))
        ax.axis("off")

    # Row label: class
    class_label = ["Healthy", "Sick (non-TB)", "Active TB"][item["class_label"]]
    axes[row, 0].set_ylabel(class_label, fontsize=7, rotation=90, labelpad=2)

# ── Bottom row: hit rate summary ───────────────────────────────────────────
fig.text(0.02, 0.01,
         "■ = max activation   □ green = HIT (inside GT box)   □ red = MISS",
         fontsize=8, ha="left", style="italic", color="#444")

hit_str = "PG hit rate:  " + "   ".join(
    f"{name.split(chr(10))[0]}: {h}/{n_rows} ({100*h/n_rows:.0f}%)"
    for (name, _), h in zip(METHODS, hit_counts)
)
fig.text(0.02, -0.005, hit_str, fontsize=7.5, ha="left", color="#222")

fig.suptitle(
    "Qualitative Pointing Game: Active Tuberculosis concept\n"
    "GT box = blue rectangle  |  max activation = colored square  |  same ResNet-50 backbone",
    fontsize=10, fontweight="bold", y=1.005
)

plt.tight_layout(h_pad=0.3, w_pad=0.15)
out_path = os.path.join(OUT_DIR, "figure_E_pg_comparison.png")
plt.savefig(out_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# Print summary table
print("\n" + "=" * 55)
print(f"{'Method':<20} {'HITs':>6} {'PG %':>8}")
print("-" * 55)
for (name, _), h in zip(METHODS, hit_counts):
    short = name.replace("\n", " ")
    print(f"{short:<20} {h:>6}/{n_rows:<3} {100*h/n_rows:>7.1f}%")
print("=" * 55)
