"""
Generate qualitative figures for the GRAPE NeurIPS 2026 paper.

Figure A — Pointing Game grid (equivalent to paper Figure 6):
  Grid of N chest X-rays with:
    - Ground-truth bounding box (blue rectangle)
    - Max activation point for each variant (colored square)
    - Similarity map as heatmap overlay

Figure B — Similarity maps per concept (equivalent to paper Figure 1):
  For one TB image:
    - Input image
    - Similarity map for each of the 3 concepts (heatmap overlay)
    - Prototype image from atlas (most similar training patch)
    - Similarity scores bar chart

Figure C — Before/After interaction demo (equivalent to paper Figure 1 bottom):
  Same image, show how spatial bounding box changes the similarity map.

Output: ${OUT_DIR} (default: figures/)
"""

import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch.utils.data import DataLoader

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
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
CONCEPT_NAMES = ["Active TB", "Obsolete TB", "Pulmonary TB"]
CLASS_NAMES_DISPLAY = ["Healthy", "Sick (non-TB)", "Active TB"]

os.makedirs(OUT_DIR, exist_ok=True)

# Colormap: white → red for similarity maps
cmap_sim = LinearSegmentedColormap.from_list(
    "sim", ["#ffffff", "#ff6600", "#cc0000"], N=256
)
cmap_unc = LinearSegmentedColormap.from_list(
    "unc", ["#ffffff", "#0066ff", "#000099"], N=256
)

# ── Load model (best variant: no_C_novlm, highest PG) ────────────────────────
def load_model(name, use_gnn, use_vlm, use_uncertainty):
    ckpt_path = os.path.join(CKPT_BASE, f"tbx11k_{name}", "csr_stage4_best.pt")
    model = CSRModel(
        num_concepts=NUM_CONCEPTS,
        num_prototypes=100,
        num_classes=NUM_CLASSES,
        proto_dim=256,
        use_gnn=use_gnn,
        use_uncertainty=use_uncertainty,
        use_vlm=use_vlm,
        pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model

print("Loading models...")
model_grape    = load_model("no_C_novlm",   use_gnn=True,  use_vlm=False, use_uncertainty=True)
model_baseline = load_model("baseline_CSR", use_gnn=False, use_vlm=False, use_uncertainty=False)

# Build graph for GRAPE
print("Building graph...")
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
train_loader = get_dataloader(train_ds, batch_size=128, num_workers=4, shuffle=False)
all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
edge_index, edge_weight = build_cooccurrence_graph(all_labels, threshold=0.1)
edge_weight = normalize_edge_weights(edge_index, edge_weight, NUM_CONCEPTS)
model_grape.set_concept_graph(edge_index, edge_weight)

# ── Load bbox_eval dataset ────────────────────────────────────────────────────
print("Loading bbox_eval split...")
bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

# Find images with bboxes for each concept
samples_per_concept = {0: [], 1: [], 2: []}
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    for cidx in item["bbox"]:
        if len(samples_per_concept[cidx]) < 6:
            samples_per_concept[cidx].append(i)

# ── Helper: get similarity maps ───────────────────────────────────────────────
def get_maps(model, image_tensor):
    """Returns sim_maps (K, H, W) and uncertainty maps (K, H, W) at 224×224."""
    img = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img, return_maps=True)
    # sim_maps: (1, K, M, h, w) → max over M → (K, h, w)
    if "sim_maps" in out:
        maps = out["sim_maps"][0]          # (K, M, h, w)
        maps = maps.amax(dim=1).cpu()      # (K, h, w)
    else:
        maps = torch.zeros(NUM_CONCEPTS, 7, 7)

    # Upsample to 224×224
    maps_up = torch.nn.functional.interpolate(
        maps.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False
    )[0]  # (K, 224, 224)

    # Uncertainty maps if available
    if "uncertainty_maps" in out:
        unc = out["uncertainty_maps"][0].cpu()  # (K, h, w)
        unc_up = torch.nn.functional.interpolate(
            unc.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        )[0]
    else:
        unc_up = None

    return maps_up.numpy(), unc_up.numpy() if unc_up is not None else None

def get_max_point(sim_map):
    """Return (row, col) of maximum activation in a 2D map."""
    idx = np.argmax(sim_map)
    r, c = divmod(idx, sim_map.shape[1])
    return r, c

def tensor_to_rgb(t):
    """Convert CHW tensor [0,1] to HWC uint8 numpy."""
    img = t.permute(1, 2, 0).numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE A: Pointing Game Grid (main paper figure)
# Rows: selected images with Active TB bboxes (most common)
# Columns: [Image+bbox] [GRAPE sim map + max point] [Baseline sim map + max point]
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure A: Pointing Game Grid...")

concept_idx = 0   # Active TB — highest PG, most interesting
idxs = samples_per_concept[concept_idx][:6]

fig, axes = plt.subplots(len(idxs), 3, figsize=(10, 3.5 * len(idxs)))
fig.suptitle("Pointing Game: GRAPE vs. Prototype Baseline\n(Active Tuberculosis concept, blue box = GT annotation)",
             fontsize=12, fontweight="bold", y=1.01)

col_titles = ["Input + GT Box", "GRAPE (GNN+Uncert.)\nMax activation = ■", "Prototype Baseline\nMax activation = ■"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=9, pad=4)

for row, idx in enumerate(idxs):
    item = bbox_ds[idx]
    image_t = item["image"]
    rgb = tensor_to_rgb(image_t)
    bbox_dict = {int(k): v for k, v in item["bbox"].items()}
    gt_class = item["class_label"]

    # Get maps for both models
    maps_grape,    unc_grape    = get_maps(model_grape,    image_t)
    maps_baseline, _            = get_maps(model_baseline, image_t)

    sim_grape    = maps_grape[concept_idx]
    sim_baseline = maps_baseline[concept_idx]

    # Max activation points
    r_g, c_g = get_max_point(sim_grape)
    r_b, c_b = get_max_point(sim_baseline)

    # GT bbox (in 224×224 space)
    bbox = bbox_dict.get(concept_idx, None)

    # ── Col 0: Image + GT box ──
    ax = axes[row, 0]
    ax.imshow(rgb)
    if bbox:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="#2196F3", facecolor="none")
        ax.add_patch(rect)
    ax.set_ylabel(f"{'HIT' if bbox and (x1<=c_g<=x2 and y1<=r_g<=y2) else ''}", fontsize=8)
    ax.axis("off")

    # ── Col 1: GRAPE sim map + max point ──
    ax = axes[row, 1]
    ax.imshow(rgb)
    ax.imshow(sim_grape, cmap=cmap_sim, alpha=0.55, vmin=0, vmax=max(float(sim_grape.max()), 1e-6))
    if bbox:
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="#2196F3", facecolor="none")
        ax.add_patch(rect)
    # Max point as square
    inside = bbox and (x1 <= c_g <= x2 and y1 <= r_g <= y2)
    color = "#00cc00" if inside else "#ff0000"
    ax.add_patch(patches.Rectangle((c_g-4, r_g-4), 8, 8,
                                   linewidth=2, edgecolor="white", facecolor=color))
    ax.set_title("HIT ✓" if inside else "MISS ✗", fontsize=8, color=color)
    ax.axis("off")

    # ── Col 2: Baseline sim map + max point ──
    ax = axes[row, 2]
    ax.imshow(rgb)
    ax.imshow(sim_baseline, cmap=cmap_sim, alpha=0.55, vmin=0, vmax=max(float(sim_baseline.max()), 1e-6))
    if bbox:
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="#2196F3", facecolor="none")
        ax.add_patch(rect)
    inside_b = bbox and (x1 <= c_b <= x2 and y1 <= r_b <= y2)
    color_b = "#00cc00" if inside_b else "#ff0000"
    ax.add_patch(patches.Rectangle((c_b-4, r_b-4), 8, 8,
                                   linewidth=2, edgecolor="white", facecolor=color_b))
    ax.set_title("HIT ✓" if inside_b else "MISS ✗", fontsize=8, color=color_b)
    ax.axis("off")

plt.tight_layout()
fig_a_path = os.path.join(OUT_DIR, "figure_A_pointing_game.png")
plt.savefig(fig_a_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_a_path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE B: Per-concept similarity maps for one TB image
# Shows: Input | Concept 0 map | Concept 1 map | Concept 2 map | Uncertainty
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure B: Per-concept similarity maps...")

# Find a good Active TB image (class=2, has concept_0 bbox)
tb_idx = None
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    if item["class_label"] == 2 and 0 in {int(k) for k in item["bbox"]}:
        tb_idx = i
        break

if tb_idx is None:
    tb_idx = samples_per_concept[0][0]

item = bbox_ds[tb_idx]
image_t = item["image"]
rgb = tensor_to_rgb(image_t)
bbox_dict = {int(k): v for k, v in item["bbox"].items()}

maps_grape, unc_grape = get_maps(model_grape, image_t)

# Predict
with torch.no_grad():
    out = model_grape(image_t.unsqueeze(0).to(DEVICE), return_maps=True)
logits = out["logits"][0].cpu()
probs = torch.softmax(logits, dim=0).numpy()
pred_class = int(logits.argmax())

# Get similarity scores (K, M) → max over M
if "sim_scores" in out:
    scores = out["sim_scores"][0].cpu().numpy()  # (K, M)
    scores_max = scores.max(axis=1)              # (K,)
else:
    scores_max = np.array([maps_grape[k].max() for k in range(NUM_CONCEPTS)])

ncols = NUM_CONCEPTS + 2  # Input + K concepts + uncertainty
fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4))

# Col 0: Input + GT box
ax = axes[0]
ax.imshow(rgb)
for cidx, bbox in bbox_dict.items():
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                              linewidth=2, edgecolor="#2196F3", facecolor="none")
    ax.add_patch(rect)
ax.set_title(f"Input\nGT: {CLASS_NAMES_DISPLAY[item['class_label']]}\nPred: {CLASS_NAMES_DISPLAY[pred_class]}\n(p={probs[pred_class]:.2f})",
             fontsize=8)
ax.axis("off")

# Cols 1..(K): Similarity maps per concept
concept_colors = ["#ff4444", "#4444ff", "#44aa44"]
for k in range(NUM_CONCEPTS):
    ax = axes[k + 1]
    ax.imshow(rgb)
    sim = maps_grape[k]
    ax.imshow(sim, cmap=cmap_sim, alpha=0.6, vmin=0, vmax=max(float(sim.max()), 1e-6))
    # GT bbox for this concept if available
    if k in bbox_dict:
        x1, y1, x2, y2 = bbox_dict[k]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="#2196F3", facecolor="none")
        ax.add_patch(rect)
        # Max activation point
        r_max, c_max = get_max_point(sim)
        inside = x1 <= c_max <= x2 and y1 <= r_max <= y2
        col = "#00cc00" if inside else "#ff0000"
        ax.add_patch(patches.Rectangle((c_max-4, r_max-4), 8, 8,
                                       linewidth=2, edgecolor="white", facecolor=col))
    ax.set_title(f"{CONCEPT_NAMES[k]}\nsim={scores_max[k]:.3f}", fontsize=8)
    ax.axis("off")

# Last col: Uncertainty map (concept 0 — active TB)
ax = axes[-1]
ax.imshow(rgb)
if unc_grape is not None and float(unc_grape[0].max()) > 1e-8:
    unc = unc_grape[0]
    vmax = max(float(unc.max()), 1e-6)
    im = ax.imshow(unc, cmap=cmap_unc, alpha=0.6, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Uncertainty Map\n(Active TB)", fontsize=8)
else:
    ax.set_title("Uncertainty\n(not available)", fontsize=8)
ax.axis("off")

fig.suptitle("GRAPE: Per-Concept Similarity Maps and Uncertainty\n(TBX11K Active Tuberculosis image)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig_b_path = os.path.join(OUT_DIR, "figure_B_similarity_maps.png")
plt.savefig(fig_b_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_b_path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE C: Interaction demo — positive box changes similarity map
# Shows: Before interaction | Doctor draws +box | After interaction
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure C: Interaction demo...")

item = bbox_ds[tb_idx]
image_t = item["image"]
rgb = tensor_to_rgb(image_t)
bbox_dict = {int(k): v for k, v in item["bbox"].items()}

# Get maps before interaction
maps_before, unc_before = get_maps(model_grape, image_t)

# Simulate interaction: apply positive box for Active TB (concept 0)
# Use the GT bbox as the positive box
if 0 in bbox_dict:
    x1, y1, x2, y2 = bbox_dict[0]
else:
    x1, y1, x2, y2 = 56, 56, 168, 168  # center fallback

# Importance map: 1 inside box, alpha=0.2 outside (per paper Eq. 12, alpha=0.2)
alpha = 0.2
importance = torch.full((IMAGE_SIZE, IMAGE_SIZE), alpha)
importance[y1:y2, x1:x2] = 1.0

# Apply: multiply similarity map by importance
sim_before = maps_before[0].copy()
sim_after  = sim_before * importance.numpy()
# Renormalize to [0,1] for display
sim_after  = sim_after / (sim_after.max() + 1e-6) * sim_before.max()

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

panels = [
    ("Before Interaction\n(Active TB similarity)", sim_before, None),
    ("Doctor Feedback\n(positive box on lesion)", sim_before, (x1, y1, x2, y2)),
    ("After Interaction\n(importance-weighted)", sim_after, (x1, y1, x2, y2)),
]

for col, (title, sim, box) in enumerate(panels):
    ax = axes[col]
    ax.imshow(rgb)
    ax.imshow(sim, cmap=cmap_sim, alpha=0.6, vmin=0, vmax=maps_before[0].max())
    if box:
        bx1, by1, bx2, by2 = box
        rect = patches.Rectangle((bx1, by1), bx2-bx1, by2-by1,
                                  linewidth=2.5,
                                  edgecolor="#2196F3" if col == 1 else "#2196F3",
                                  facecolor="none",
                                  linestyle="--" if col == 1 else "-")
        ax.add_patch(rect)
        if col == 1:
            ax.text(bx1, by1 - 6, "bb⁺", fontsize=9, color="#2196F3", fontweight="bold")
    r_max, c_max = get_max_point(sim)
    ax.add_patch(patches.Rectangle((c_max-4, r_max-4), 8, 8,
                                   linewidth=2, edgecolor="white", facecolor="#ff6600"))
    ax.set_title(title, fontsize=9, fontweight="bold" if col == 2 else "normal")
    ax.axis("off")

# Arrows between panels
fig.text(0.35, 0.5, "→", fontsize=24, ha="center", va="center", color="#333")
fig.text(0.67, 0.5, "→", fontsize=24, ha="center", va="center", color="#333")

fig.suptitle("GRAPE Spatial Interaction Demo: Doctor draws positive bounding box\nto guide model attention toward the TB lesion region",
             fontsize=10, fontweight="bold")
plt.tight_layout()
fig_c_path = os.path.join(OUT_DIR, "figure_C_interaction.png")
plt.savefig(fig_c_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_c_path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE D: Multi-class decision overview
# Shows 3 images (one per class) × 3 concept maps side by side
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure D: Multi-class overview...")

# Find one image per class from bbox_eval
class_samples = {0: None, 1: None, 2: None}
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    c = item["class_label"]
    if class_samples[int(c)] is None:
        class_samples[int(c)] = i
    if all(v is not None for v in class_samples.values()):
        break

fig, axes = plt.subplots(3, NUM_CONCEPTS + 1, figsize=(4*(NUM_CONCEPTS+1), 4*3))

row_labels = ["Healthy", "Sick (non-TB)", "Active TB"]
for row, (cls, idx) in enumerate(class_samples.items()):
    if idx is None:
        continue
    item = bbox_ds[idx]
    image_t = item["image"]
    rgb = tensor_to_rgb(image_t)
    maps, _ = get_maps(model_grape, image_t)

    with torch.no_grad():
        out = model_grape(image_t.unsqueeze(0).to(DEVICE))
    probs = torch.softmax(out["logits"][0].cpu(), dim=0).numpy()
    pred = int(out["logits"][0].argmax())

    # Col 0: input
    ax = axes[row, 0]
    ax.imshow(rgb)
    ax.set_ylabel(row_labels[cls], fontsize=10, fontweight="bold")
    ax.set_title(f"GT: {row_labels[cls]}\nPred: {row_labels[pred]} (p={probs[pred]:.2f})",
                 fontsize=8)
    ax.axis("off")

    for k in range(NUM_CONCEPTS):
        ax = axes[row, k+1]
        ax.imshow(rgb)
        sim = maps[k]
        ax.imshow(sim, cmap=cmap_sim, alpha=0.55, vmin=0, vmax=sim.max()+1e-6)
        if row == 0:
            ax.set_title(CONCEPT_NAMES[k], fontsize=9, fontweight="bold")
        ax.axis("off")

plt.suptitle("GRAPE: Concept similarity maps across three TBX11K classes\n(columns: per-concept activation; brighter = more similar to prototype)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig_d_path = os.path.join(OUT_DIR, "figure_D_multiclass.png")
plt.savefig(fig_d_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_d_path}")

print(f"\nAll figures saved to {OUT_DIR}/")
print("  figure_A_pointing_game.png  — Pointing Game grid (paper Figure 6 equivalent)")
print("  figure_B_similarity_maps.png — Per-concept maps + uncertainty")
print("  figure_C_interaction.png     — Before/after spatial interaction")
print("  figure_D_multiclass.png      — Multi-class decision overview")
