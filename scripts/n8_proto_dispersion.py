"""
N8 — Prototype–Anchor Cosine Dispersion Analysis (Figure 4)

For each dataset (TBX11K K=3 and NIH K=14), we:
  1. Load the Stage-3 checkpoint (after VLM alignment training)
  2. Extract all K×M prototype vectors
  3. Compute cosine similarity of each prototype to its text anchor
  4. Compute pairwise inter-prototype cosine within each concept
  5. Plot side-by-side histograms → figures/proto_dispersion.png

Saves: figures/proto_dispersion.png
       results/n8_proto_dispersion.json

Checkpoints are read from ${CKPT_BASE} (default: checkpoints).
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.csr_baseline import CSRModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_BASE = os.environ.get("CKPT_BASE", "checkpoints")

CONFIGS = {
    "TBX11K (K=3)": {
        "ckpt":        os.path.join(CKPT_BASE, "tbx11k_gnn_vlm", "csr_stage3.pt"),
        "num_concepts": 3,
        "num_classes":  3,
        "concept_names": ["Active TB", "Obsolete TB", "Pulmonary TB"],
        "color":        "#E64A19",
    },
    "NIH (K=14)": {
        "ckpt":        os.path.join(CKPT_BASE, "nih_official_grape_seed0", "csr_stage3.pt"),
        "num_concepts": 14,
        "num_classes":  2,
        "concept_names": [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia",
            "Infiltration", "Mass", "Nodule", "Pleural Thickening",
            "Pneumonia", "Pneumothorax",
        ],
        "color":        "#1565C0",
    },
}

NUM_PROTO  = 100
PROTO_DIM  = 256

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_prototypes(cfg):
    """Returns prototype tensor (K, M, D) and text anchors (K, D)."""
    import torch.nn.functional as F
    K = cfg["num_concepts"]
    m = CSRModel(
        num_concepts=K, num_prototypes=NUM_PROTO,
        num_classes=cfg["num_classes"], proto_dim=PROTO_DIM,
        use_gnn=False, use_uncertainty=False, use_vlm=True,
        pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(cfg["ckpt"], map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()

    # Manually restore None buffers that load_state_dict skips
    state = ckpt["model_state"]
    if state.get("vlm_aligner.text_embeddings") is not None:
        m.vlm_aligner.text_embeddings = state["vlm_aligner.text_embeddings"].to(DEVICE)

    with torch.no_grad():
        protos = m.prototype_learner.prototypes.data   # (K, M, D)
        if protos.dim() == 2:
            protos = protos.view(K, NUM_PROTO, PROTO_DIM)
        # Explicit L2-normalize to guard against small floating-point drift
        protos = F.normalize(protos.float(), p=2, dim=-1)

        t_emb   = m.vlm_aligner.text_embeddings.float()                   # (K, 768)
        # Project: (K, 768) @ W^T where W is (D, 768)
        w = m.vlm_aligner.text_projection.projection.weight.float()       # (D, 768)
        anchors = F.normalize(t_emb @ w.T, p=2, dim=-1)                   # (K, D)

    return protos.cpu(), anchors.cpu()


def proto_anchor_cosines(protos, anchors):
    """For each concept k, cosine similarity of each prototype to its anchor.
    Returns flat array of K*M cosines."""
    K, M, D = protos.shape
    cosines  = []
    for k in range(K):
        p_k = protos[k]   # (M, D)  — already unit-norm from training
        a_k = anchors[k]  # (D,)
        a_k = a_k / (a_k.norm() + 1e-8)
        cos = (p_k @ a_k).numpy()   # (M,)
        cosines.append(cos)
    return np.concatenate(cosines)


def inter_proto_cosines(protos):
    """Mean pairwise cosine within each concept, pooled across concepts."""
    K, M, D = protos.shape
    all_pairs = []
    for k in range(K):
        p = protos[k].float()         # (M, D)
        # cosine matrix (M, M)
        gram = (p @ p.T).numpy()
        # upper triangle excluding diagonal
        idx = np.triu_indices(M, k=1)
        all_pairs.append(gram[idx])
    return np.concatenate(all_pairs)


results_json = {}
all_anchor_cos, all_inter_cos, all_labels, all_colors = [], [], [], []

print("Extracting prototype statistics...")
for name, cfg in CONFIGS.items():
    if not os.path.exists(cfg["ckpt"]):
        print(f"  WARNING: checkpoint not found for {name}, skipping.")
        continue
    print(f"  Loading {name}...")
    protos, anchors = load_prototypes(cfg)
    anchor_cos  = proto_anchor_cosines(protos, anchors)
    inter_cos   = inter_proto_cosines(protos)
    results_json[name] = {
        "anchor_cos_mean":  float(anchor_cos.mean()),
        "anchor_cos_std":   float(anchor_cos.std()),
        "inter_cos_mean":   float(inter_cos.mean()),
        "inter_cos_std":    float(inter_cos.std()),
        "n_concepts": cfg["num_concepts"],
        "n_proto":    NUM_PROTO,
    }
    all_anchor_cos.append(anchor_cos)
    all_inter_cos.append(inter_cos)
    all_labels.append(name)
    all_colors.append(cfg["color"])
    print(f"    anchor cos: {anchor_cos.mean():.3f}±{anchor_cos.std():.3f}")
    print(f"    inter  cos: {inter_cos.mean():.3f}±{inter_cos.std():.3f}")


# ── Plot ──────────────────────────────────────────────────────────────────────
if all_anchor_cos:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), facecolor="white")
    for ax, data_list, title, xlabel in [
        (axes[0], all_anchor_cos, "Prototype–Anchor Alignment",
         "Cosine similarity to text anchor"),
        (axes[1], all_inter_cos,  "Inter-Prototype Dispersion",
         "Pairwise intra-concept cosine"),
    ]:
        # Zoom to the region where data lives
        all_vals = np.concatenate(data_list)
        lo = max(all_vals.min() - 0.02, -1.0)
        hi = min(all_vals.max() + 0.01,  1.0)
        bins = np.linspace(lo, hi, 60)

        for data, label, color in zip(data_list, all_labels, all_colors):
            ax.hist(data, bins=bins, alpha=0.65, color=color, label=label,
                    density=True, edgecolor="none")
            ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.5,
                       label=f"{label} mean={data.mean():.3f}")

        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=1.2)
    out = "figures/proto_dispersion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {out}")

with open("results/n8_proto_dispersion.json", "w") as f:
    json.dump(results_json, f, indent=2)
print("Saved → results/n8_proto_dispersion.json")
