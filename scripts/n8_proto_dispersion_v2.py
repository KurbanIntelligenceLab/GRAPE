"""
N8 v2 -- Prototype-Anchor Cosine Dispersion Analysis, real checkpoints,
extended with the M-ablation checkpoints (M in {1,5,25,100}) so the figure
shows collapse is invariant to prototype count directly, not just in a table.

For TBX11K (K=3) and NIH (K=14) main checkpoints, and TBX11K's M-ablation
checkpoints:
  1. Load the Stage-3 checkpoint (after VLM alignment training)
  2. Extract all K x M prototype vectors
  3. Compute cosine similarity of each prototype to its text anchor
  4. Compute pairwise inter-prototype cosine within each concept
  5. Plot histograms -> figures/proto_dispersion_v2.png

Usage:
  .venv/bin/python3 scripts/n8_proto_dispersion_v2.py
"""

import os, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.csr_baseline import CSRModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROTO_DIM = 256

MAIN_CONFIGS = {
    "TBX11K (K=3, M=100)": {
        "ckpt": "checkpoints/tbx11k_seed42/csr_stage3.pt",
        "num_concepts": 3, "num_classes": 3, "num_proto": 100,
        "color": "#E64A19",
    },
    "NIH (K=14, M=100)": {
        "ckpt": "checkpoints/nih_seed42/csr_stage3.pt",
        "num_concepts": 14, "num_classes": 2, "num_proto": 100,
        "color": "#1565C0",
    },
}

# M-ablation checkpoints (TBX11K, seed 42) trained this session for D3/D5.
M_CONFIGS = {
    "TBX11K, M=1":  {"ckpt": "checkpoints/tbx11k_m1_seed42/csr_stage3.pt",  "num_proto": 1},
    "TBX11K, M=5":  {"ckpt": "checkpoints/tbx11k_m5_seed42/csr_stage3.pt",  "num_proto": 5},
    "TBX11K, M=25": {"ckpt": "checkpoints/tbx11k_m25_seed42/csr_stage3.pt", "num_proto": 25},
}
M_COLORS = {"TBX11K, M=1": "#9E9E9E", "TBX11K, M=5": "#FFA000",
            "TBX11K, M=25": "#7B1FA2", "TBX11K, M=100": "#E64A19"}

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_prototypes(ckpt_path, num_concepts, num_proto, num_classes=3, use_vlm=True):
    m = CSRModel(
        num_concepts=num_concepts, num_prototypes=num_proto,
        num_classes=num_classes, proto_dim=PROTO_DIM,
        use_gnn=False, use_uncertainty=False, use_vlm=use_vlm,
        pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()

    state = ckpt["model_state"]
    if use_vlm and state.get("vlm_aligner.text_embeddings") is not None:
        m.vlm_aligner.text_embeddings = state["vlm_aligner.text_embeddings"].to(DEVICE)

    with torch.no_grad():
        protos = m.prototype_learner.prototypes.data
        if protos.dim() == 2:
            protos = protos.view(num_concepts, num_proto, PROTO_DIM)
        protos = F.normalize(protos.float(), p=2, dim=-1)

        anchors = None
        if use_vlm and hasattr(m, "vlm_aligner") and m.vlm_aligner is not None:
            t_emb = m.vlm_aligner.text_embeddings.float()
            w = m.vlm_aligner.text_projection.projection.weight.float()
            anchors = F.normalize(t_emb @ w.T, p=2, dim=-1)

    return protos.cpu(), (anchors.cpu() if anchors is not None else None)


def proto_anchor_cosines(protos, anchors):
    K, M, D = protos.shape
    cosines = []
    for k in range(K):
        p_k = protos[k]
        a_k = anchors[k]
        a_k = a_k / (a_k.norm() + 1e-8)
        cosines.append((p_k @ a_k).numpy())
    return np.concatenate(cosines)


def inter_proto_cosines(protos):
    K, M, D = protos.shape
    all_pairs = []
    for k in range(K):
        if M < 2:
            continue
        p = protos[k].float()
        gram = (p @ p.T).numpy()
        idx = np.triu_indices(M, k=1)
        all_pairs.append(gram[idx])
    return np.concatenate(all_pairs) if all_pairs else np.array([])


results_json = {}

print("=== Main checkpoints (TBX11K M=100, NIH M=100) ===")
main_anchor, main_inter, main_labels, main_colors = [], [], [], []
for name, cfg in MAIN_CONFIGS.items():
    if not os.path.exists(cfg["ckpt"]):
        print(f"  WARNING: {cfg['ckpt']} not found, skipping {name}")
        continue
    print(f"  Loading {name}...")
    protos, anchors = load_prototypes(cfg["ckpt"], cfg["num_concepts"], cfg["num_proto"],
                                       cfg["num_classes"], use_vlm=True)
    anchor_cos = proto_anchor_cosines(protos, anchors)
    inter_cos = inter_proto_cosines(protos)
    results_json[name] = {
        "anchor_cos_mean": float(anchor_cos.mean()), "anchor_cos_std": float(anchor_cos.std()),
        "inter_cos_mean": float(inter_cos.mean()), "inter_cos_std": float(inter_cos.std()),
        "n_concepts": cfg["num_concepts"], "n_proto": cfg["num_proto"],
    }
    main_anchor.append(anchor_cos); main_inter.append(inter_cos)
    main_labels.append(name); main_colors.append(cfg["color"])
    print(f"    anchor cos: {anchor_cos.mean():.4f}+-{anchor_cos.std():.4f}")
    print(f"    inter  cos: {inter_cos.mean():.4f}+-{inter_cos.std():.4f}")

print("\n=== M-ablation checkpoints (TBX11K, seed 42) ===")
m_inter, m_labels, m_colors = [], [], []
for name, cfg in M_CONFIGS.items():
    if not os.path.exists(cfg["ckpt"]):
        print(f"  WARNING: {cfg['ckpt']} not found, skipping {name}")
        continue
    print(f"  Loading {name}...")
    protos, _ = load_prototypes(cfg["ckpt"], 3, cfg["num_proto"], 3, use_vlm=True)
    inter_cos = inter_proto_cosines(protos)
    if len(inter_cos) == 0:
        print(f"    M={cfg['num_proto']}: no pairwise cosine (single prototype)")
        continue
    results_json[name] = {
        "inter_cos_mean": float(inter_cos.mean()), "inter_cos_std": float(inter_cos.std()),
        "n_proto": cfg["num_proto"],
    }
    m_inter.append(inter_cos); m_labels.append(name); m_colors.append(M_COLORS[name])
    print(f"    inter cos: {inter_cos.mean():.4f}+-{inter_cos.std():.4f}")

# Add TBX11K M=100 to the M-ablation panel for comparison
if "TBX11K (K=3, M=100)" in results_json:
    m_inter.append(main_inter[main_labels.index("TBX11K (K=3, M=100)")])
    m_labels.append("TBX11K, M=100")
    m_colors.append(M_COLORS["TBX11K, M=100"])

# ---- Plot: 3 panels (anchor alignment, inter-proto dispersion main, M-ablation) ----
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), facecolor="white")

# Panel 1: anchor alignment (TBX11K vs NIH, M=100)
ax = axes[0]
all_vals = np.concatenate(main_anchor)
lo, hi = max(all_vals.min() - 0.02, -1.0), min(all_vals.max() + 0.01, 1.0)
bins = np.linspace(lo, hi, 60)
for data, label, color in zip(main_anchor, main_labels, main_colors):
    ax.hist(data, bins=bins, alpha=0.65, color=color, label=label, density=True, edgecolor="none")
    ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.5)
ax.set_xlim(lo, hi)
ax.set_xlabel("Cosine similarity to text anchor", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Prototype-Anchor Alignment", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

# Panel 2: inter-prototype dispersion (TBX11K vs NIH, M=100)
ax = axes[1]
all_vals = np.concatenate(main_inter)
lo, hi = max(all_vals.min() - 0.02, -1.0), min(all_vals.max() + 0.01, 1.0)
bins = np.linspace(lo, hi, 60)
for data, label, color in zip(main_inter, main_labels, main_colors):
    ax.hist(data, bins=bins, alpha=0.65, color=color, label=label, density=True, edgecolor="none")
    ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.5)
ax.set_xlim(lo, hi)
ax.set_xlabel("Pairwise intra-concept cosine", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Inter-Prototype Dispersion", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

# Panel 3: M-ablation (TBX11K, M in {5,25,100})
ax = axes[2]
if m_inter:
    all_vals = np.concatenate(m_inter)
    lo, hi = max(all_vals.min() - 0.02, -1.0), min(all_vals.max() + 0.01, 1.0)
    bins = np.linspace(lo, hi, 60)
    for data, label, color in zip(m_inter, m_labels, m_colors):
        ax.hist(data, bins=bins, alpha=0.55, color=color, label=label, density=True, edgecolor="none")
        ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlim(lo, hi)
ax.set_xlabel("Pairwise intra-concept cosine", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Dispersion vs. Prototype Count M\n(TBX11K, seed 42)", fontsize=10, fontweight="bold")
ax.legend(fontsize=7, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=1.3)
out = "figures/proto_dispersion_v2.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved -> {out}")

with open("results/n8_proto_dispersion_v2.json", "w") as f:
    json.dump(results_json, f, indent=2)
print("Saved -> results/n8_proto_dispersion_v2.json")
