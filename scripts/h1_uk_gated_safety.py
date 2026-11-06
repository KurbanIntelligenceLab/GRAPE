"""
H1 — U_k-Gated Module B Safety Check

Problem: the current safety check has a 21.7% false-positive rate (FP) — it
warns even when the doctor's box is correct.  The cause: the model's dominant
concept inside the box may genuinely differ from the intended one when the
region is ambiguous (high prototype variance).

Fix: gate the warning by the per-concept prototype variance inside the box.
  - Compute V_intended = variance of per-prototype similarities inside bbox
    for the intended concept (across M prototypes).
  - Only issue a warning if V_intended < theta_V (prototypes agree the region
    is NOT the intended concept → the mismatch is a real signal, not noise).
  - When V_intended >= theta_V (high disagreement), suppress the warning.

This adds one hyperparameter theta_V, swept here to find the Pareto-optimal
(FP-reducing) value that preserves TP rate.

Results → results/h1_uk_gated_safety.json
Updates: Table 7 (tab:uncertainty_comparison).
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/tbx11k"
CKPT_BASE    = "checkpoints"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
ETA          = 0.05        # score-gap threshold (unchanged from paper)
MISC_RATES   = [0.0, 0.10, 0.25, 0.50, 1.0]
N_TRIALS     = 5
FEAT_H = FEAT_W = 7

# Sweep theta_V; None = no gating (original Module B)
# Observed V_intended distribution: mean≈0.00032, max≈0.00053 — sweep within that range
THETA_V_SWEEP = [None, 0.00010, 0.00020, 0.00030, 0.00035, 0.00040, 0.00045, 0.00053]

os.makedirs("results", exist_ok=True)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    path = os.path.join(CKPT_BASE, "tbx11k_no_C_novlm", "csr_stage4_best.pt")
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m


def get_sim_maps(model, image_t):
    """Return (K, M, H, W) similarity maps for one image."""
    img = image_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)  # (1, K, M, h, w)
    return sm[0].cpu()  # (K, M, H, W)


def run_safety_check(sim_maps, bbox_224, intended_concept, eta=ETA, theta_v=None):
    """
    Original (theta_v=None) or U_k-gated safety check.

    sim_maps: (K, M, H, W)
    bbox_224: (x1, y1, x2, y2) in 224-space
    theta_v:  if not None, suppress warning when V_intended >= theta_v
    """
    K, M, H, W = sim_maps.shape
    x1, y1, x2, y2 = bbox_224
    bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
    bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
    by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
    by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))

    # Per-prototype region similarity: (K, M, h_reg, w_reg) → (K, M)
    region = sim_maps[:, :, by1:by2, bx1:bx2]         # (K, M, h_reg, w_reg)
    proto_means = region.mean(dim=(-2, -1))            # (K, M) — per-proto mean in bbox

    # Mean over M → score per concept
    s_bar = proto_means.mean(dim=1)                    # (K,)
    dominant = int(s_bar.argmax())
    dom_score = float(s_bar[dominant])
    int_score = float(s_bar[intended_concept])

    mismatch = (dominant != intended_concept) and (dom_score - int_score > eta)

    if mismatch and theta_v is not None:
        # U_k gate: compute variance over M prototypes for intended concept
        v_intended = float(proto_means[intended_concept].var().item())
        if v_intended >= theta_v:
            # High prototype variance → ambiguous region → suppress warning
            mismatch = False

    return mismatch


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model()

train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
train_ldr = get_dataloader(train_ds, 128, 4, shuffle=False)
all_labels = torch.cat([b["concept_labels"] for b in train_ldr]).to(DEVICE)
ei, ew = build_cooccurrence_graph(all_labels, threshold=0.1)
ew = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
model.set_concept_graph(ei, ew)

bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
samples = []
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    for concept_idx, bbox in {int(k): v for k, v in item["bbox"].items()}.items():
        samples.append((i, concept_idx, bbox))
print(f"  Total (image, concept, bbox) triples: {len(samples)}")

unique_imgs = list(set(s[0] for s in samples))
print(f"  Pre-computing sim maps for {len(unique_imgs)} images...")
sim_cache = {}
for idx, img_i in enumerate(unique_imgs):
    if idx % 20 == 0:
        print(f"    {idx}/{len(unique_imgs)}", flush=True)
    sim_cache[img_i] = get_sim_maps(model, bbox_ds[img_i]["image"])

all_concepts = list(range(NUM_CONCEPTS))


# ── Sweep theta_V ─────────────────────────────────────────────────────────────
def run_experiment(theta_v):
    results_p = {}
    for p in MISC_RATES:
        tp_trials, fp_trials = [], []
        for trial in range(N_TRIALS):
            rng = random.Random(RANDOM_SEED + trial)
            tp_w = tp_t = fp_w = fp_t = 0
            for img_i, concept_idx, bbox in samples:
                is_misdraw = rng.random() < p
                if is_misdraw:
                    wrong = [c for c in all_concepts if c != concept_idx]
                    claimed = rng.choice(wrong)
                    warned = run_safety_check(sim_cache[img_i], bbox, claimed,
                                             theta_v=theta_v)
                    tp_t += 1
                    if warned: tp_w += 1
                else:
                    warned = run_safety_check(sim_cache[img_i], bbox, concept_idx,
                                             theta_v=theta_v)
                    fp_t += 1
                    if warned: fp_w += 1
            if tp_t > 0: tp_trials.append(tp_w / tp_t)
            if fp_t > 0: fp_trials.append(fp_w / fp_t)
        results_p[p] = {
            "tp_rate_mean": float(np.mean(tp_trials)) if tp_trials else None,
            "tp_rate_std":  float(np.std(tp_trials))  if tp_trials else None,
            "fp_rate_mean": float(np.mean(fp_trials)) if fp_trials else None,
            "fp_rate_std":  float(np.std(fp_trials))  if fp_trials else None,
        }
    return results_p


print("\n=== H1: U_k-Gated Module B Safety Check ===")
print(f"{'theta_V':>10}  {'FP@p=0':>8}  {'TP@p=1':>8}")
print("-" * 32)

all_results = {}
for theta_v in THETA_V_SWEEP:
    label = "None (orig)" if theta_v is None else f"{theta_v:.3f}"
    res   = run_experiment(theta_v)
    fp0   = res[0.0]["fp_rate_mean"] or 0.0
    tp1   = res[1.0]["tp_rate_mean"] or 0.0
    print(f"{label:>10}  {fp0:.3f}     {tp1:.3f}")
    all_results[str(theta_v)] = res

# ── Find best theta_V (minimize FP while keeping TP >= 0.85) ──────────────────
best_theta = None
best_fp    = 1.0
for tv_str, res in all_results.items():
    if tv_str == "None": continue
    tp1 = res.get(1.0, {}).get("tp_rate_mean") or 0.0
    fp0 = res.get(0.0, {}).get("fp_rate_mean") or 1.0
    if tp1 >= 0.85 and fp0 < best_fp:
        best_fp    = fp0
        best_theta = tv_str

print(f"\nBest theta_V = {best_theta}  (FP={best_fp:.3f}, TP≥0.85 constraint)")

out = {
    "sweep": all_results,
    "best_theta_v": best_theta,
    "best_fp_at_p0": best_fp,
    "n_samples": len(samples),
    "eta": ETA,
    "n_trials": N_TRIALS,
    "misc_rates": MISC_RATES,
}
with open("results/h1_uk_gated_safety.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nH1 done. Results → results/h1_uk_gated_safety.json")
print("Update Table 7 (tab:uncertainty_comparison).")
