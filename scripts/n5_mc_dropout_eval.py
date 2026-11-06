"""
N5 — MC-Dropout Uncertainty Baseline vs Prototype Variance (Module B)

Compares three safety-check strategies on TBX11K bbox_eval:
  (1) Prototype Variance (GRAPE Module B) — disagreement among M=100 prototypes
  (2) MC-Dropout Score Variance (Variant A) — variance of sim scores over T=30 passes
  (3) MC-Dropout Prediction Entropy (Variant B) — predictive entropy over T=30 passes

Protocol (same as C7 safety eval):
  - For each bbox triple (image, concept, box):
    - With prob p=0.5: randomly reassign to wrong concept (TP trial)
    - With prob 1-p=0.5: keep correct concept (FP trial)
  - Report TP rate (misdraw caught) and FP rate (correct draw warned)
  - T=5 random seeds, mean ± std reported

The threshold η for each method is calibrated to match the FP rate of the
prototype-variance baseline (~21.7%), for a fair TP comparison.

Results → Table N5 in paper comparing uncertainty methods.
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.mc_dropout import (
    mc_forward_passes, prototype_variance_safety_check,
    mc_dropout_score_variance_check, mc_dropout_entropy_check,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/tbx11k"
CKPT_PATH   = "checkpoints/tbx11k_no_C_novlm/csr_stage4_best.pt"
RESULTS_PATH = "results/n5_mc_dropout_results.json"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
T_PASSES    = 30       # MC-Dropout forward passes
DROPOUT_P   = 0.1
ETA_PROTO   = 0.05     # Prototype variance threshold (same as C7)
N_TRIALS    = 5
MISC_RATE_P = 0.5      # miscorrection probability for TP measurement
RANDOM_SEED = 42

os.makedirs("results", exist_ok=True)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GRAPE model (no VLM)...")
model = CSRModel(
    num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
    proto_dim=256, use_gnn=True, use_uncertainty=True,
    use_vlm=False, pretrained_backbone=False,
).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# ── Build and register concept graph ─────────────────────────────────────────
print("Building concept graph...")
_train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
_train_loader = get_dataloader(_train_ds, 128, 4, shuffle=False)
_all_labels = torch.cat([b["concept_labels"] for b in _train_loader]).to(DEVICE)
_ei, _ew = build_cooccurrence_graph(_all_labels, threshold=0.10)
_ew_norm = normalize_edge_weights(_ei, _ew, NUM_CONCEPTS)
model.set_concept_graph(_ei, _ew_norm)
del _train_ds, _train_loader, _all_labels

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading bbox_eval dataset...")
bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
all_concepts = list(range(NUM_CONCEPTS))

# Build evaluation triples: (image_idx, concept_idx, bbox)
samples = []
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    for c_idx, bbox in {int(k): v for k, v in item["bbox"].items()}.items():
        samples.append((i, c_idx, bbox))
print(f"  {len(samples)} (image, concept, bbox) triples from {len(bbox_ds)} images")

# ── Pre-compute prototype sim maps (for prototype-variance check) ─────────────
print(f"Pre-computing prototype sim maps for {len(set(s[0] for s in samples))} images...")
sim_cache = {}
mc_cache  = {}

with torch.no_grad():
    for img_i in set(s[0] for s in samples):
        item = bbox_ds[img_i]
        img  = item["image"].unsqueeze(0).to(DEVICE)
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()  # (K,M,H,W)
        sim_cache[img_i] = sm

print(f"Pre-computing MC-Dropout passes (T={T_PASSES}) for {len(sim_cache)} images...")
for img_i in sorted(sim_cache.keys()):
    item = bbox_ds[img_i]
    img  = item["image"].unsqueeze(0).to(DEVICE)
    mc_cache[img_i] = mc_forward_passes(model, img, T=T_PASSES, dropout_p=DROPOUT_P)

print("Pre-computation complete.")


# ── Calibrate MC-Dropout thresholds to match FP rate ─────────────────────────
# Run prototype check at p=0 (all correct draws) to get baseline FP rate
# Then sweep MC-Dropout thresholds to find the closest match.

def run_all_methods(p: float, seed: int):
    """Run all three methods and collect TP/FP rates for this trial."""
    rng = random.Random(seed)
    tp_proto, fp_proto = 0, 0
    tp_mc_var, fp_mc_var = 0, 0
    tp_mc_ent, fp_mc_ent = 0, 0
    tp_total, fp_total = 0, 0

    for img_i, concept_idx, bbox in samples:
        sm = sim_cache[img_i]             # (K, M, H, W)
        mc = mc_cache[img_i]

        is_misdraw = rng.random() < p
        if is_misdraw:
            wrong = [c for c in all_concepts if c != concept_idx]
            claimed = rng.choice(wrong)
            tp_total += 1
            # Method 1: prototype variance
            if prototype_variance_safety_check(sm, bbox, concept_idx, claimed,
                                               IMAGE_SIZE, ETA_PROTO):
                tp_proto += 1
            # Method 2: MC-Dropout score variance (η_var calibrated below)
            if mc_dropout_score_variance_check(mc, bbox, concept_idx, claimed,
                                               IMAGE_SIZE, _ETA_VAR):
                tp_mc_var += 1
            # Method 3: MC-Dropout entropy (η_ent calibrated below)
            if mc_dropout_entropy_check(mc, _ETA_ENT):
                tp_mc_ent += 1
        else:
            claimed = concept_idx
            fp_total += 1
            if prototype_variance_safety_check(sm, bbox, concept_idx, claimed,
                                               IMAGE_SIZE, ETA_PROTO):
                fp_proto += 1
            if mc_dropout_score_variance_check(mc, bbox, concept_idx, claimed,
                                               IMAGE_SIZE, _ETA_VAR):
                fp_mc_var += 1
            if mc_dropout_entropy_check(mc, _ETA_ENT):
                fp_mc_ent += 1

    return {
        "proto":  {"tp": tp_proto / tp_total if tp_total else 0,
                   "fp": fp_proto / fp_total if fp_total else 0},
        "mc_var": {"tp": tp_mc_var / tp_total if tp_total else 0,
                   "fp": fp_mc_var / fp_total if fp_total else 0},
        "mc_ent": {"tp": tp_mc_ent / tp_total if tp_total else 0,
                   "fp": fp_mc_ent / fp_total if fp_total else 0},
    }


# Calibrate: sweep η_var and η_ent to match prototype FP rate at p=0
print("\nCalibrating MC-Dropout thresholds...")
_ETA_VAR = 0.01   # placeholder — will be updated
_ETA_ENT = 0.5    # placeholder — will be updated

# Measure prototype FP rate at p=0
fp_rates_proto = []
for trial in range(N_TRIALS):
    r = run_all_methods(p=0.0, seed=RANDOM_SEED + trial)
    fp_rates_proto.append(r["proto"]["fp"])
target_fp = float(np.mean(fp_rates_proto))
print(f"  Prototype baseline FP rate: {target_fp:.4f}")

# Sweep η_var
best_eta_var, best_var_diff = 0.01, 1e9
for eta_var_cand in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    _ETA_VAR = eta_var_cand
    fps = []
    for trial in range(N_TRIALS):
        r = run_all_methods(p=0.0, seed=RANDOM_SEED + trial)
        fps.append(r["mc_var"]["fp"])
    diff = abs(float(np.mean(fps)) - target_fp)
    if diff < best_var_diff:
        best_var_diff = diff
        best_eta_var = eta_var_cand
print(f"  Calibrated η_var = {best_eta_var}  (FP diff from target: {best_var_diff:.4f})")
_ETA_VAR = best_eta_var

# Sweep η_ent
best_eta_ent, best_ent_diff = 0.5, 1e9
for eta_ent_cand in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    _ETA_ENT = eta_ent_cand
    fps = []
    for trial in range(N_TRIALS):
        r = run_all_methods(p=0.0, seed=RANDOM_SEED + trial)
        fps.append(r["mc_ent"]["fp"])
    diff = abs(float(np.mean(fps)) - target_fp)
    if diff < best_ent_diff:
        best_ent_diff = diff
        best_eta_ent = eta_ent_cand
print(f"  Calibrated η_ent = {best_eta_ent}  (FP diff from target: {best_ent_diff:.4f})")
_ETA_ENT = best_eta_ent

# ── Main evaluation at p=0.5 ─────────────────────────────────────────────────
print(f"\nEvaluating at p={MISC_RATE_P} (TP) and p=0 (FP), {N_TRIALS} trials...")
tp_results = {"proto": [], "mc_var": [], "mc_ent": []}
fp_results = {"proto": [], "mc_var": [], "mc_ent": []}

for trial in range(N_TRIALS):
    r_tp = run_all_methods(p=MISC_RATE_P, seed=RANDOM_SEED + trial)
    r_fp = run_all_methods(p=0.0,         seed=RANDOM_SEED + trial)
    for method in ["proto", "mc_var", "mc_ent"]:
        tp_results[method].append(r_tp[method]["tp"])
        fp_results[method].append(r_fp[method]["fp"])

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"N5 Results: Uncertainty Method Comparison (TBX11K, {len(samples)} pairs)")
print(f"{'='*65}")
print(f"{'Method':>30}  {'TP rate':>14}  {'FP rate':>14}")
print("-" * 65)

METHOD_NAMES = {
    "proto":  "Prototype variance (GRAPE)",
    "mc_var": f"MC-Dropout score var (T={T_PASSES})",
    "mc_ent": f"MC-Dropout entropy   (T={T_PASSES})",
}
summary = {}
for method, name in METHOD_NAMES.items():
    tp_mean = float(np.mean(tp_results[method]))
    tp_std  = float(np.std(tp_results[method]))
    fp_mean = float(np.mean(fp_results[method]))
    fp_std  = float(np.std(fp_results[method]))
    summary[method] = {
        "tp_mean": tp_mean, "tp_std": tp_std,
        "fp_mean": fp_mean, "fp_std": fp_std,
    }
    print(f"{name:>30}  {tp_mean:.3f}±{tp_std:.3f}  {fp_mean:.3f}±{fp_std:.3f}")

print(f"\nCalibration: η_var={_ETA_VAR}, η_ent={_ETA_ENT}")

full_results = {
    "summary": summary,
    "calibration": {"eta_proto": ETA_PROTO,
                    "eta_var":   _ETA_VAR,
                    "eta_ent":   _ETA_ENT},
    "target_fp": target_fp,
    "n_pairs": len(samples),
    "T_passes": T_PASSES,
    "dropout_p": DROPOUT_P,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(full_results, f, indent=2)
print(f"\nN5 done. Results → {RESULTS_PATH}")
